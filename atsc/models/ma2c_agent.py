from __future__ import annotations
import torch
from copy import deepcopy
from dataclasses import dataclass, field
from random import shuffle
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List
from .ia2c_agent import IA2CAgents, IA2CReplayBuffer, IA2CArguments
from ..envs.atsc_env import TrafficSimulator
from .model_utils import get_distance


@dataclass
class MA2CArguments(IA2CArguments):
    neighbor_link: List[List[int]] = field(default=None, metadata={'help': "The neighbors of each agent. Initiated through `init_from_env`."})
    distance: List[List[int]] = field(default=None, metadata={'help': "A square. `distance[i][j]` represents the distance between node i and node j."})
    neighbor_discount: float = field(default=0.75, metadata={'help': "The discount factor about the contribution of another agent to an agent."})

    def init_from_env(self, env: TrafficSimulator):
        super().init_from_env(env)
        self.observation_dims = deepcopy(env.n_s_ls)
        self.action_dims = deepcopy(env.n_a_ls)
        self.neighbor_link = []
        name2id = {name: i for i, name in enumerate(env.node_names)}
        for i, node_name in enumerate(env.node_names):
            node = env.nodes[node_name]
            neighbors = []
            for neighbor_name in node.neighbor:
                j = name2id[neighbor_name]
                neighbors.append(j)
                self.observation_dims[i] += self.action_dims[j]
            self.neighbor_link.append(neighbors)
        self.distance = get_distance(self.neighbor_link)


class MA2CReplayBuffer(IA2CReplayBuffer):
    def __init__(self):
        super().__init__()
        self.distance = None
    
    def rollout(self, device: str = 'cuda', batch_size: int = 1, do_shuffle: bool = True):
        self.distance = self.distance.to(device=device)
        num_agents = self.action[0].shape[-1]
        indices = list(range(len(self)))
        if do_shuffle:
            shuffle(indices)
        for j in range(0, self.size, batch_size):
            prev_hidden_state = [torch.stack([self.prev_hidden_state[i][k] for i in indices[j:j+batch_size]], dim=1).to(device) for k in range(num_agents)]
            prev_cell_state = [torch.stack([self.prev_cell_state[i][k] for i in indices[j:j+batch_size]], dim=1).to(device) for k in range(num_agents)]
            observation = [torch.stack([self.observation[i][k] for i in indices[j:j+batch_size]]).to(device) for k in range(num_agents)]
            next_observation = [torch.stack([self.next_observation[i][k] for i in indices[j:j+batch_size]]).to(device) for k in range(num_agents)]
            action = torch.stack([self.action[i] for i in indices[j:j+batch_size]]).to(device)
            reward = torch.stack([self.reward[i] for i in indices[j:j+batch_size]]).to(device)
            reward = reward @ self.distance.T
            yield prev_hidden_state, prev_cell_state, observation, next_observation, action, reward
    
    def set_distance(self, distance: torch.Tensor):
        self.distance = distance


class MA2CAgents(IA2CAgents):
    def __init__(self, args: MA2CArguments):
        super().__init__(args)
        self.distance_discount = args.neighbor_discount ** torch.tensor(args.distance, dtype=torch.float32, device=args.device)
    
    def export_state_dict(self) -> Dict[str, Any]:
        state_dict = super().export_state_dict()
        state_dict['distance_discount'] = self.distance_discount.clone()
        return state_dict
    
    @classmethod
    def load_state_dict(cls, args: IA2CArguments, state_dict: Dict[str, Any]):
        obj: MA2CAgents = cls(args)
        for encoder, s in zip(obj.encoders, state_dict['encoders_state_dict']):
            encoder.load_state_dict(s, strict=True)
            encoder.to(obj.device)
        for policy_proj, s in zip(obj.policy_projs, state_dict['policy_projs_state_dict']):
            policy_proj.load_state_dict(s, strict=True)
            policy_proj.to(obj.device)
        for value_proj, s in zip(obj.value_projs, state_dict['value_projs_state_dict']):
            value_proj.load_state_dict(s, strict=True)
            value_proj.to(obj.device)
        obj.distance_discount = state_dict['distance_discount'].clone().to(obj.device)
        return obj
    
    def train(self, replay_buffer: MA2CReplayBuffer, args: MA2CArguments, writer: SummaryWriter, global_steps: int):
        replay_buffer.set_distance(self.distance_discount)
        return super().train(replay_buffer, args, writer, global_steps)
