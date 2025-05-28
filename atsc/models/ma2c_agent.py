from __future__ import annotations
import torch
from copy import deepcopy
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional
from .ia2c_agent import IA2CAgents, IA2CReplayBuffer, IA2CArguments
from ..envs.atsc_env import TrafficSimulator


def get_distance(neighbor_link: List[List[int]]):
    """Compute the shortest distance between each node pair using Floyd algorithm. The edge weights are set to 1.
    Args:
        neighbor_link (`List[List[int]]`):
            `neighbor_link[i]` represents the neighbors (index) of node i.
    Returns:
        distance (`List[List[int]]`):
            A square. `distance[i][j]` represents the distance between node i and node j.
    """
    num_nodes = len(neighbor_link)
    distance = [[num_nodes for j in range(num_nodes)] for i in range(num_nodes)]  # initiate with INF
    for i in range(num_nodes):
        distance[i][i] = 0
        for j in neighbor_link[i]:
            distance[i][j] = 1
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance


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
    """
    Exactly the same as `IA2CReplayBuffer`.
    """


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
        replay_buffer.reward = [self.distance_discount @ reward for reward in replay_buffer.reward]
        return super().train(replay_buffer, args, writer, global_steps)
