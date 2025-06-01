from __future__ import annotations
"""
Extend SAC to multi-agent scenarios, similar to SA2C.
For simplicity, it is implemented as an on-policy algorithm.
@author: Yansheng Mao
"""
import numpy as np
import torch
from copy import deepcopy
from dataclasses import dataclass, field
from random import shuffle
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional, Tuple
from .atsc_agent import ATSCArguments, ATSCAgentCollection, ReplayBuffer
from .model_utils import SiluMLP, ConstantRescale, get_distance
from ..envs.atsc_env import TrafficSimulator
from ..logger_utils import get_logger


logger = get_logger()


@dataclass
class MSACArguments(ATSCArguments):
    observation_dims: List[int] = field(default=None, metadata={'help': "The dimensions of observations of all the agents, including its own observation and the fingerprints (policies at the last time step) of the neighbors. Initiated via `init_from_env`."})
    action_dims: List[int] = field(default=None, metadata={'help': "The number of valid actions of all the agents. Initiated via `init_from_env`."})
    distance: List[List[int]] = field(default=None, metadata={'help': "The distance between each pair of agents. Initiated via `init_from_env`."})
    neighbor_discount: float = field(default=0.75, metadata={'help': "The discount factor about the contribution of another agent to an agent."})
    lstm_hidden_dim: int = field(default=None, metadata={'help': "The dimension of the hidden states of LSTM encoders."})
    critic_hidden_dims: List[int] = field(default=None, metadata={'help': "The hidden dimensions of critic MLPs."})
    policy_hidden_dims: List[int] = field(default=None, metadata={'help': "The hidden dimensions of policy MLPs."})
    device: str = field(default='cuda', metadata={'help': "The device of MSACAgent."})
    batch_size: int = field(default=256, metadata={'help': "The batch size during training. Notice it is not the number of steps in an exploration phase!"})
    num_policy_train_epochs: int = field(default=1, metadata={'help': "The number of epochs in training the policy network."})
    num_critic_train_epochs: int = field(default=2, metadata={'help': "The number of epochs in training the critic network."})
    encoder_learning_rate: float = field(default=1e-6, metadata={'help': "The learning rate to train the encoders. It is recommended to use a small learning rate."})
    policy_learning_rate: float = field(default=1e-5, metadata={'help': "The learning rate to train the policy networks."})
    critic_learning_rate: float = field(default=5e-5, metadata={'help': "The learning rate to train the critic networks. It is recommended to use a larger learning rate for critic than that for policy."})
    weight_decay: float = field(default=.0, metadata={'help': "The weight decay for training the networks."})
    max_grad_norm: Optional[float] = field(default=None, metadata={'help': "The maximum grad norm during updating."})
    
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


class MSACReplayBuffer(ReplayBuffer):
    def __init__(self):
        self.last_hidden_state: List[List[torch.Tensor]] = []
        self.last_cell_state: List[List[torch.Tensor]] = []
        self.observation: List[List[torch.Tensor]] = []
        self.action: List[List[int]] = []
        self.next_observation: List[List[torch.Tensor]] = []
        self.reward: List[torch.Tensor] = []
        self.size = 0
        self.distance_discount = None
    
    def env_side(self, next_observation: List[torch.Tensor], reward: torch.Tensor, done: bool):
        self.next_observation.append([n.unsqueeze(0).detach().cpu() for n in next_observation])
        self.reward.append(reward.detach().cpu())
        self.size += 1

    def model_side(self, last_hidden_state: List[torch.Tensor], last_cell_state: List[torch.Tensor], observation: List[torch.Tensor], action: List[int]):
        self.last_hidden_state.append([l.detach().cpu() for l in last_hidden_state])
        self.last_cell_state.append([l.detach().cpu() for l in last_cell_state])
        self.observation.append([l.detach().cpu() for l in observation])
        self.action.append(torch.tensor(action, dtype=torch.long, device='cpu'))

    def __len__(self) -> int:
        return self.size

    def extend(self, _replay_buffer: MSACReplayBuffer):
        self.last_hidden_state.extend(_replay_buffer.last_hidden_state)
        self.last_cell_state.extend(_replay_buffer.last_cell_state)
        self.observation.extend(_replay_buffer.observation)
        self.action.extend(_replay_buffer.action)
        self.next_observation.extend(_replay_buffer.next_observation)
        self.reward.extend(_replay_buffer.reward)
        self.size += _replay_buffer.size
    
    def rollout(self, batch_size: int, device: str = 'cuda', do_shuffle: bool = True):
        num_agents = self.action[0].shape[-1]
        indices = list(range(len(self)))
        if do_shuffle:
            shuffle(indices)
        for j in range(0, self.size, batch_size):
            last_hidden_state = [torch.stack([self.last_hidden_state[i][k] for i in indices[j:j+batch_size]], dim=1).to(device) for k in range(num_agents)]
            last_cell_state = [torch.stack([self.last_cell_state[i][k] for i in indices[j:j+batch_size]], dim=1).to(device) for k in range(num_agents)]
            observation = [torch.stack([self.observation[i][k] for i in indices[j:j+batch_size]]).to(device) for k in range(num_agents)]
            next_observation = [torch.stack([self.next_observation[i][k] for i in indices[j:j+batch_size]]).to(device) for k in range(num_agents)]
            action = torch.stack([self.action[i] for i in indices[j:j+batch_size]]).to(device)
            reward = torch.stack([self.reward[i] for i in indices[j:j+batch_size]]).to(device)
            reward = reward @ self.distance_discount.T
            yield last_hidden_state, last_cell_state, observation, next_observation, action, reward

    def reset(self):
        del self.last_hidden_state[:]
        del self.last_cell_state[:]
        del self.observation[:]
        del self.action[:]
        del self.next_observation[:]
        del self.reward[:]
        self.size = 0

    def set_distance_discount(self, distance_discount: torch.Tensor):
        self.distance_discount = distance_discount


class MSACAgents(ATSCAgentCollection):
    def __init__(self, args: MSACArguments):
        # Config
        self.device = args.device
        self.hidden_dim = args.lstm_hidden_dim
        self.num_agents = len(args.observation_dims)
        self.distance_discount = args.neighbor_discount ** torch.tensor(args.distance, dtype=torch.float32, device=args.device)
        # Modules
        self.encoders = nn.ModuleList([nn.LSTM(odim, args.lstm_hidden_dim, 1, batch_first=True, device=self.device) for odim in args.observation_dims])
        self.v_projs = nn.ModuleList([
            nn.Sequential(
                SiluMLP(args.lstm_hidden_dim, args.critic_hidden_dims, 1, device=self.device),
                ConstantRescale(1000, -1)
            ) for _ in range(self.num_agents)
        ])
        self.q_projs = nn.ModuleList([
            nn.Sequential(
                SiluMLP(args.lstm_hidden_dim, args.critic_hidden_dims, adim, device=self.device),
                ConstantRescale(1000, -1)
            ) for adim in args.action_dims
        ])
        self.policy_projs = nn.ModuleList([
            nn.Sequential(
                SiluMLP(args.lstm_hidden_dim, args.policy_hidden_dims, adim, device=self.device),
                nn.Softmax(dim=-1)
            ) for adim in args.action_dims
        ])
        # Init
        self.reset()
    
    def forward(self, observation: List[torch.Tensor], replay_buffer: Optional[MSACReplayBuffer] = None, return_policy: bool = False) -> List[int] | Tuple[List[int], List[torch.Tensor]]:
        restored_last_hidden_state = self._lstm_hidden_states
        restored_last_cell_state = self._lstm_cell_states
        restored_observation = []
        new_hidden_states, new_cell_states = [], []
        actions, policies = [], []
        for o, encoder, policy_proj, last_hidden_state, last_cell_state in zip(observation, self.encoders, self.policy_projs, self._lstm_hidden_states, self._lstm_cell_states):
            o = o.unsqueeze(0)  # Add time dim
            restored_observation.append(o)
            _, (hidden_state, cell_state) = encoder(o, (last_hidden_state, last_cell_state))
            new_hidden_states.append(hidden_state)
            new_cell_states.append(cell_state)
            policy = policy_proj(hidden_state)[0]
            policies.append(policy)
            action = Categorical(policy).sample((1,)).item()
            actions.append(action)
        self._lstm_hidden_states = new_hidden_states
        self._lstm_cell_states = new_cell_states
        if replay_buffer is not None:
            replay_buffer.model_side(restored_last_hidden_state, restored_last_cell_state, restored_observation, actions)
        return (actions, policies) if return_policy else actions

    def train(self, replay_buffer: MSACReplayBuffer, args: MSACArguments, writer: SummaryWriter, global_steps: int):
        replay_buffer.set_distance_discount(self.distance_discount)
        policy_side_optimizer = torch.optim.AdamW(
            [
                {'params': self.encoders.parameters(), 'lr': args.encoder_learning_rate},
                {'params': self.policy_projs.parameters(), 'lr': args.policy_learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        critic_side_optimizer = torch.optim.AdamW(
            [
                {'params': self.encoders.parameters(), 'lr': args.encoder_learning_rate},
                {'params': self.v_projs.parameters(), 'lr': args.critic_learning_rate},
                {'params': self.q_projs.parameters(), 'lr': args.critic_learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        past_v_projs = deepcopy(self.v_projs)
        past_q_projs = deepcopy(self.q_projs)
        for _ in range(args.num_critic_train_epochs):
            recorded_v_loss, recorded_q_loss = [], []
            for last_hidden_state, last_cell_state, observation, next_observation, action, reward in replay_buffer.rollout(args.batch_size, device=self.device):
                critic_side_optimizer.zero_grad()
                observation = [torch.concat((o, next_o), dim=1) for o, next_o in zip(observation, next_observation)]
                sum_loss_q, sum_loss_v = 0, 0
                for i in range(self.num_agents):
                    output, _ = self.encoders[i](observation[i], (last_hidden_state[i], last_cell_state[i]))
                    log_prob = torch.log(torch.gather(self.policy_projs[i](output[:, 0, :]), -1, action[:, i].unsqueeze(-1)).squeeze(-1) + 1e-8)
                    target_q = (reward[:, i] + args.gamma * past_v_projs[i](output[:, 1, :]).squeeze(-1)).detach()
                    pred_q = torch.gather(self.q_projs[i](output[:, 0, :]), -1, action[:, i].unsqueeze(-1)).squeeze(-1)
                    loss_q = nn.functional.mse_loss(pred_q, target_q) / self.num_agents
                    target_v = torch.gather(past_q_projs[i](output[:, 0, :]), -1, action[:, i].unsqueeze(-1)).squeeze(-1) - log_prob
                    pred_v = self.v_projs[i](output[:, 0, :]).squeeze(-1)
                    loss_v = nn.functional.mse_loss(pred_v, target_v) / self.num_agents
                    (loss_v + loss_q).backward()  # Jointly backward to prevent retaining compute-graph issue.
                    sum_loss_q += loss_q.item()
                    sum_loss_v += loss_v.item()
                    del loss_v, loss_q
                if args.max_grad_norm is not None:
                    clip_grad_norm_(self.encoders.parameters(), args.max_grad_norm)
                    clip_grad_norm_(self.v_projs.parameters(), args.max_grad_norm)
                    clip_grad_norm_(self.q_projs.parameters(), args.max_grad_norm)
                critic_side_optimizer.step()
                recorded_q_loss.append(sum_loss_q)
                recorded_v_loss.append(sum_loss_v)
        del past_v_projs, past_q_projs
        policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        for _ in range(args.num_policy_train_epochs):
            recorded_policy_loss = []
            for last_hidden_state, last_cell_state, observation, _, _, _ in replay_buffer.rollout(args.batch_size, device=self.device):
                policy_side_optimizer.zero_grad()
                sum_loss = 0
                for i in range(self.num_agents):
                    output, _ = self.encoders[i](observation[i], (last_hidden_state[i], last_cell_state[i]))
                    target_policy = nn.functional.softmax(self.q_projs[i](output[:, 0, :]), dim=-1).detach()
                    pred_policy = torch.log(self.policy_projs[i](output[:, 0, :]) + 1e-8)
                    loss = policy_loss_fn(pred_policy, target_policy) / self.num_agents
                    loss.backward()
                    sum_loss += loss.item()
                    del loss
                if args.max_grad_norm is not None:
                    clip_grad_norm_(self.encoders.parameters(), args.max_grad_norm)
                    clip_grad_norm_(self.policy_projs.parameters(), args.max_grad_norm)
                policy_side_optimizer.step()
                recorded_policy_loss.append(sum_loss)
        log_info = {
            'mean_q_loss': np.mean(recorded_q_loss),
            'mean_v_loss': np.mean(recorded_v_loss),
            'mean_policy_loss': np.mean(recorded_policy_loss),
        }
        writer.add_scalar('train/q_loss', log_info['mean_q_loss'], global_steps)
        writer.add_scalar('train/v_loss', log_info['mean_v_loss'], global_steps)
        writer.add_scalar('train/policy_loss', log_info['mean_policy_loss'], global_steps)
        logger.info(str(log_info))
        replay_buffer.reset()

    def export_state_dict(self) -> Dict[str, Any]:
        state_dict = {
            'encoders': self.encoders.state_dict(),
            'policy_projs': self.policy_projs.state_dict(),
            'v_projs': self.v_projs.state_dict(),
            'q_projs': self.q_projs.state_dict(),
            'distance_discount': self.distance_discount.clone().cpu(),
        }
        return state_dict

    @classmethod
    def load_state_dict(cls, args: MSACArguments, state_dict: Dict[str, Any]) -> MSACAgents:
        obj: MSACAgents = cls(args)
        obj.encoders.load_state_dict(state_dict['encoders'], strict=True)
        obj.encoders.to(obj.device)
        obj.policy_projs.load_state_dict(state_dict['policy_projs'], strict=True)
        obj.policy_projs.to(obj.device)
        obj.v_projs.load_state_dict(state_dict['v_projs'], strict=True)
        obj.v_projs.to(obj.device)
        obj.q_projs.load_state_dict(state_dict['q_projs'], strict=True)
        obj.q_projs.to(obj.device)
        obj.distance_discount = state_dict['distance_discount'].to(obj.device)
        return obj

    def reset(self):
        # Notice that there is a layer dim (1)
        self._lstm_hidden_states = [torch.zeros((1, self.hidden_dim), dtype=torch.float32, device=self.device) for _ in range(self.num_agents)]
        self._lstm_cell_states = [torch.zeros((1, self.hidden_dim), dtype=torch.float32, device=self.device) for _ in range(self.num_agents)]
