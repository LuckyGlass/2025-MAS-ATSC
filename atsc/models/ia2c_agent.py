from __future__ import annotations
"""
IA2C collection of agents, implemented with LSTM.
@author: Yansheng Mao
"""
import numpy as np
import torch
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from random import shuffle
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional
from .atsc_agent import ATSCAgentCollection, ReplayBuffer
from .model_utils import SiluMLP
from ..config import ATSCArguments
from ..envs.atsc_env import TrafficSimulator
from ..logger_utils import get_logger


logger = get_logger()


@dataclass
class IA2CArguments(ATSCArguments):
    num_agents: int = field(default=None, metadata={'help': "The number of agents in the collection. It is initiated by calling `init_from_env`."})
    observation_dims: List[int] = field(default=None, metadata={'help': "The dimension of observations of the agents. It is initiated by calling `init_from_env`."})
    action_dims: List[int] = field(default=None, metadata={'help': "The number of valid actions (phases) of the agents. It is initiated by calling `init_from_env`."})
    lstm_hidden_dim: int = field(default=None)
    policy_proj_hidden_dims: List[int] = field(default=None)
    value_proj_hidden_dims: List[int] = field(default=None)
    num_train_policy_epochs: int = field(default=1)
    num_train_value_epochs: int = field(default=3)
    weight_decay: float = field(default=.0)
    max_grad_norm: Optional[float] = field(default=None)
    encoder_learning_rate: float = field(default=1e-4)
    policy_learning_rate: float = field(default=1e-4)
    value_learning_rate: float = field(default=1e-3)
    batch_size: int = field(default=200)
    regularization_scale: float = field(default=0.01)
    
    def init_from_env(self, env: TrafficSimulator):
        super().init_from_env(env)
        self.num_agents = len(env.nodes)
        self.observation_dims = env.n_s_ls
        self.action_dims = env.n_a_ls


class IA2CReplayBuffer(ReplayBuffer):
    """
    The replay buffer for on-policy LSTM-based IA2C agents.
    """
    def __init__(self):
        self.next_observation: List[List[torch.Tensor]] = []
        self.reward: List[torch.Tensor] = []
        self.observation: List[List[torch.Tensor]] = []
        self.action: List[torch.LongTensor] = []
        self.prev_hidden_state: List[List[torch.Tensor]] = []
        self.prev_cell_state: List[List[torch.Tensor]] = []
        self.size = 0
    
    def reset(self):
        del self.next_observation[:], self.reward[:], self.observation[:], self.action[:], self.prev_cell_state[:], self.prev_hidden_state[:]
        self.size = 0

    def env_side(self, next_observation: List[torch.Tensor], reward: torch.Tensor, done: bool):
        self.next_observation.append([o.unsqueeze(0).cpu() for o in next_observation])
        self.reward.append(reward.cpu())
        self.size += 1

    def model_side(self, observation: List[torch.Tensor], action: List[int], prev_hidden_state: List[torch.Tensor], prev_cell_state: List[torch.Tensor]):
        self.observation.append([o.cpu() for o in observation])
        self.action.append(torch.tensor(action, dtype=torch.long, device='cpu'))
        self.prev_hidden_state.append([p.cpu() for p in prev_hidden_state])
        self.prev_cell_state.append([p.cpu() for p in prev_cell_state])
    
    def extend(self, _replay_buffer: IA2CReplayBuffer):
        self.next_observation.extend(_replay_buffer.next_observation)
        self.reward.extend(_replay_buffer.reward)
        self.observation.extend(_replay_buffer.observation)
        self.action.extend(_replay_buffer.action)
        self.prev_hidden_state.extend(_replay_buffer.prev_hidden_state)
        self.prev_cell_state.extend(_replay_buffer.prev_cell_state)
        self.size += _replay_buffer.size
    
    def __len__(self) -> int:
        return self.size
    
    def rollout(self, device: str = 'cuda', batch_size: int = 1, do_shuffle: bool = True):
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
            yield prev_hidden_state, prev_cell_state, observation, next_observation, action, reward


class IA2CValueProj(nn.Module):
    """
    To scale the output.
    """
    def __init__(self, lstm_hidden_dim: int, mlp_hidden_dims: List[int], device: str = 'cuda'):
        super().__init__()
        self.proj = SiluMLP(lstm_hidden_dim, mlp_hidden_dims, 1, device=device)
    
    def forward(self, x: torch.Tensor):
        return (self.proj(x)[..., 0] - 5) * 100


class IA2CAgents(ATSCAgentCollection):
    r"""
    The naive IA2C agents. Each agent has memory about the previous observations, encoded with LSTM, i.e., the states
    are $h_{t}^{(i)}$. The update rule is $h_{t}^{(i)}=LSTM(h_{t-1}^{(i)},o_{t}^{(i)};\theta_{t}^{(i)})$. The policy is
    $\pi_{\phi_{i}}(a|h_{t}^{(i)})$ and the \\
    critic is $V_{\rho_{i}}(h_{t}^{(i)})$.
    
    Notice that naive IA2C does not include neighbors' policies within the observation of an agent.
    """
    def __init__(self, args: IA2CArguments):
        self.device = args.device
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.num_agents = args.num_agents
        self.encoders = [nn.LSTM(observation_dim, self.lstm_hidden_dim, batch_first=True, device=self.device) for observation_dim in args.observation_dims]
        self.policy_projs = [nn.Sequential(SiluMLP(self.lstm_hidden_dim, args.policy_proj_hidden_dims, action_dim, device=self.device), nn.Softmax(dim=-1),    ) for action_dim in args.action_dims]
        self.value_projs = [IA2CValueProj(self.lstm_hidden_dim, args.value_proj_hidden_dims, device=self.device) for _ in range(args.num_agents)]
        self.target_value_projs = []
        self.reset()

    def reset(self):
        self.lstm_hidden_states = [torch.zeros((1, self.lstm_hidden_dim), device=self.device) for _ in range(self.num_agents)]
        self.lstm_cell_states = [torch.zeros((1, self.lstm_hidden_dim), device=self.device) for _ in range(self.num_agents)]
    
    def forward(self, observation: List[torch.Tensor], replay_buffer: Optional[IA2CReplayBuffer] = None) -> List[int]:
        prev_hidden_state, prev_cell_state, action = [], [], []
        restored_observation = []
        for i in range(self.num_agents):
            prev_hidden_state.append(self.lstm_hidden_states[i].detach())
            prev_cell_state.append(self.lstm_cell_states[i].detach())
            restored_observation.append(observation[i].detach().unsqueeze(0))
            _, (self.lstm_hidden_states[i], self.lstm_cell_states[i]) = self.encoders[i].forward(observation[i].unsqueeze(0), (self.lstm_hidden_states[i], self.lstm_cell_states[i]))
            policy = self.policy_projs[i].forward(self.lstm_hidden_states[i])[0]  # Remove batch dim
            action.append(torch.distributions.Categorical(policy).sample([1]).item())  # A scalar
        if replay_buffer is not None:
            replay_buffer.model_side(restored_observation, action, prev_hidden_state, prev_cell_state)
        return action
    
    def export_state_dict(self) -> Dict[str, Any]:
        state_dict = {
            'encoders_state_dict': [e.state_dict() for e in self.encoders],
            'policy_projs_state_dict': [p.state_dict() for p in self.policy_projs],
            'value_projs_state_dict': [v.state_dict() for v in self.value_projs],
        }
        return state_dict
    
    @classmethod
    def load_state_dict(cls, args: IA2CArguments, state_dict: Dict[str, Any]):
        obj: IA2CAgents = cls(args)
        for encoder, s in zip(obj.encoders, state_dict['encoders_state_dict']):
            encoder.load_state_dict(s, strict=True)
            encoder.to(obj.device)
        for policy_proj, s in zip(obj.policy_projs, state_dict['policy_projs_state_dict']):
            policy_proj.load_state_dict(s, strict=True)
            policy_proj.to(obj.device)
        for value_proj, s in zip(obj.value_projs, state_dict['value_projs_state_dict']):
            value_proj.load_state_dict(s, strict=True)
            value_proj.to(obj.device)
        return obj

    def train_forward(self, prev_hidden_states: List[torch.Tensor], prev_cell_states: List[torch.Tensor], observations: List[torch.Tensor], next_observations: List[torch.Tensor], rewards: torch.Tensor, actions: torch.Tensor):
        policies, values, next_values = [], [], []
        for i in range(self.num_agents):
            prev_hidden_state, prev_cell_state, observation, next_observation = prev_hidden_states[i], prev_cell_states[i], observations[i], next_observations[i]
            series = torch.concat([observation, next_observation], dim=1)
            # hidden_state [batch, timeline, hiddendim]
            hidden_state, _ = self.encoders[i](series, (prev_hidden_state, prev_cell_state))
            policies.append(self.policy_projs[i](hidden_state[:, 0, :]))
            values.append(self.value_projs[i](hidden_state[:, 0, :]))
            next_values.append(self.target_value_projs[i](hidden_state[:, 1, :]))
        policies = torch.stack(policies, dim=1)  # (batch, agent, action)
        values = torch.stack(values, dim=1)  # (batch, agent)
        next_values = torch.stack(next_values, dim=1)  # (batch, agent)
        return policies, values, next_values
    
    def train(self, replay_buffer: IA2CReplayBuffer, args: IA2CArguments, writer: SummaryWriter, global_steps: int):
        encoder_parameters = [p for m in self.encoders for p in m.parameters()]
        policy_parameters = [p for m in self.policy_projs for p in m.parameters()]
        value_parameters = [p for m in self.value_projs for p in m.parameters()]
        policy_side_optimizer = AdamW(
            [
                {'params': encoder_parameters, 'lr': args.encoder_learning_rate},
                {'params': policy_parameters, 'lr': args.policy_learning_rate}
            ],
            weight_decay=args.weight_decay,
        )
        value_side_optimizer = AdamW(
            [
                {'params': encoder_parameters, 'lr': args.encoder_learning_rate},
                {'params': value_parameters, 'lr': args.value_learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        value_side_loss_fn = torch.nn.MSELoss()
        # Train critic
        self.target_value_projs = deepcopy(self.value_projs)
        for value_epoch in range(args.num_train_value_epochs):
            value_losses = []
            for prev_hidden_state, prev_cell_state, observation, next_observation, action, reward in replay_buffer.rollout(batch_size=args.batch_size, device=args.device):
                _, values, next_values = self.train_forward(prev_hidden_state, prev_cell_state, observation, next_observation, reward, action)
                loss = value_side_loss_fn(values, reward + args.gamma * next_values.detach())
                value_losses.append(loss.item())
                loss.backward()
                if args.max_grad_norm is not None:
                    clip_grad_norm_(encoder_parameters + value_parameters, args.max_grad_norm)
                value_side_optimizer.step()
                value_side_optimizer.zero_grad()
        del self.target_value_projs[:]
        self.target_value_projs = deepcopy(self.value_projs)
        # Train policy
        for policy_epoch in range(args.num_train_policy_epochs):
            policy_losses = []
            mean_values = []
            mean_advantages = []
            for prev_hidden_state, prev_cell_state, observation, next_observation, action, reward in replay_buffer.rollout(batch_size=args.batch_size, device=args.device):
                policies, values, next_values = self.train_forward(prev_hidden_state, prev_cell_state, observation, next_observation, reward, action)
                log_policies = torch.log(policies)
                advantages = (reward + args.gamma * next_values - values).detach()
                log_action_prob = torch.gather(log_policies, index=action.unsqueeze(-1).to(self.device), dim=-1)[:, :, 0]
                loss = -torch.mean(log_action_prob * advantages) - args.regularization_scale * torch.mean(torch.sum(log_policies * policies, dim=-1))
                policy_losses.append(loss.item())
                mean_values.append(torch.mean(values).item())
                mean_advantages.append(torch.mean(advantages).item())
                loss.backward()
                if args.max_grad_norm is not None:
                    clip_grad_norm_(encoder_parameters + policy_parameters, args.max_grad_norm)
                policy_side_optimizer.step()
                policy_side_optimizer.zero_grad()
        del self.target_value_projs[:]
        train_log = {
            'mean_policy_loss': np.mean(policy_losses),
            'mean_value_loss': np.mean(value_losses),
            'mean_value': np.mean(mean_values),
            'mean_advantage': np.mean(mean_advantages),
        }
        writer.add_scalar('train/policy_loss', train_log['mean_policy_loss'], global_steps)
        writer.add_scalar('train/value_loss', train_log['mean_value_loss'], global_steps)
        writer.add_scalar('train/value', train_log['mean_value'], global_steps)
        writer.add_scalar('train/advantage', train_log['mean_advantage'], global_steps)
        logger.info(str(train_log))
        replay_buffer.reset()
