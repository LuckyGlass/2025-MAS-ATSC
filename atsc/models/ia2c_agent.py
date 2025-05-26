"""
IA2C collection of agents, implemented with LSTM.
@author: Yansheng Mao
"""
import torch
from dataclasses import dataclass, field
from random import shuffle
from torch import nn
from torch.optim import AdamW
from typing import List, Optional
from .astc_agent import ATSCAgentCollection, ReplayBuffer
from .model_utils import SiluMLP
from ..config import ATSCArguments


@dataclass
class IA2CArguments(ATSCArguments):
    num_agents: int = field(default=None)
    observation_dims: List[int] = field(default=None)
    action_dims: List[int] = field(default=None)
    lstm_hidden_dim: int = field(default=None)
    policy_proj_hidden_dims: List[int] = field(default=None)
    value_proj_hidden_dims: List[int] = field(default=None)
    device: str = field(default='cuda')
    num_train_policy_epochs: int = field(default=1)
    num_train_value_epochs_per_policy_update: int = field(default=3)
    encoder_learning_rate: float = field(default=1e-4)
    policy_learning_rate: float = field(default=1e-4)
    value_learning_rate: float = field(default=1e-3)
    batch_size: int = field(default=200)
    regularization_scale: float = field(default=0.01)


class IA2CReplayBuffer(ReplayBuffer):
    """
    The replay buffer for on-policy LSTM-based IA2C agents.
    """
    def __init__(self):
        self.next_observation: List[List[torch.Tensor]] = []
        self.reward: List[torch.Tensor] = []
        self.observation: List[List[torch.Tensor]]
        self.action: List[torch.LongTensor] = []
        self.prev_hidden_state: List[List[torch.Tensor]] = []
        self.prev_cell_state: List[List[torch.Tensor]] = []
        self.size = 0

    def env_side(self, next_observation: List[torch.Tensor], reward: torch.Tensor, done: bool):
        self.next_observation.append(next_observation)
        self.reward.append(reward)
        self.size += 1

    def model_side(self, observation: List[torch.Tensor], action: List[int], prev_hidden_state: List[torch.Tensor], prev_cell_state: List[torch.Tensor]):
        self.observation.append(observation)
        self.action.append(torch.tensor(action, dtype=torch.long))
        self.prev_hidden_state.append(prev_hidden_state)
        self.prev_cell_state.append(prev_cell_state)
    
    def __len__(self) -> int:
        return self.size
    
    def rollout(self, batch_size: int = 1, do_shuffle: bool = True):
        num_agents = self.action[0].shape[-1]
        indices = list(range(len(self)))
        if do_shuffle:
            shuffle(indices)
        for j in range(0, self.size, batch_size):
            prev_hidden_state = [torch.stack([self.prev_hidden_state[i][k] for i in indices[j:j+batch_size]], dim=1) for k in range(num_agents)]
            prev_cell_state = [torch.stack([self.prev_cell_state[i][k] for i in indices[j:j+batch_size]], dim=1) for k in range(num_agents)]
            observation = [torch.stack([self.observation[i][k] for i in indices[j:j+batch_size]]) for k in range(num_agents)]
            next_observation = [torch.stack([self.next_observation[i][k] for i in indices[j:j+batch_size]]) for k in range(num_agents)]
            action = torch.stack([self.action[i] for i in indices[j:j+batch_size]])
            reward = torch.stack([self.reward[i] for i in indices[j:j+batch_size]])
            yield prev_hidden_state, prev_cell_state, observation, next_observation, action, reward


class IA2CAgents(ATSCAgentCollection):
    def __init__(self, args: IA2CArguments):
        self.device = args.device
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.num_agents = args.num_agents
        self.encoders = [nn.LSTM(observation_dim, self.lstm_hidden_dim, batch_first=True, device=self.device) for observation_dim in args.observation_dims]
        self.policy_projs = [nn.Sequential(SiluMLP(self.lstm_hidden_dim, args.policy_proj_hidden_dims, action_dim, device=self.device), nn.Softmax(dim=-1)) for action_dim in args.action_dims]
        self.value_projs = [SiluMLP(self.lstm_hidden_dim, args.value_proj_hidden_dims, 1, device=self.device) for _ in range(args.num_agents)]
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
            restored_observation.append(observation[i].detach())
            _, (self.lstm_hidden_states[i], self.lstm_cell_states[i]) = self.encoders[i].forward(observation[i], (prev_hidden_state, prev_cell_state))
            policy = self.policy_projs[i].forward(self.lstm_hidden_states[i])[0]  # Remove batch dim
            action.append(torch.distributions.Categorical(policy).sample(1).item())  # A scalar
        if replay_buffer is not None:
            replay_buffer.model_side(restored_observation, action, prev_hidden_state, prev_cell_state)
        return action

    def train_forward(self, prev_hidden_states: List[torch.Tensor], prev_cell_states: List[torch.Tensor], observations: List[torch.Tensor], next_observations: List[torch.Tensor], rewards: torch.Tensor, actions: torch.Tensor):
        """
        """
        policies, values, next_values = [], [], []
        for i in range(self.num_agents):
            prev_hidden_state, prev_cell_state, observation, next_observation = prev_hidden_states[i], prev_cell_states[i], observations[i], next_observations[i]
            series = torch.stack([observation, next_observation], dim=1)
            # hidden_state [batch, timeline, hiddendim]
            hidden_state, _ = self.encoders[i](series, (prev_hidden_state, prev_cell_state))
            policies.append(self.policy_forward[i](hidden_state[:, 0, :]))
            values.append(self.value_projs[i](hidden_state[:, 0, :])[:, 0])
            next_values.append(self.value_projs[i](hidden_state[:, 1, :]))
        policies = torch.stack(policies, dim=1)  # (batch, agent, action)
        values = torch.stack(values, dim=1)  # (batch, agent)
        next_values = torch.stack(next_values, dim=1)  # (batch, agent)
        return policies, values, next_values
    
    def train(self, replay_buffer: IA2CReplayBuffer, args: IA2CArguments):
        policy_side_optimizer = AdamW(
            [{'params': encoder.parameters(), 'lr': args.encoder_learning_rate} for encoder in self.encoders] +
            [{'params': policy_proj.parameters(), 'lr': args.policy_learning_rate} for policy_proj in self.policy_projs]
        )
        value_side_optimizer = AdamW(
            [{'params': encoder.parameters(), 'lr': args.encoder_learning_rate} for encoder in self.encoders] +
            [{'params': value_proj.parameters(), 'lr': args.value_learning_rate} for value_proj in self.value_projs]
        )
        value_side_loss_fn = torch.nn.MSELoss()
        for policy_epoch in range(args.num_train_policy_epochs):
            for prev_hidden_state, prev_cell_state, observation, next_observation, action, reward in replay_buffer.rollout(args.batch_size):
                for value_epoch in range(args.num_train_value_epochs_per_policy_update):
                    _, values, next_values = self.train_forward(prev_hidden_state, prev_cell_state, observation, next_observation, reward, action)
                    loss = value_side_loss_fn(values, reward + args.gamma * next_values.detach())
                    loss.backward()
                    value_side_optimizer.step()
                    value_side_optimizer.zero_grad()
                policies, values, next_values = self.train_forward(prev_hidden_state, prev_cell_state, observation, next_observation, reward, action)
                log_policies = torch.log(policies)
                advantages = reward + args.gamma * next_values - values
                log_action_prob = torch.gather(log_policies, action.unsqueeze(-1), dim=-1)[:, :, 0]
                loss = -torch.mean(log_action_prob * advantages) + args.regularization_scale * torch.mean(torch.sum(log_policies * policies, dim=-1))
                loss.backward()
                policy_side_optimizer.step()
                policy_side_optimizer.zero_grad()
