from __future__ import annotations
"""
Custom implementation of IC3Net (https://arxiv.org/abs/1812.09755).
IC3Net consists of a shared LSTM encoder, separate input projections, a shared gate, and separate policy projections.
It is a policy-based method without critic models.
@author: Yansheng Mao
"""
import numpy as np
import torch
from copy import deepcopy
from dataclasses import dataclass, field
from random import shuffle
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional, Tuple
from .atsc_agent import ATSCArguments, ATSCAgentCollection, ReplayBuffer
from .model_utils import SiluMLP
from ..envs.atsc_env import TrafficSimulator
from ..logger_utils import get_logger


logger = get_logger()


@dataclass
class IC3NetArguments(ATSCArguments):
    observation_dims: List[int] = field(default=None, metadata={'help': "The dimensions of the observations of the agents. Initiated via `init_from_env`."})
    action_dims: List[int] = field(default=None, metadata={'help': "The number of valid actions of the agents. Initiated via `init_from_env`."})
    lstm_hidden_dim: int = field(default=None, metadata={'help': "The hidden dimension of the shared LSTM encoder."})
    input_proj_hidden_dims: List[int] = field(default=None, metadata={'help': "The hidden dimensions of the separate observation encoders."})
    gate_proj_hidden_dims: List[int] = field(default=None, metadata={'help': "The hidden dimensions of the shared gate projection."})
    policy_proj_hidden_dims: List[int] = field(default=None, metadata={'help': "The hidden dimensions of the separate policy projections."})
    num_train_epochs: int = field(default=1, metadata={'help': "The number of epochs to train the agent."})
    batch_size: int = field(default=2000, metadata={'help': "The batch size for training."})
    learning_rate: float = field(default=1e-5, metadata={'help': "The learning rate for all modules."})
    weight_decay: float = field(default=.0, metadata={'help': "The weight decay for all modules."})
    max_grad_norm: Optional[float] = field(default=None, metadata={'help': "The maximum norm of the gradients."})
    
    def init_from_env(self, env: TrafficSimulator):
        super().init_from_env(env)
        self.observation_dims = deepcopy(env.n_s_ls)
        self.action_dims = deepcopy(env.n_a_ls)


class IC3NetReplayBuffer(ReplayBuffer):
    def __init__(self):
        self.observation: List[List[torch.Tensor]] = []
        self.gate: List[torch.Tensor] = []
        self.action: List[torch.LongTensor] = []
        self.prev_hidden_state: List[torch.Tensor] = []
        self.prev_cell_state: List[torch.Tensor] = []
        self.reward: List[torch.Tensor] = []
        self.done: List[bool] = []
        self.size = 0
        self.valid_indices = None

    def env_side(self, next_observation: List[torch.Tensor], reward: torch.Tensor, done: bool):
        self.reward.append(reward.detach().cpu().clone())
        self.done.append(done)
        self.size += 1
    
    def model_side(self, observation: List[torch.Tensor], gate: torch.Tensor, action: List[int], prev_hidden_state: torch.Tensor, prev_cell_state: torch.Tensor):
        self.observation.append([o.detach().cpu().clone() for o in observation])
        self.gate.append(gate.detach().cpu().clone())
        self.action.append(torch.tensor(action, dtype=torch.long, device='cpu'))
        self.prev_hidden_state.append(prev_hidden_state.detach().cpu().clone())
        self.prev_cell_state.append(prev_cell_state.detach().cpu().clone())
    
    def __len__(self) -> int:
        return self.size
    
    def extend(self, _replay_buffer: IC3NetReplayBuffer):
        self.observation.extend(_replay_buffer.observation)
        self.gate.extend(_replay_buffer.gate)
        self.action.extend(_replay_buffer.action)
        self.prev_hidden_state.extend(_replay_buffer.prev_hidden_state)
        self.prev_cell_state.extend(_replay_buffer.prev_cell_state)
        self.reward.extend(_replay_buffer.reward)
        self.done.extend(_replay_buffer.done)
        self.size += _replay_buffer.size

    def toggle_accumulate(self, gamma: float):
        sum_reward, sum_gamma, last_gamma = 0, 0, 1 / gamma
        self.valid_indices = []
        for i in range(self.size - 1, -1, -1):
            if self.done[i]:
                sum_reward, sum_gamma, last_gamma = 0, 0, 1 / gamma
            sum_reward = gamma * sum_reward + self.reward[i]
            sum_gamma = gamma * sum_gamma + 1
            last_gamma *= gamma
            if last_gamma < 0.1 * sum_gamma:
                self.valid_indices.append(i)
    
    def reset(self):
        del self.observation[:]
        del self.gate[:]
        del self.action[:]
        del self.prev_hidden_state[:]
        del self.prev_cell_state[:]
        del self.reward[:]
        del self.done[:]
        self.size = 0
        self.valid_indices = None
    
    def rollout(self, device: str = 'cuda', batch_size: int = 1):
        num_agents = len(self.observation[0])
        indices = deepcopy(self.valid_indices)
        shuffle(indices)
        for i in range(0, len(indices), batch_size):
            observation = [torch.stack([self.observation[k][j] for k in indices[i:i+batch_size]]).to(device) for j in range(num_agents)]
            gate = torch.stack([self.gate[k] for k in indices[i:i+batch_size]]).to(device)
            action = torch.stack([self.action[k] for k in indices[i:i+batch_size]]).to(device)
            prev_hidden_state = torch.concat([self.prev_hidden_state[k] for k in indices[i:i+batch_size]], dim=1).to(device)
            prev_cell_state = torch.concat([self.prev_cell_state[k] for k in indices[i:i+batch_size]], dim=1).to(device)
            reward = torch.stack([self.reward[k] for k in indices[i:i+batch_size]]).to(device)
            yield observation, gate, action, prev_hidden_state, prev_cell_state, reward


class IC3NetAgents(ATSCAgentCollection):
    def __init__(self, args: IC3NetArguments):
        self.num_agents = len(args.observation_dims)
        self.hidden_dim = args.lstm_hidden_dim
        self.device = args.device
        self.encoder = nn.LSTM(args.lstm_hidden_dim, args.lstm_hidden_dim, batch_first=True, device=args.device)
        self.input_projs = nn.ModuleList([SiluMLP(odim, args.input_proj_hidden_dims, args.lstm_hidden_dim, device=args.device) for odim in args.observation_dims])
        self.message_linear = nn.Linear(args.lstm_hidden_dim, args.lstm_hidden_dim, device=args.device, bias=False)
        self.gate_proj = nn.Sequential(
            SiluMLP(args.lstm_hidden_dim, args.gate_proj_hidden_dims, 1, device=args.device),
            nn.Sigmoid(),
        )
        self.policy_projs = nn.ModuleList([
            nn.Sequential(
                SiluMLP(args.lstm_hidden_dim, args.policy_proj_hidden_dims, adim, device=args.device),
                nn.Softmax(dim=-1)
            ) for adim in args.action_dims
        ])
        self.reset()
        
    def reset(self):
        self._lstm_hidden_states = torch.zeros((1, self.num_agents, self.hidden_dim), device=self.device)
        self._lstm_cell_states = torch.zeros((1, self.num_agents, self.hidden_dim), device=self.device)
    
    def forward(self, observation: List[torch.Tensor], replay_buffer: Optional[IC3NetReplayBuffer] = None, return_policy: bool = False) -> List[int] | Tuple[List[int], List[torch.Tensor]]:
        restored_hidden_states, restored_cell_states = self._lstm_hidden_states, self._lstm_cell_states
        gate_probs = self.gate_proj(self._lstm_hidden_states.squeeze(0)).squeeze(-1)
        gate = torch.bernoulli(gate_probs)
        messages = gate.unsqueeze(1) * self.message_linear(self._lstm_hidden_states.squeeze(0)) / (self.num_agents - 1)
        global_message = torch.sum(messages, dim=0, keepdim=True)
        messages = global_message - messages
        lstm_inputs = torch.stack([p(o) for o, p in zip(observation, self.input_projs)], dim=0) + messages
        lstm_inputs = lstm_inputs.unsqueeze(1)
        _, (self._lstm_hidden_states, self._lstm_cell_states) = self.encoder(lstm_inputs, (self._lstm_hidden_states, self._lstm_cell_states))
        actions, policies = [], []
        for i in range(self.num_agents):
            policy_i = self.policy_projs[i](self._lstm_hidden_states[0, i, :])
            action_i = torch.distributions.Categorical(policy_i).sample((1,))
            actions.append(action_i.item())
            policies.append(policy_i)
        if replay_buffer is not None:
            replay_buffer.model_side(observation, gate, actions, restored_hidden_states, restored_cell_states)
        return (actions, policies) if return_policy else actions
    
    def train(self, replay_buffer: IC3NetReplayBuffer, args: IC3NetArguments, writer: SummaryWriter, global_steps: int):
        replay_buffer.toggle_accumulate(args.gamma)
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.input_projs.parameters()) +
            list(self.message_linear.parameters()) + 
            list(self.gate_proj.parameters()) +
            list(self.policy_projs.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for _ in range(args.num_train_epochs):
            recorded_losses = []
            for o, g, a, h, c, r in replay_buffer.rollout(self.device, args.batch_size):
                optimizer.zero_grad()
                messages = g.unsqueeze(-1) * self.message_linear(h.squeeze(0)).reshape(-1, self.num_agents, self.hidden_dim)
                global_message = torch.sum(messages, dim=1, keepdim=True)
                messages = global_message - messages
                lstm_inputs = torch.stack([self.input_projs[i](o[i]) for i in range(self.num_agents)], dim=1) + messages
                gate_probs = self.gate_proj(h.squeeze(0)).reshape(-1, self.num_agents)
                gate_probs = torch.where(g.bool(), gate_probs, 1 - gate_probs)
                h1, _ = self.encoder(lstm_inputs.reshape(-1, 1, self.hidden_dim), (h, c))
                h1 = h1.reshape(-1, self.num_agents, self.hidden_dim)
                action_probs = torch.stack([torch.gather(self.policy_projs[i](h1[:, i, :]), 1, a[:, i, None]).squeeze(1) for i in range(self.num_agents)], dim=1)
                loss = torch.mean(- r * (torch.log(gate_probs) + torch.log(action_probs)))
                loss.backward()
                if args.max_grad_norm is not None:
                    clip_grad_norm_(self.encoder.parameters(), args.max_grad_norm)
                    clip_grad_norm_(self.input_projs.parameters(), args.max_grad_norm)
                    clip_grad_norm_(self.message_linear.parameters(), args.max_grad_norm)
                    clip_grad_norm_(self.gate_proj.parameters(), args.max_grad_norm)
                    clip_grad_norm_(self.policy_projs.parameters(), args.max_grad_norm)
                optimizer.step()
                recorded_losses.append(loss.item())
                del loss
        log_info = {
            'mean_loss': np.mean(recorded_losses),
        }
        logger.info(str(log_info))
        writer.add_scalar('train/loss', log_info['mean_loss'], global_steps)
        replay_buffer.reset()

    def export_state_dict(self) -> Dict[str, Any]:
        return {
            'e': self.encoder.state_dict(),
            'i': self.input_projs.state_dict(),
            'm': self.message_linear.state_dict(),
            'g': self.gate_proj.state_dict(),
            'p': self.policy_projs.state_dict(),
        }
    
    @classmethod
    def load_state_dict(cls, args: IC3NetArguments, state_dict: Dict[str, Any]) -> IC3NetAgents:
        obj: IC3NetAgents = cls(args)
        obj.encoder.load_state_dict(state_dict['e'])
        obj.encoder.to(obj.device)
        obj.input_projs.load_state_dict(state_dict['i'])
        obj.input_projs.to(obj.device)
        obj.message_linear.load_state_dict(state_dict['m'])
        obj.message_linear.to(obj.device)
        obj.gate_proj.load_state_dict(state_dict['g'])
        obj.gate_proj.to(obj.device)
        obj.policy_projs.load_state_dict(state_dict['p'])
        obj.policy_projs.to(obj.device)
        return obj
