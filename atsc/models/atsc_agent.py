"""
Agent framework (`ATSCAgentCollection`) for ATSC.
@author: Yansheng Mao
"""
import torch
from abc import abstractmethod, ABC
from typing import List, Optional


class ReplayBuffer(ABC):
    @abstractmethod
    def env_side(self, next_observation: List[torch.Tensor], reward: torch.Tensor, done: bool):
        pass

    @abstractmethod
    def model_side(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        pass


class ATSCAgentCollection(ABC):
    """
    The agent framework for ATSC.
    For multi-agent methods, it is the collection of all the agents.
    For centralized training methods, it is the centralized agent.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, observation: List[torch.Tensor], replay_buffer: Optional[ReplayBuffer] = None) -> List[int]:
        """
        Decide the next actions given the current observations and the termination flag.
        Args:
            observation (`List[Tensor]`):
                The list of observations from each agents.
        Returns:
            actions (`List[int]`): The actions of the agents.
        """
    
    @abstractmethod
    def train(self, replay_buffer: ReplayBuffer, args):
        pass
