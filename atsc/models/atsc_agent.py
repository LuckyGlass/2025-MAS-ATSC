from __future__ import annotations
"""
Agent framework (`ATSCAgentCollection`) for ATSC.
@author: Yansheng Mao
"""
import torch
from abc import abstractmethod, ABC
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional
from ..config import ATSCArguments


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

    @abstractmethod
    def extend(self, _replay_buffer: ReplayBuffer):
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
    def train(self, replay_buffer: ReplayBuffer, args: ATSCArguments, writer: SummaryWriter, global_steps: int):
        pass
    
    @abstractmethod
    def export_state_dict(self) -> Dict[str, Any]:
        """
        Export a dictionary, storing all the necessary data to recover the agent (collection).\\
        The torch modules are stored as state_dict.\\
        The exported dictionary can be loaded by load_state_dict.\\
        This is used when saving the final model or starting parallel exploration.
        """

    @classmethod
    @abstractmethod
    def load_state_dict(cls, args: ATSCArguments, state_dict: Dict[str, Any]) -> ATSCAgentCollection:
        """
        Load the state dict. Refer to `ATSCAgentCollection.export_state_dict`.
        """

    def reset(self):
        """
        Some agent collections, such as `ExtendedIA2CAgents`, have hidden states tracking the current episode.
        `.reset` is called at the start of each episode to reset the hidden states. By default, it does nothing.
        One should rewrite this method to reset the hidden states.
        """
