from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ATSCArguments:
    total_steps: int = field(default=1000000, metadata={'help': "The total number of sampling steps during training."})
    train_steps: int = field(default=1000, metadata={'help': "The interval (steps) of training."})
    gamma: float = field(default=0.95, metadata={'help': "The discount factor."})
    env_type: Literal['large_grid', 'real_net'] = field(default=None, metadata={'help': "The environment."})
    env_config_path: str = field(default=None, metadata={'help': "The path to the config file (.ini). Only the `[ENV_CONFIG]` part is used."})
