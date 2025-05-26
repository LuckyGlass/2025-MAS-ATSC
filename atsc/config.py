import yaml
from dacite import from_dict
from dataclasses import dataclass, field
from typing import Literal, Type
from .envs.atsc_env import TrafficSimulator


@dataclass
class ATSCArguments:
    base_dir: str = field(default=None, metadata={'help': "The dir to save the models."})
    device: str = field(default='cuda')
    total_steps: int = field(default=1000000, metadata={'help': "The total number of sampling steps during training."})
    train_steps: int = field(default=1000, metadata={'help': "The interval (steps) of training."})
    max_episode_steps: int = field(default=100, metadata={'help': "The maximum number of steps in an episode during training."})
    gamma: float = field(default=0.95, metadata={'help': "The discount factor."})
    env_type: Literal['large_grid', 'real_net'] = field(default=None, metadata={'help': "The environment."})
    env_config_path: str = field(default=None, metadata={'help': "The path to the config file (.ini). Only the `[ENV_CONFIG]` part is used."})
    env_simulator_port: int = field(default=0, metadata={'help': "The port of the simulator. Keep the ports different when running two simulators simultaneously."})
    
    def init_from_env(self, env: TrafficSimulator):
        pass


def load_from_yaml(config_cls: Type, config_path: str) -> ATSCArguments:
    """Load ATSC config from a YAML file.
    Args:
        config_cls (`Type`):
            The class of the config to load, e.g., `IA2CArguments`.
        config_path (`str`):
            The path to the config (**YAML file**).
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return from_dict(data_class=config_cls, data=config_dict)
