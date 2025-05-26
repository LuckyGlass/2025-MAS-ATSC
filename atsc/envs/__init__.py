"""
Utils to load environments.
@author: Yansheng Mao
"""
import configparser
import os
from typing import Literal
from .real_net_env import RealNetEnv
from .large_grid_env import LargeGridEnv
from .large_grid_data.build_file import build_large_grid


def load_env(env_type: Literal['large_grid', 'real_net'], env_config_path: str, base_dir: str, port: int = 0, train_mode: bool = True):
    config = configparser.ConfigParser()
    config.read(env_config_path)
    if env_type == 'large_grid':
        env_cls, builder_func = LargeGridEnv, build_large_grid
    elif env_type == 'real_net':
        env_cls, builder_func = RealNetEnv, None  # TODO
    else:
        raise ValueError(f"Unknown env_type `{env_type}`; supported types are `large_grid` and `real_net`.")
    data_path = config['ENV_CONFIG'].get('data_path')
    os.makedirs(data_path, exist_ok=True)
    builder_func(data_path)
    env = env_cls(config['ENV_CONFIG'], port, base_dir)
    env.train_mode = train_mode
    return env
