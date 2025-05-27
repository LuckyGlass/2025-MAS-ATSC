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


def build_data(env_type: Literal['large_grid', 'real_net'], env_config_path: str):
    config = configparser.ConfigParser()
    config.read(env_config_path)
    data_path = config['ENV_CONFIG'].get('data_path')
    os.makedirs(data_path, exist_ok=True)
    if env_type == 'large_grid':
        build_large_grid(data_path)
    elif env_type == 'real_net':
        pass  # TODO
    else:
        raise ValueError(f"Unknown env_type `{env_type}`; supported types are `large_grid` and `real_net`.")


def load_env(env_type: Literal['large_grid', 'real_net'], env_config_path: str, base_dir: str, seed: int, port: int = 0, train_mode: bool = True):
    config = configparser.ConfigParser()
    config.read(env_config_path)
    if env_type == 'large_grid':
        env_cls = LargeGridEnv
    elif env_type == 'real_net':
        env_cls = RealNetEnv
    else:
        raise ValueError(f"Unknown env_type `{env_type}`; supported types are `large_grid` and `real_net`.")
    env = env_cls(config['ENV_CONFIG'], seed, port, base_dir)
    env.train_mode = train_mode
    return env
