import argparse
import multiprocessing as mp
from atsc.train import train
from atsc.models import *
from atsc.config import load_from_yaml
from atsc.envs import load_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', choices=['IA2C-Queue', 'MA2C-Queue'])
    args = parser.parse_args()
    if args.exp_name == 'IA2C-Queue':
        config = load_from_yaml(IA2CArguments, 'config/ia2c_large_grid.yaml')
        # env is used only to initialize IA2CArguments
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True)
        config.init_from_env(env)
        agent = IA2CAgents(config)
        replay_buffer = IA2CReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'MA2C-Queue':
        config: MA2CArguments = load_from_yaml(MA2CArguments, 'config/ma2c_large_grid.yaml')
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = MA2CAgents(config)
        replay_buffer = MA2CReplayBuffer()
        train(config, agent, replay_buffer)


if __name__ == '__main__':
    # multiprocessing is used during exploration.
    # The maximum number of processes is controlled by argument `max_explore_processes`.
    mp.set_start_method('spawn')
    main()
