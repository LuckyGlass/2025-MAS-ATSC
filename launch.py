import argparse
import multiprocessing as mp
from atsc.train import train
from atsc.models import *
from atsc.config import load_from_yaml
from atsc.envs import load_env, build_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', choices=['IA2C-Queue', 'IA2C-Queue-Real', 'MA2C-Queue', 'MA2C-Queue-Real', 'MSAC-Queue', 'MSAC-Queue-Real', 'IC3Net-Queue', 'IC3Net-Queue-Real',
                                             'IA2C-Wait',  'IA2C-Wait-Real',  'MA2C-Wait',  'MA2C-Wait-Real',  'MSAC-Wait',  'MSAC-Wait-Real',  'IC3Net-Wait',  'IC3Net-Wait-Real'])
    args = parser.parse_args()
    if args.exp_name == 'IA2C-Queue':
        config = load_from_yaml(IA2CArguments, 'config/ia2c_large_grid.yaml')
        build_data(config.env_type, config.env_config_path)
        # env is used only to initialize IA2CArguments
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True)
        config.init_from_env(env)
        agent = IA2CAgents(config)
        replay_buffer = IA2CReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'IA2C-Queue-Real':
        config = load_from_yaml(IA2CArguments, 'config/ia2c_real_net.yaml')
        build_data(config.env_type, config.env_config_path)
        # env is used only to initialize IA2CArguments
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True)
        config.init_from_env(env)
        agent = IA2CAgents(config)
        replay_buffer = IA2CReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'MA2C-Queue':
        config: MA2CArguments = load_from_yaml(MA2CArguments, 'config/ma2c_large_grid.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = MA2CAgents(config)
        replay_buffer = MA2CReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'MA2C-Queue-Real':
        config: MA2CArguments = load_from_yaml(MA2CArguments, 'config/ma2c_real_net.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = MA2CAgents(config)
        replay_buffer = MA2CReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'MSAC-Queue':
        config: MSACArguments = load_from_yaml(MSACArguments, 'config/msac_large_grid.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = MSACAgents(config)
        replay_buffer = MSACReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'MSAC-Queue-Real':
        config: MSACArguments = load_from_yaml(MSACArguments, 'config/msac_real_net.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = MSACAgents(config)
        replay_buffer = MSACReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'IC3Net-Queue':
        config: IC3NetArguments = load_from_yaml(IC3NetArguments, 'config/ic3net_large_grid.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = IC3NetAgents(config)
        replay_buffer = IC3NetReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'IC3Net-Queue-Real':
        config: IC3NetArguments = load_from_yaml(IC3NetArguments, 'config/ic3net_real_net.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = IC3NetAgents(config)
        replay_buffer = IC3NetReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'IA2C-Wait':
        config = load_from_yaml(IA2CArguments, 'config/ia2c_large_grid_wait.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True)
        config.init_from_env(env)
        agent = IA2CAgents(config)
        replay_buffer = IA2CReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'IA2C-Wait-Real':
        config = load_from_yaml(IA2CArguments, 'config/ia2c_real_net_wait.yaml')
        build_data(config.env_type, config.env_config_path)
        # env is used only to initialize IA2CArguments
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True)
        config.init_from_env(env)
        agent = IA2CAgents(config)
        replay_buffer = IA2CReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'MA2C-Wait':
        config: MA2CArguments = load_from_yaml(MA2CArguments, 'config/ma2c_large_grid_wait.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = MA2CAgents(config)
        replay_buffer = MA2CReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'MA2C-Wait-Real':
        config: MA2CArguments = load_from_yaml(MA2CArguments, 'config/ma2c_real_net_wait.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = MA2CAgents(config)
        replay_buffer = MA2CReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'MSAC-Wait':
        config: MSACArguments = load_from_yaml(MSACArguments, 'config/msac_large_grid_wait.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = MSACAgents(config)
        replay_buffer = MSACReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'MSAC-Wait-Real':
        config: MSACArguments = load_from_yaml(MSACArguments, 'config/msac_real_net_wait.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = MSACAgents(config)
        replay_buffer = MSACReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'IC3Net-Wait':
        config: IC3NetArguments = load_from_yaml(IC3NetArguments, 'config/ic3net_large_grid_wait.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = IC3NetAgents(config)
        replay_buffer = IC3NetReplayBuffer()
        train(config, agent, replay_buffer)
    elif args.exp_name == 'IC3Net-Wait-Real':
        config: IC3NetArguments = load_from_yaml(IC3NetArguments, 'config/ic3net_real_net_wait.yaml')
        build_data(config.env_type, config.env_config_path)
        env = load_env(config.env_type, config.env_config_path, config.base_dir, 0, config.env_simulator_port, True, include_fingerprint=config.include_fingerprint)
        config.init_from_env(env)
        agent = IC3NetAgents(config)
        replay_buffer = IC3NetReplayBuffer()
        train(config, agent, replay_buffer)
    
    


if __name__ == '__main__':
    # multiprocessing is used during exploration.
    # The maximum number of processes is controlled by argument `max_explore_processes`.
    mp.set_start_method('spawn')
    main()
