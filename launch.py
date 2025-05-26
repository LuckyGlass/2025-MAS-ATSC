from atsc.train import train
from atsc.models import IA2CAgents, IA2CArguments, IA2CReplayBuffer
from atsc.config import load_from_yaml
from atsc.envs import load_env


def main():
    config = load_from_yaml(IA2CArguments, 'config/ia2c_large_grid.yaml')
    env = load_env(config.env_type, config.env_config_path, config.base_dir, config.env_simulator_port, True)
    config.init_from_env(env)
    agent = IA2CAgents(config)
    replay_buffer = IA2CReplayBuffer()
    train(config, env, agent, replay_buffer)


if __name__ == '__main__':
    main()
