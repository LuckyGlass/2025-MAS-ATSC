"""
Training framework for ATSC.
@author: Yansheng Mao
"""
import torch
from .config import ATSCArguments
from .envs.atsc_env import TrafficSimulator
from .models import ATSCAgentCollection, ReplayBuffer


def explore(model: ATSCAgentCollection, args: ATSCArguments, env: TrafficSimulator, replay_buffer: ReplayBuffer):
    observation = env.reset()
    observation = [torch.from_numpy(o) for o in observation]
    global_rewards = []
    sampled_steps = 0
    while True:
        sampled_steps += 1
        action = model.forward(observation)
        next_observation, reward, done, global_reward = env.step(action)
        next_observation = [torch.from_numpy(o) for o in next_observation]
        global_rewards.append(global_reward)
        replay_buffer.env_side(next_observation, reward, done)
        if done:
            break
        observation = next_observation
    return global_rewards, sampled_steps


def train(args: ATSCArguments, env: TrafficSimulator, model: ATSCAgentCollection, replay_buffer: ReplayBuffer):
    cur_steps = 0
    while cur_steps < args.total_steps:
        while len(replay_buffer) <= args.train_steps:
            global_rewards, sampled_steps = explore(model, args, env, replay_buffer)
            cur_steps += sampled_steps
        model.train(replay_buffer, args)
