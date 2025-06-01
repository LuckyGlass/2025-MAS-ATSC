"""
Training framework for ATSC.
@author: Yansheng Mao
"""
import datetime
import multiprocessing as mp
import numpy as np
import os
import time
import torch
import tqdm
from copy import deepcopy
from queue import Empty
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Type
from .config import ATSCArguments
from .envs import build_data, load_env
from .envs.atsc_env import TrafficSimulator
from .logger_utils import get_logger, suppress_all_output
from .models import ATSCAgentCollection, ReplayBuffer


logger = get_logger()


@torch.no_grad()
def explore_worker(
    model_cls: Type[ATSCAgentCollection],
    replay_buffer_cls: Type[ReplayBuffer],
    args: ATSCArguments,
    env_seed: int,
    env_port: int,
    replay_buffer_queue: mp.Queue
):
    try:
        tmp_state_dict_path = os.path.join(args.base_dir, 'temp_state_dict.torch')
        model = model_cls.load_state_dict(args, torch.load(tmp_state_dict_path))
        model.reset()
        replay_buffer = replay_buffer_cls()
        with suppress_all_output():
            env = load_env(args.env_type, args.env_config_path, args.base_dir, env_seed, port=env_port, train_mode=True, include_fingerprint=args.include_fingerprint)
            observation = env.reset()
        observation = [torch.from_numpy(o).to(dtype=torch.float32, device=args.device) for o in observation]  # Add time dim
        global_rewards, waits, queues = [], [], []
        sampled_steps = 0
        while True:
            sampled_steps += 1
            if args.include_fingerprint:
                action, policy = model.forward(observation, replay_buffer=replay_buffer, return_policy=True)
                env.update_fingerprint([p.cpu().numpy() for p in policy])
            else:
                action = model.forward(observation, replay_buffer=replay_buffer)
            next_observation, reward, done, global_reward, wait, queue = env.step(action)
            reward = torch.from_numpy(reward).to(dtype=torch.float32, device=args.device)
            next_observation = [torch.from_numpy(o).to(dtype=torch.float32, device=args.device) for o in next_observation]  # Add time dim
            global_rewards.append(global_reward)
            waits.append(wait)
            queues.append(queue)
            replay_buffer.env_side(next_observation, reward, done)
            if done or sampled_steps >= args.max_episode_steps:
                break
            observation = next_observation
        env.terminate()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_save_path = os.path.join(args.base_dir, f"temp_replay_buffer_{timestamp}_{env_port}.torch")
        torch.save(replay_buffer, tmp_save_path)
        replay_buffer_queue.put((global_rewards, waits, queues, sampled_steps, tmp_save_path, env.port))
        logger.info(f"Port = {env_port}: Finish sampling episode of {sum(global_rewards)} reward with {sampled_steps} steps.")
    except:
        if 'env' in locals():
            env.terminate()
            logger.info(f"Port = {env_port}: Detect SIGTERM. Close environment...")


def train(args: ATSCArguments, model: ATSCAgentCollection, replay_buffer: ReplayBuffer):
    writer = SummaryWriter(log_dir=os.path.join(args.base_dir, 'tensorboard'))
    os.makedirs(args.base_dir, exist_ok=True)
    cur_steps = 0
    env_seed = args.env_start_seed
    tmp_state_dict_path = os.path.join(args.base_dir, 'temp_state_dict.torch')
    last_save_tag = -1
    with tqdm.tqdm(total=args.total_steps, desc="Total steps") as pbar:
        while cur_steps < args.total_steps:
            # Explore
            processes = []
            global_rewards, waits, queues = [], [], []
            model_state_dict = model.export_state_dict()
            torch.save(model_state_dict, tmp_state_dict_path)
            available_ports = list(range(args.env_simulator_port, args.env_simulator_port + args.max_explore_processes))
            replay_buffer_queue = mp.Queue()
            cur_explore_steps = 0
            num_active_processes = 0
            while cur_explore_steps < args.train_steps:
                if len(available_ports) > 0:
                    num_active_processes += 1
                    tmp_port = available_ports.pop()
                    process = mp.Process(target=explore_worker, args=(type(model), type(replay_buffer), args, env_seed, tmp_port, replay_buffer_queue))
                    logger.info(f"Sampling an episode on port {tmp_port} with random seed {env_seed}.")
                    env_seed += 1
                    process.start()
                    processes.append(process)
                else:
                    time.sleep(0.1)
                while True:
                    try:
                        tmp_global_rewards, tmp_waits, tmp_queues, sampled_steps, tmp_save_path, tmp_port = replay_buffer_queue.get_nowait()
                        logger.info(f"Loading results {tmp_save_path} from port {tmp_port}...")
                        global_rewards += tmp_global_rewards
                        waits += tmp_waits
                        queues += tmp_queues
                        cur_explore_steps += sampled_steps
                        cur_steps += sampled_steps
                        pbar.update(sampled_steps)
                        tmp_replay_buffer = torch.load(tmp_save_path, weights_only=False)
                        replay_buffer.extend(tmp_replay_buffer)
                        os.remove(tmp_save_path)
                        available_ports.append(tmp_port)
                    except Empty:
                        break
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()  # necessary
            while True:
                try:
                    tmp_save_path = replay_buffer_queue.get_nowait()[4]
                    os.remove(tmp_save_path)
                except Empty:
                    break
            log_data = {
                'global_step': cur_steps,
                'mean_reward': np.mean(global_rewards),
                'std_reward': np.std(global_rewards),
                'mean_wait': np.mean(waits),
                'std_wait': np.std(waits),
                'mean_queue': np.mean(queues),
                'std_queue': np.std(queues),
            }
            writer.add_scalar('explore/mean_reward', log_data['mean_reward'], cur_steps)
            writer.add_scalar('explore/std_reward', log_data['std_reward'], cur_steps)
            writer.add_scalar('explore/mean_wait', log_data['mean_wait'], cur_steps)
            writer.add_scalar('explore/std_wait', log_data['std_wait'], cur_steps)
            writer.add_scalar('explore/mean_queue', log_data['mean_queue'], cur_steps)
            writer.add_scalar('explore/std_queue', log_data['std_queue'], cur_steps)
            logger.info(str(log_data))
            # Train
            logger.info("Start training...")
            model.train(replay_buffer, args, writer, cur_steps)
            
            if args.save_interval is not None and last_save_tag < cur_steps // args.save_interval:
                last_save_tag = cur_steps // args.save_interval
                save_path = os.path.join(args.base_dir, f'checkpoint_{cur_steps}.torch')
                torch.save(model.export_state_dict(), save_path)
    save_path = os.path.join(args.base_dir, f'final_checkpoint.torch')
    torch.save(model.export_state_dict(), save_path)
