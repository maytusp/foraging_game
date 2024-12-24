# Edit: 20Dec2024
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


import supersuit as ss
from environment_pickup import Environment
from utils import *
from models import PPOLSTMAgent



@dataclass
class Args:
    ckpt_path = "checkpoints/ppo_ps_pickup/final_model.pt"
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "PPO Foraging Game"
    wandb_entity: str = "maytusp"
    capture_video: bool = False
    saved_dir = "logs/ppo_ps_pickup_100M"
    video_save_dir = os.path.join(saved_dir, "vids")
    visualize = True
    agent_visible = True

    # Algorithm specific arguments
    env_id: str = "Foraging-Single-v1"
    total_episodes: int = 100
    num_channels = 4
    num_obs_grid = 5



if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.visualize:
        from visualize import *
        from moviepy.editor import *

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = Environment(agent_visible=args.agent_visible)
    grid_size = (env.image_size, env.image_size)
    num_channels = env.num_channels
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    # Vectorise env
    envs = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(envs, 1, num_cpus=0, base_class="gymnasium")

    agent = PPOLSTMAgent(num_actions).to(device)
    agent.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    agent.eval()


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, num_agents, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, num_agents, agent.lstm.hidden_size).to(device),
    )
    average_sr = 0
    collected_items = 0
    running_rewards = 0.0
    running_length = 0
    for episode_id in range(1, args.total_episodes + 1):
        
        next_obs_dict, _ = envs.reset(seed=args.seed)
        next_obs, next_locs, next_eners = extract_dict(next_obs_dict, device, use_message=False)
        next_done = torch.zeros((num_agents)).to(device)
        returns = 0
        ep_step = 0
        frames = []
        while not next_done[0]:
            if args.visualize:
                single_env = envs.vec_envs[0].unwrapped.par_env
                frame = visualize_environment(single_env, ep_step)
                frames.append(frame.transpose((1, 0, 2)))
            
            with torch.no_grad():

                action, action_logprob, _, value, next_lstm_state = agent.get_action_and_value((next_obs, next_locs, next_eners), 
                                                                                                    next_lstm_state, next_done)
                # print(f"step {ep_step} agent_actions = {action}")
            env_action = action.cpu().numpy()
            next_obs_dict, reward, terminations, truncations, infos = envs.step(env_action)
            next_obs, next_locs, next_eners = extract_dict(next_obs_dict, device, use_message=False)
            next_done = np.logical_or(terminations, truncations)
            next_done = torch.tensor(terminations).to(device)
            next_obs = torch.Tensor(next_obs).to(device)
            reward = torch.tensor(reward).to(device)
            returns += torch.sum(reward).cpu()
            ep_step+=1
        collected_items += infos[0]['episode']['collect']
        running_length += infos[0]['episode']['l']
        
        print(f"EPISODE {episode_id}: {infos[0]['episode']['collect']}")
        running_rewards += (returns / 2)

        if args.visualize: # and returns > 5:
            os.makedirs(args.video_save_dir, exist_ok=True)
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_videofile(os.path.join(args.video_save_dir, f"ep_{episode_id}_collected_items={infos[0]['episode']['collect']}_length_{infos[0]['episode']['l']}.mp4"), codec="libx264")
        
        if infos[0]['episode']['collect'] == len(single_env.foods):
            average_sr += 1

    print(f"Average SR: {average_sr / args.total_episodes}")
    print(f"Average Reward {running_rewards / args.total_episodes}")
    print(f"Average Collected Items: {collected_items / args.total_episodes}")
    print(f"Average Length: {running_length / args.total_episodes}")
    envs.close()
    # writer.close()