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

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from models.nets import *
from environments.environment_single import *
from models import PPOLSTMAgent
from train_single_ppo import make_env

@dataclass
class Args:
    ckpt_path = "checkpoints/ppo/30M.pt"
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "PPO Foraging Game"
    wandb_entity: str = "maytusp"
    capture_video: bool = False
    video_save_dir = "vids/ppo_30M"
    visualize = True

    # Algorithm specific arguments
    env_id: str = "Foraging-Single-v1"
    total_episodes: int = 100
    num_channels = 4
    num_obs_grid = 5



if __name__ == "__main__":
    env = Environment()
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.visualize:
        from visualize import visualize_environment
        from moviepy.editor import *

    # if args.track:
    #     import wandb

    #     wandb.init(
    #         project=args.wandb_project_name,
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         monitor_gym=True,
    #         save_code=True,
    #     )
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, i, args.capture_video, run_name) for i in range(1)],
    )
    num_actions = envs.single_action_space.n
    agent = PPOLSTMAgent(num_actions).to(device)
    agent.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    agent.eval()


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
    )
    average_sr = 0
    for episode_id in range(1, args.total_episodes + 1):
        
        next_obs_dict, _ = envs.reset(seed=args.seed)
        next_obs, next_locs = next_obs_dict["image"], next_obs_dict["location"]
        next_obs = torch.Tensor(next_obs).to(device)
        next_locs = torch.Tensor(next_locs).to(device)
        next_done = torch.zeros((1)).to(device)
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        returns = 0
        ep_step = 0
        frames = []
        while not next_done:
            if args.visualize:
                frame = visualize_environment(envs.envs[0], ep_step)
                frames.append(frame.transpose((1, 0, 2)))
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
            env_action = action.cpu().numpy()
            next_obs_dict, reward, terminations, truncations, infos = envs.step(env_action)
            next_obs, next_locs = next_obs_dict["image"], next_obs_dict["location"]
            next_done = np.logical_or(terminations, truncations)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            returns += np.mean(reward)
            ep_step+=1
        if args.visualize: # and returns > 5:
            os.makedirs(args.video_save_dir, exist_ok=True)
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_videofile(os.path.join(args.video_save_dir, f"ep_{episode_id}_return={returns}.mp4"), codec="libx264")
        print(f"Total Reward: {returns}")
        if returns > 0:
            average_sr += 1

    print(f"Average SR: {average_sr / args.total_episodes}")
    envs.close()
    # writer.close()