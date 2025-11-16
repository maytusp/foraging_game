
# test_gpt.py
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from environments.torch_pickup_high_v1 import TorchForagingEnv, EnvConfig
from utils.process_data import init_logs
from models.pickup_models import PPOTransformerCommAgent  # <-- changed import

# CUDA_VISIBLE_DEVICES=1 python -m scripts.torch_pickup.test_gpt
@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    # Algorithm specific arguments
    env_id: str = "Foraging-Single-v1"
    """the id of the environment"""
    total_episodes: int = 1000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    save_trajectory: bool = True
    visualize: bool = False
    # Populations
    num_networks = 3
    reset_iteration: int = 1
    self_play_option: bool = False
    log_every = 32
    n_words = 4
    image_size = 3
    N_i = 2
    grid_size = 5
    max_steps = 10
    fully_visible_score = False
    agent_visible = False
    time_pressure = True
    mode = "test"
    torch_deterministic: bool = True
    cuda: bool = True

    name2step = {"gpt_ppo_2net_invisible": 102400000,
                "gpt_sp_ppo_2net_invisible": 102400000,
                "gpt_ppo_3net_invisible": 102400000, 
                }
    name2numnet = {"gpt_ppo_2net_invisible": 2,
                "gpt_sp_ppo_2net_invisible": 2,
                "gpt_ppo_3net_invisible": 3, 
                }
    combination_name = f"grid{grid_size}_img{image_size}_ni{N_i}_nw{n_words}_ms{max_steps}"



if __name__ == "__main__":
    args = tyro.cli(Args)
    for model_name in args.name2step.keys():
        args.model_name = model_name
        args.model_step = args.name2step[model_name]
        args.num_networks = args.name2numnet[model_name]
        
        for seed in [1,2,3]:
            args.seed = seed
            # loop over network pairs here
            ckpt_dir = f"checkpoints/torch_pickup_high_v1/{args.model_name}/{args.combination_name}/seed{args.seed}/"
            for i in range(args.num_networks):
                for j in range(i+1):
                    # TRY NOT TO MODIFY: seeding
                    random.seed(args.seed)
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    torch.backends.cudnn.deterministic = args.torch_deterministic
                    
                    network_pairs = f"{i}-{j}"
                    selected_networks = network_pairs.split("-")
                    selected_networks[0], selected_networks[1] = int(selected_networks[0]), int(selected_networks[1])
                    # Where to save evaluation logs / trajectories
                    args.saved_dir = (
                                f"logs/torch_pickup_high_v1/{args.model_name}/{network_pairs}/"
                                f"{args.combination_name}_{args.model_step}/seed{args.seed}/mode_{args.mode}"
                                )   
                    os.makedirs(args.saved_dir, exist_ok=True)


                    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

                    cfg = EnvConfig(
                        grid_size=args.grid_size,
                        image_size=args.image_size,
                        num_agents=2,
                        num_channels=2,
                        num_foods=args.N_i,
                        num_walls=0,
                        max_steps=args.max_steps,
                        agent_visible=args.agent_visible,
                        food_energy_fully_visible=args.fully_visible_score,
                        mode=args.mode,
                        seed=args.seed,
                        time_pressure=args.time_pressure,
                    )
                    envs = TorchForagingEnv(cfg, device=device, num_envs=args.num_envs)
                    num_agents = cfg.num_agents
                    num_channels =  cfg.num_channels

                    num_actions = envs.num_actions

                    if args.visualize:
                        from visualize_torch import *
                        from moviepy.editor import *
                        import pygame

                        screen, font = init_pygame(envs.cfg.grid_size)
                        clock = pygame.time.Clock()
                    # Initialize dicts for keeping agent models and experiences
                    agents = {}
                    optimizers = {}
                    obs = {}
                    locs = {}
                    r_messages = {}
                    actions = {}
                    s_messages = {}
                    action_logprobs = {}
                    message_logprobs = {}
                    rewards = {}
                    dones = {}
                    values = {}

                    next_obs, next_locs, next_done, next_memory_state = {}, {}, {}, {}
                    # TRY NOT TO MODIFY: start the game
                    next_obs, next_locs = envs._obs_core()
                    next_r_messages = torch.zeros((args.num_envs, num_agents), dtype=torch.int64).to(device)

                    swap_agent = {0: 1, 1: 0}

                    # --- create population of transformer agents ---
                    for network_id in range(args.num_networks):
                        agents[network_id] = PPOTransformerCommAgent(
                            num_actions=num_actions,
                            grid_size=args.grid_size,
                            n_words=args.n_words,
                            embedding_size=16,
                            num_channels=num_channels,
                            image_size=args.image_size,
                            d_model=128,
                            n_layers=2,
                            n_heads=4,
                        ).to(device)
                        ckpt_path = os.path.join(ckpt_dir, f"agent_{network_id}_step_{args.model_step}.pt")
                        
                        agents[network_id].load_state_dict(
                            torch.load(ckpt_path, map_location=device),
                            strict=True, 
                        )
                        agents[network_id].eval()
                    

                    # Storage tensors
                    for i in range(num_agents):
                        next_done[i] = torch.zeros(args.num_envs).to(device)
                        # memory state: [num_envs, d_model]
                        next_memory_state[i] = agents[0].init_memory(args.num_envs, device)

                    start_time = time.time()
                    global_step = 0
                    initial_memory_state = {}

                    

                    # --- log performance ---
                    episodes_since_log = 0
                    sum_return_since_log = 0.0
                    sum_length_since_log = 0.0
                    successes_since_log = 0.0

                    ep_ret = torch.zeros(1, num_agents, device=device)
                    ep_len = torch.zeros(1, num_agents, device=device)


                    for i in range(num_agents):
                        initial_memory_state[i] = next_memory_state[i].clone()
                    log_data = {}
                    for episode_id in range(args.total_episodes):
                        logs = init_logs(device, cfg, envs, agents)
                        frames = []
                        # Reset memory state whenever we re-sample networks
                        for i in range(num_agents):
                            next_memory_state[i] = agents[0].init_memory(args.num_envs, device)
                        for ep_step in range(0, args.max_steps):
                            action = {}
                            s_message = {}
                            action_logprob = {}
                            message_logprob = {}
                            value = {}
                        # Logging current observations
                            if args.save_trajectory:
                                logs["log_obs"][ep_step] = next_obs
                                logs["log_locs"][ep_step] = next_locs
                                logs["log_r_messages"][ep_step] = (
                                    next_r_messages.squeeze().to(device)
                                )
                            if args.visualize and not args.save_trajectory:
                                frame = visualize_torch_environment(
                                    envs, ep_step, screen, font, b=0
                                )
                                frames.append(frame)

                            for i in range(num_agents):
                                network_id = selected_networks[i]
                                with torch.no_grad():
                                    (
                                        action[i],
                                        action_logprob[i],
                                        _,
                                        s_message[i],
                                        message_logprob[i],
                                        _,
                                        value[i],
                                        next_memory_state[i],
                                    ) = agents[network_id].get_action_and_value(
                                        (next_obs[:, i, :, :, :], next_locs[:, i, :], next_r_messages[:, i]),
                                        next_memory_state[i],
                                        next_done[i],
                                    )
                                    if ep_step == 0:
                                        next_memory_state[i] = agents[0].init_memory(args.num_envs, device)
                            # Step env
                            acts_BA = torch.stack(
                                [action[i].long().to(device) for i in range(num_agents)], dim=1
                            )  # [B, A]

                
                            (next_obs, next_locs), all_rewards, all_terminations, all_truncations, infos = \
                                envs._step_core(acts_BA)

                            # Logging
                            if args.save_trajectory:
                                logs["log_actions"][ep_step] = acts_BA.squeeze(0)
                                logs["log_s_messages"][ep_step] = torch.cat([s_message[0], s_message[1]]).squeeze(0)
                                logs["log_rewards"][ep_step] = all_rewards.squeeze(0)

                            for i in range(num_agents):
                                next_r_messages[:, i] = s_message[swap_agent[i]]
                                next_done[i] = (all_terminations | all_truncations).float()


                            # Logging returns
                            ep_ret += all_rewards
                            ep_len += 1

                            finished = (all_terminations | all_truncations).bool()
                            ### START FINISH EPISODE ###
                            if finished.any():
                                b_idx = finished.nonzero(as_tuple=False).squeeze(1)
                                finished_returns = ep_ret[b_idx].mean(dim=1).detach().cpu()
                                finished_lengths = ep_len[b_idx].mean(dim=1).detach().cpu()
                                next_r_messages[b_idx] = torch.zeros(
                                    (b_idx.numel(), num_agents), dtype=torch.int64
                                ).to(device)

                                sum_return_since_log += float(finished_returns.sum())
                                sum_length_since_log += float(finished_lengths.sum())
                                episodes_since_log += finished_returns.numel()
                                # count successes: episodes with return >= 1.0
                                successes_since_log += float((finished_returns >= 1.0).sum())

                                
                                if args.save_trajectory:
                                    with torch.no_grad():
                                        ep_len_scalar = int(ep_len[b_idx, 0].item())
                                        # message sent from agent0 to agent1 --> use agent1's embedding
                                        logs["log_s_message_embs"][:ep_len_scalar, :, 0] = agents[selected_networks[1]].message_encoder(
                                            logs["log_s_messages"][:ep_len_scalar, 0]
                                        )
                                        # message sent from agent1 to agent0 --> use agent0's embedding
                                        logs["log_s_message_embs"][:ep_len_scalar, :, 1] = agents[selected_networks[0]].message_encoder(
                                            logs["log_s_messages"][:ep_len_scalar, 1]
                                        )

                                    # Move to numpy
                                    logs["log_s_message_embs"] = logs["log_s_message_embs"].cpu().numpy()
                                    logs["log_obs"] = logs["log_obs"].cpu().numpy()
                                    logs["log_locs"] = logs["log_locs"].cpu().numpy()
                                    logs["log_r_messages"] = logs["log_r_messages"].cpu().numpy()
                                    logs["log_actions"] = logs["log_actions"].cpu().numpy()
                                    logs["log_s_messages"] = logs["log_s_messages"].cpu().numpy()
                                    logs["log_rewards"] = logs["log_rewards"].cpu().numpy()

                                    import pickle

                                    log_data[f"episode_{episode_id}"] = logs


                                if args.visualize:
                                    os.makedirs(args.video_save_dir, exist_ok=True)
                                    clip = ImageSequenceClip(frames, fps=5)
                                    clip.write_videofile(
                                        os.path.join(
                                            args.video_save_dir,
                                            f"ep_{episode_id}_r={returns}.mp4",
                                        ),
                                        codec="libx264",
                                    )

                                ep_ret[b_idx] = 0
                                ep_len[b_idx] = 0
                                break
                                ### END FINISH EPISODE ###
                    # Save trajectories (all episodes for this pair)
                    if args.save_trajectory:
                        import pickle

                        with open(
                            os.path.join(args.saved_dir, "trajectory.pkl"), "wb"
                        ) as f:
                            pickle.dump(log_data, f)
                    mean_ret_since_log = sum_return_since_log / episodes_since_log
                    mean_len_since_log = sum_length_since_log / episodes_since_log
                    success_rate_since_log = successes_since_log / episodes_since_log
                    print(f"NUM EPISODES: {episodes_since_log}")
                    print(f"AVG SR: {success_rate_since_log}")
                    print(f"AVG LENGTH: {mean_len_since_log}")
                    print(f"AVG RETURN: {mean_ret_since_log}")
                    print(f"-----------------")
                    # Save summary scores
                    with open(os.path.join(args.saved_dir, "score.txt"), "w") as log_file:
                        print(
                            f"Success Rate: {success_rate_since_log}",
                            file=log_file,
                        )
                        print(
                            f"Average Reward {mean_ret_since_log}",
                            file=log_file,
                        )
                        print(
                            f"Average Length: {mean_len_since_log}",
                            file=log_file,
                        )

                            
                envs.close()