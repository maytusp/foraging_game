# Created 25 Aug 2025
# Note: This code only work for single environment. It doesn't support parallel environments.
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

# CUDA_VISIBLE_DEVICES=1 python -m scripts.torch_pickup.test_all_pairs
import supersuit as ss
from environments.torch_pickup_high_v1 import *
from utils.process_data import *
from models.pickup_models import PPOLSTMCommAgent




@dataclass
class Args:

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    wandb_project_name: str = "PPO Foraging Game"
    wandb_entity: str = "maytusp"
    capture_video: bool = False

    visualize = False
    save_trajectory = True
    ablate_message = False
    ablate_type = "noise" # zero, noise
    agent_visible = False
    fully_visible_score = False
    identical_item_obs = False
    zero_memory = False
    memory_transfer = False
    
    # Algorithm specific arguments
    env_id: str = "Foraging-Single-v1"
    total_episodes: int = 1000
    """vocab size"""
    image_size = 3
    """number of observation grid"""
    N_att = 2
    """number of attributes"""
    N_val = 10
    """number of values"""
    N_i = 2
    """number of items"""
    grid_size = 5
    max_steps = 10
    """grid size"""
    mode = "test"
    # network_pairs = "0-0" # population training evaluation
    # selected_networks = network_pairs.split("-")
    
    num_nets_to_model_step = {3: 1177600000}
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    swap_agent = {0:1, 1:0}
    # Loop over all network pair combinations (0-0, 0-1, …, 2-2)
    for num_networks in [3]:
        args.model_step = args.num_nets_to_model_step[num_networks]
        args.num_networks = num_networks
        args.model_name = f"pop_ppo_3net_invisible_wospeedrw"
        for seed in [1,2]: # ,2,3]:
            for i in range(args.num_networks):
                for j in range(args.num_networks):
                    args.seed = seed
                    args.n_words = 4
                    args.combination_name = f"grid{args.grid_size}_img{args.image_size}_ni{args.N_i}_nw{args.n_words}_ms{args.max_steps}"
                    # Update the network pair and dependent paths/parameters
                    network_pairs = f"{i}-{j}"
                    selected_networks = network_pairs.split("-")
                    args.ckpt_path = f"checkpoints/torch_pickup_high_v1/{args.model_name}/{args.combination_name}/seed{args.seed}/agent_{selected_networks[0]}_step_{args.model_step}.pt"
                    args.ckpt_path2 = f"checkpoints/torch_pickup_high_v1/{args.model_name}/{args.combination_name}/seed{args.seed}/agent_{selected_networks[1]}_step_{args.model_step}.pt"
                    args.saved_dir = f"logs/torch_pickup_high_v1/{args.model_name}/{network_pairs}/{args.combination_name}_{args.model_step}/seed{args.seed}/mode_{args.mode}"
                    args.video_save_dir = os.path.join(args.saved_dir, "vids")
                    if args.ablate_message:
                        args.saved_dir = os.path.join(args.saved_dir, args.ablate_type)
                    else:
                        args.saved_dir = os.path.join(args.saved_dir, "normal")            
                    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
                    print(f"saved at: {args.saved_dir}")


                    os.makedirs(args.saved_dir, exist_ok=True)
                    # TRY NOT TO MODIFY: seeding
                    random.seed(args.seed)
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    torch.backends.cudnn.deterministic = args.torch_deterministic

                    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


                    cfg = EnvConfig(
                        grid_size=args.grid_size,
                        image_size=args.image_size,
                        num_agents=2,               # keep your setting
                        num_foods=args.N_i,
                        num_walls=0,
                        max_steps=args.max_steps,
                        agent_visible=args.agent_visible,
                        food_energy_fully_visible=args.fully_visible_score,
                        mode=args.mode,
                        seed=args.seed,
                    )
                    envs = TorchForagingEnv(cfg, device=device, num_envs=1)
                    num_agents = cfg.num_agents
                    num_channels = 2
                    num_actions = envs.num_actions
                    if args.visualize:
                        from visualize_torch import *
                        from moviepy.editor import *
                        import pygame
                        screen, font = init_pygame(envs.cfg.grid_size)
                        clock = pygame.time.Clock()
                    agent0 = PPOLSTMCommAgent(num_actions=num_actions, 
                                                grid_size=args.grid_size, 
                                                n_words=args.n_words, 
                                                embedding_size=16, 
                                                num_channels=num_channels, 
                                                image_size=args.image_size).to(device)
                    agent0.load_state_dict(torch.load(args.ckpt_path, map_location=device))
                    agent0.eval()

                    agent1 = PPOLSTMCommAgent(num_actions=num_actions, 
                                                grid_size=args.grid_size, 
                                                n_words=args.n_words, 
                                                embedding_size=16, 
                                                num_channels=num_channels, 
                                                image_size=args.image_size).to(device)
                    agent1.load_state_dict(torch.load(args.ckpt_path2, map_location=device))
                    agent1.eval()
                    
                    random.seed(args.seed)
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)


                    # TRY NOT TO MODIFY: start the game
                    global_step = 0
                    start_time = time.time()
                    next_lstm_state = (
                        torch.zeros(agent0.lstm.num_layers, num_agents, agent0.lstm.hidden_size).to(device),
                        torch.zeros(agent0.lstm.num_layers, num_agents, agent0.lstm.hidden_size).to(device),
                    )

                    collected_items = 0
                    running_rewards = 0.0
                    running_length = 0
                    running_success_length=0
                    num_success_episodes=0
                    
                    next_obs, next_locs = envs._obs_core()
                    next_r_messages = torch.zeros((1, num_agents), dtype=torch.int64).to(device) # action: sent message

                    log_data = {}

                    for episode_id in range(1, args.total_episodes + 1):
                        ep_len = 0

                        next_done = torch.zeros((num_agents)).to(device)
                        returns = 0
                        ep_step = 0
                        frames = []
                        next_lstm_state = (
                            torch.zeros(agent0.lstm.num_layers, num_agents, agent0.lstm.hidden_size).to(device),
                            torch.zeros(agent0.lstm.num_layers, num_agents, agent0.lstm.hidden_size).to(device),
                        )
                        # print(f"Episode {episode_id}")
                        ############### Logging #########################
                        log_obs = torch.zeros((cfg.max_steps, num_agents, num_channels, args.image_size, args.image_size)).to(device) # obs: vision
                        log_locs = torch.zeros((cfg.max_steps, num_agents, 2)).to(device) # obs: location
                        log_r_messages = torch.zeros((cfg.max_steps, num_agents), dtype=torch.int64).to(device) # obs: received message
                        log_actions = torch.zeros((cfg.max_steps, num_agents)).to(device) # action: physical action
                        log_s_messages = torch.zeros((cfg.max_steps, num_agents), dtype=torch.int64).to(device) -1 # action: sent message
                        log_rewards = torch.zeros((cfg.max_steps, num_agents)).to(device)
                        log_s_message_embs = torch.zeros((cfg.max_steps, agent0.embedding_size, num_agents)).to(device) # obs: received message

                        log_food_dict = {}
                        log_food_dict['target_food_id'] = envs.target_food_id.squeeze(0).cpu().numpy()
                        log_food_dict['location'] = envs.food_pos.squeeze(0).cpu().numpy()
                        log_food_dict['score'] = envs.food_energy.squeeze(0).cpu().numpy()

                        log_who_see_target = envs.score_visible_to_agent.squeeze(0).cpu().numpy()

                        ############################################################
                        while not next_done[0]:

                            if args.visualize and not(args.save_trajectory):
                                frame = visualize_torch_environment(envs, ep_step, screen, font, b=0)
                                frames.append(frame)

                            with torch.no_grad():
                                if args.ablate_message:
                                    if args.ablate_type == "zero":
                                        next_r_messages = torch.zeros_like(next_r_messages).to(device)
                                    elif args.ablate_type == "noise":
                                        next_r_messages = torch.randint(0, 10, next_r_messages.shape).to(device)
                                    else:
                                        raise Exception("only zero and noise are allowed")

                                ######### Logging #################
                                if args.save_trajectory:
                                    log_obs[ep_step] = next_obs
                                    log_locs[ep_step] = next_locs
                                    log_r_messages[ep_step] = next_r_messages.squeeze().to(device)
                                ###################################

                                (h0,c0) = (next_lstm_state[0][:,0,:].unsqueeze(1), next_lstm_state[1][:,0,:].unsqueeze(1))
                                (h1,c1) = (next_lstm_state[0][:,1,:].unsqueeze(1), next_lstm_state[1][:,1,:].unsqueeze(1))

                                action0, _, _, s_message0, _, _, _, (new_h0, new_c0) = agent0.get_action_and_value((next_obs[0,0,...].unsqueeze(0),
                                                                                                            next_locs[0,0,...].unsqueeze(0), 
                                                                                                            next_r_messages[0,0,...].unsqueeze(0)),
                                                                                                            (h0,c0), next_done[0])
                                action1, _, _, s_message1, _, _, _, (new_h1, new_c1) = agent1.get_action_and_value((next_obs[0,1,...].unsqueeze(0),
                                                                                                            next_locs[0,1,...].unsqueeze(0),
                                                                                                            next_r_messages[0,1,...].unsqueeze(0)),
                                                                                                            (h1,c1), next_done[1])

                                action = torch.cat((action0, action1), dim=0)
                                s_message = torch.cat((s_message0, s_message1), dim=0)

                                new_h = torch.cat((new_h0, new_h1), dim=1)
                                new_c = torch.cat((new_c0, new_c1), dim=1)
                                next_lstm_state = (new_h, new_c)
                                if args.zero_memory:
                                    next_lstm_state = (
                                        torch.zeros(agent0.lstm.num_layers, num_agents, agent0.lstm.hidden_size).to(device),
                                        torch.zeros(agent0.lstm.num_layers, num_agents, agent0.lstm.hidden_size).to(device),
                                    )
                                
                                

                            acts_BA = action.unsqueeze(0)  # [B,A] = [1,2]
                            (next_obs, next_locs), all_rewards, all_terminations, all_truncations, infos =  envs._step_core(acts_BA)
                            env_info = (all_rewards, all_terminations, all_truncations)
                
                            for i in range(num_agents):
                                next_r_messages[:,i] = s_message[swap_agent[i]] # agent exchange msgs
                                next_done[i] = (all_terminations | all_truncations).float()
                                

                            ####### Logging #########
                            if args.save_trajectory:
                                log_actions[ep_step] = action.squeeze(0)
                                log_s_messages[ep_step] = s_message.squeeze(0)
                                # print(log_s_messages)
                                log_rewards[ep_step] = all_rewards.squeeze(0)
                            ##########################
                            returns += all_rewards[:,0].squeeze(0).item() # agents get the same reward, so we pick one of those
                            ep_len+=1
                            ep_step+=1
                            

                        collected_items += int(returns >= 1)
                        running_length += ep_len

                        if returns >= 1: # count success
                            running_success_length += ep_len
                            num_success_episodes += 1

                        if args.save_trajectory:
                            with torch.no_grad():
                                # message sent from agent0 to agent1 --> use agent1's embedding
                                log_s_message_embs[:ep_len, :, 0] = agent1.message_encoder(log_s_messages[:ep_len, 0])
                                # message sent from agent1 to agent0 --> use agent0's embedding
                                log_s_message_embs[:ep_len, :, 1] = agent0.message_encoder(log_s_messages[:ep_len, 1])

                            log_s_message_embs = log_s_message_embs.cpu().numpy()
                            log_obs = log_obs.cpu().numpy()
                            log_locs = log_locs.cpu().numpy()
                            log_r_messages = log_r_messages.cpu().numpy()
                            log_actions = log_actions.cpu().numpy()
                            log_s_messages = log_s_messages.cpu().numpy()
                            log_rewards = log_rewards.cpu().numpy()
                            import pickle
                            # print("log_r_messages", log_r_messages)
                            # Combine all your data into a dictionary
                            log_data[f"episode_{episode_id}"] = {
                                "log_food_dict": log_food_dict,
                                "log_s_message_embs": log_s_message_embs,
                                "log_obs": log_obs,
                                "log_locs": log_locs,
                                "log_r_messages": log_r_messages,
                                "log_actions": log_actions,
                                "log_s_messages": log_s_messages,
                                "log_rewards": log_rewards,
                                "who_see_target": log_who_see_target
                            }

                        running_rewards += returns

                        if args.visualize: # and returns > 5:
                            print(len(frames))
                            os.makedirs(args.video_save_dir, exist_ok=True)
                            clip = ImageSequenceClip(frames, fps=5)
                            clip.write_videofile(os.path.join(args.video_save_dir, f"ep_{episode_id}_r={returns}.mp4"), codec="libx264")

                    # Save the dictionary to a pickle file
                    if args.save_trajectory:
                        with open(os.path.join(args.saved_dir, "trajectory.pkl"), "wb") as f:
                            pickle.dump(log_data, f)

                    with open(os.path.join(args.saved_dir, "score.txt"), "w") as log_file:
                        print(f"Success Rate: {collected_items / args.total_episodes}", file=log_file)
                        print(f"Average Reward {running_rewards / args.total_episodes}", file=log_file)
                        print(f"Average Length: {running_length / args.total_episodes}", file=log_file)
                        print(f"Average Success Length: {running_success_length / num_success_episodes}", file=log_file)
                    
                    envs.close()
                    # writer.close()
