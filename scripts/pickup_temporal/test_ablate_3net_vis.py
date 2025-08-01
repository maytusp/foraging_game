# Created 28 Feb2025: TODO
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
from environments.pickup_temporal import *
from utils.process_data import *
from models.pickup_models import PPOLSTMCommAgent


@dataclass
class Args:

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    torch_deterministic: bool = True
    cuda: bool = True
    wandb_project_name: str = "PPO Foraging Game"
    wandb_entity: str = "maytusp"
    capture_video: bool = False

    visualize = False
    save_trajectory = True
    ablate_message = True
    ablate_type = "zero" # zero, noise
    fully_visible_score = False
    identical_item_obs = False
    zero_memory = False
    memory_transfer = False
    
    # Algorithm specific arguments
    env_id: str = "Foraging-Single-v1"
    total_episodes: int = 1000
    n_words = 4
    """vocab size"""
    image_size = 3
    """number of observation grid"""
    N_att = 2
    """number of attributes"""
    N_val = 10
    """number of values"""
    N_i = 2
    """number of items"""
    grid_size = 8
    freeze_dur = 6
    max_steps = 40
    """grid size"""
    mode = "train"
    agent_visible = True
    model_name = "pop_ppo_3net_ablate_message"
    num_networks = 3
    model_step = "652800000"
    combination_name = f"grid{grid_size}_img{image_size}_ni{N_i}_nw{n_words}_ms{max_steps}_freeze_dur{freeze_dur}"



if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Loop over all network pair combinations (0-0, 0-1, …, 2-2)
    for seed in range(1,4):
        args.seed = seed
        args.test_max_steps = 40
        for i in range(args.num_networks):
            for j in range(args.num_networks):
                # Update the network pair and dependent paths/parameters
                network_pairs = f"{i}-{j}"
                selected_networks = network_pairs.split("-")
                args.ckpt_path = f"checkpoints/pickup_temporal/{args.model_name}/{args.combination_name}/seed{args.seed}/agent_{selected_networks[0]}_step_{args.model_step}.pt"
                args.ckpt_path2 = f"checkpoints/pickup_temporal/{args.model_name}/{args.combination_name}/seed{args.seed}/agent_{selected_networks[1]}_step_{args.model_step}.pt"
                args.saved_dir = f"logs/pickup_temporal/ablate/{args.model_name}/{network_pairs}/{args.combination_name}_{args.model_step}/seed{args.seed}/mode_{args.mode}"
                args.video_save_dir = args.saved_dir +"/videos/"
                if args.ablate_message:
                    args.saved_dir = os.path.join(args.saved_dir, args.ablate_type)
                else:
                    args.saved_dir = os.path.join(args.saved_dir, "normal")
                        
                run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
                print(f"saved at: {args.saved_dir}")
                if args.visualize:
                    from visualize_temporal import *
                    from moviepy.editor import *
                os.makedirs(args.saved_dir, exist_ok=True)
                # TRY NOT TO MODIFY: seeding
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.backends.cudnn.deterministic = args.torch_deterministic

                device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

                env = Environment(use_message=True,
                                    agent_visible=args.agent_visible,
                                    n_words=args.n_words,
                                    seed=args.seed, 
                                    N_i = args.N_i,
                                    grid_size=args.grid_size,
                                    image_size=args.image_size,
                                    max_steps=args.test_max_steps,
                                    mode=args.mode,
                                    freeze_dur=args.freeze_dur)

                num_channels = env.num_channels
                num_agents = len(env.possible_agents)
                num_actions = env.action_space(env.possible_agents[0])['action'].n
                observation_size = env.observation_space(env.possible_agents[0]).shape

                # Vectorise env
                envs = ss.pettingzoo_env_to_vec_env_v1(env)
                envs = ss.concat_vec_envs_v1(envs, 1, num_cpus=0, base_class="gymnasium")

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
                
                next_obs_dict, _ = envs.reset()
                next_obs, next_locs, _, next_r_messages = extract_dict(next_obs_dict, device, use_message=True)

                next_r_messages = torch.tensor(next_r_messages).squeeze().to(device)

                log_data = {}

                for episode_id in range(1, args.total_episodes + 1):
                    energy_obs = {"agent0": set(), "agent1": set()}

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
                    log_obs = torch.zeros((env.max_steps, num_agents, num_channels, args.image_size, args.image_size)).to(device) # obs: vision
                    log_locs = torch.zeros((env.max_steps, num_agents, 2)).to(device) # obs: location
                    log_r_messages = torch.zeros((env.max_steps, num_agents), dtype=torch.int64).to(device) # obs: received message
                    log_actions = torch.zeros((env.max_steps, num_agents)).to(device) # action: physical action
                    log_s_messages = torch.zeros((env.max_steps, num_agents), dtype=torch.int64).to(device) # action: sent message
                    log_rewards = torch.zeros((env.max_steps, num_agents)).to(device)
                    log_s_message_embs = torch.zeros((env.max_steps, agent0.embedding_size, num_agents)).to(device) # obs: received message
                    log_r_message_embs = torch.zeros((env.max_steps, agent0.embedding_size, num_agents)).to(device) # obs: received message
                    single_env = envs.vec_envs[0].unwrapped.par_env
                    log_food_dict = {}
                    log_distractor_food_dict = {"location":[], "type":[], "score":[]}
                    
                    food_list = single_env.foods
                    log_food_dict['location'] = [food.position for food in food_list]
                    log_food_dict['spawn_time'] = single_env.selected_time

                    ############################################################
                    while not next_done[0]:

                        if args.visualize and not(args.save_trajectory):
                            single_env = envs.vec_envs[0].unwrapped.par_env
                            frame = visualize_environment(single_env, ep_step)
                            frames.append(frame.transpose((1, 0, 2)))

                        with torch.no_grad():
                            ######### Logging #################
                            if args.save_trajectory:
                                log_obs[ep_step] = next_obs
                                log_locs[ep_step] = next_locs
                                log_r_messages[ep_step] = next_r_messages.squeeze().to(device)
                            ###################################

                            (h0,c0) = (next_lstm_state[0][:,0,:].unsqueeze(1), next_lstm_state[1][:,0,:].unsqueeze(1))
                            (h1,c1) = (next_lstm_state[0][:,1,:].unsqueeze(1), next_lstm_state[1][:,1,:].unsqueeze(1))


                            action0, _, _, s_message0, _, _, _, (new_h0, new_c0) = agent0.get_action_and_value((next_obs[0].unsqueeze(0),
                                                                                                        next_locs[0].unsqueeze(0), 
                                                                                                        next_r_messages[0].unsqueeze(0)),
                                                                                                        (h0,c0), next_done[0])
                            action1, _, _, s_message1, _, _, _, (new_h1, new_c1) = agent1.get_action_and_value((next_obs[1].unsqueeze(0),
                                                                                                        next_locs[1].unsqueeze(0),
                                                                                                        next_r_messages[1].unsqueeze(0)),
                                                                                                        (h1,c1), next_done[1])

                            action = torch.cat((action0, action1), dim=0)
                            s_message = torch.cat((s_message0, s_message1), dim=0)
                            if args.ablate_message:
                                if args.ablate_type == "zero":
                                    s_message = torch.zeros_like(s_message).to(device)
                                elif args.ablate_type == "noise":
                                    s_message = torch.randint(0, args.n_words, s_message.shape).to(device)
                                else:
                                    raise Exception("only zero and noise are allowed")
                            if args.memory_transfer:
                                h1 = torch.tensor(h0)
                                c1 = torch.tensor(c0)

                            new_h = torch.cat((new_h0, new_h1), dim=1)
                            new_c = torch.cat((new_c0, new_c1), dim=1)
                            next_lstm_state = (new_h, new_c)
                            if args.zero_memory:
                                next_lstm_state = (
                                    torch.zeros(agent0.lstm.num_layers, num_agents, agent0.lstm.hidden_size).to(device),
                                    torch.zeros(agent0.lstm.num_layers, num_agents, agent0.lstm.hidden_size).to(device),
                                )
                            
                            
                            # print(f"step {ep_step} agent_actions = {action}")
                        env_action, env_message = action.cpu().numpy(), s_message.cpu().numpy()
                        next_obs_dict, reward, terminations, truncations, infos = envs.step({"action": env_action, "message": env_message})
                        next_obs, next_locs, _, next_r_messages = extract_dict(next_obs_dict, device, use_message=True)
                        # Sanity check: if messages are swapped properly
                        # print("sent", s_message)
                        # print("swap", next_r_messages)
                        next_done = np.logical_or(terminations, truncations)
                        next_done = torch.tensor(terminations).to(device)
                        next_obs = torch.Tensor(next_obs).to(device)
                        next_r_messages = torch.tensor(next_r_messages).to(device)
                        reward = torch.tensor(reward).to(device)

                        ####### Logging #########
                        if args.save_trajectory:
                            log_actions[ep_step] = action
                            log_s_messages[ep_step] = s_message.squeeze()
                            # print(log_s_messages)
                            log_rewards[ep_step] = reward
                        ##########################

                        ep_step+=1

                    returns += infos[0]['episode']['r'] # torch.sum(reward).cpu()
                    collected_items += infos[0]['episode']['success']
                    running_length += infos[0]['episode']['l']
                    if infos[0]['episode']['success']:
                        running_success_length += infos[0]['episode']['l']
                        num_success_episodes += 1

                    if args.save_trajectory:
                        with torch.no_grad():
                            # message sent from agent1 to agent0 --> use agent0's embedding
                            log_r_message_embs[:, :, 0] = agent0.message_encoder(log_r_messages[:, 0])
                            # message sent from agent0 to agent1 --> use agent1's embedding
                            log_r_message_embs[:, :, 1] = agent1.message_encoder(log_r_messages[:, 1])

                        log_r_message_embs = log_r_message_embs.cpu().numpy()
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
                            "log_r_message_embs": log_r_message_embs,
                            "log_obs": log_obs,
                            "log_locs": log_locs,
                            "log_r_messages": log_r_messages,
                            "log_actions": log_actions,
                            "log_s_messages": log_s_messages,
                            "log_rewards": log_rewards,
                            "episode_length": infos[0]['episode']['l'],
                        }

                    running_rewards += returns

                    if args.visualize: # and returns > 5:
                        print(len(frames))
                        os.makedirs(args.video_save_dir, exist_ok=True)
                        clip = ImageSequenceClip(frames, fps=5)
                        clip.write_videofile(os.path.join(args.video_save_dir, f"ep_{episode_id}_r={infos[0]['episode']['r']}.mp4"), codec="libx264")

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
