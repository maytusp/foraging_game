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
from utils import *
from models_v2 import PPOLSTMCommAgent



@dataclass
class Args:
    ckpt_path = "checkpoints/pickup_high_moderate_debug/ppo_ps_comm/model_step_294M.pt"
    range = "custom"
    ablate_message = False
    ablate_type = "zero" # zero, noise
    saved_dir = f"logs/pickup_high_moderate_debug/ppo_ps_comm_5000/{range}"
    video_save_dir = os.path.join(saved_dir, "vids")

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    visualize = False
    log_trajectory = True

    agent_visible = True
    fully_visible_score = False
    identical_item_obs = False
    


    # Algorithm specific arguments
    env_id: str = "Foraging-Single-v1"
    total_episodes: int = 5000
    num_channels = 2
    num_obs_grid = 5



if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.visualize:
        from visualize import *
        from moviepy.editor import *

    if args.range == "normal":
        from environment_pickup_high_moderate_debug import Environment
    elif args.range == "low":
        from environment_pickup_high_moderate_debug_low import Environment
    elif args.range == "custom":
        from environment_pickup_high_moderate_debug_custom_score import Environment

    os.makedirs(args.saved_dir, exist_ok=True)
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = Environment(use_message=True, agent_visible=args.agent_visible, 
                        food_ener_fully_visible=args.fully_visible_score, seed=args.seed,
                        identical_item_obs=args.identical_item_obs)
    grid_size = (env.image_size, env.image_size)
    num_channels = env.num_channels
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0])['action'].n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    # Vectorise env
    envs = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(envs, 1, num_cpus=0, base_class="gymnasium")

    agent = PPOLSTMCommAgent(num_actions=num_actions, num_channels=args.num_channels).to(device)
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
    next_obs_dict, _ = envs.reset()
    next_obs, next_locs, next_eners, next_r_messages = extract_dict(next_obs_dict, device, use_message=True)
    next_r_messages = torch.tensor(next_r_messages).squeeze().to(device)
    

    log_data = {}

    for episode_id in range(1, args.total_episodes + 1):
        energy_obs = {"agent0": set(), "agent1": set()}

        next_done = torch.zeros((num_agents)).to(device)
        returns = 0
        ep_step = 0
        frames = []
        next_lstm_state = (
            torch.zeros(agent.lstm.num_layers, num_agents, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, num_agents, agent.lstm.hidden_size).to(device),
        )

        ############### Logging #########################
        log_obs = torch.zeros((env.max_steps, num_agents, args.num_channels, args.num_obs_grid, args.num_obs_grid)).to(device) # obs: vision
        log_locs = torch.zeros((env.max_steps, num_agents, 2)).to(device) # obs: location
        log_r_messages = torch.zeros((env.max_steps, num_agents), dtype=torch.int64).to(device) # obs: received message
        log_actions = torch.zeros((env.max_steps, num_agents)).to(device) # action: physical action
        log_s_messages = torch.zeros((env.max_steps, num_agents), dtype=torch.int64).to(device) # action: sent message
        log_rewards = torch.zeros((env.max_steps, num_agents)).to(device)
        single_env = envs.vec_envs[0].unwrapped.par_env
        log_target_food_dict = {}
        log_dsitractor_food_dict = {"location":[], "type":[], "score":[]}
        
        target_food_id = single_env.target_food_id
        target_food = single_env.foods[target_food_id]
        log_target_food_dict['location'] = target_food.position
        log_target_food_dict['type'] = target_food.food_type
        log_target_food_dict['score'] = target_food.energy_score

        log_who_see_target = target_food.visible_to_agent

        for food_id in range(len(single_env.foods)):
            if food_id != target_food_id:
                distractor_food = single_env.foods[food_id]
                log_dsitractor_food_dict['location'].append(distractor_food.position)
                log_dsitractor_food_dict['type'].append(distractor_food.food_type)
                log_dsitractor_food_dict['score'].append(distractor_food.energy_score)
        ############################################################
        while not next_done[0]:
            next_obs_arr = next_obs.detach().cpu().numpy()
            energy_obs["agent0"] = energy_obs["agent0"].union(set(next_obs_arr[0,1,:,:].flatten()))
            energy_obs["agent1"] = energy_obs["agent1"].union(set(next_obs_arr[1,1,:,:].flatten()))



            if args.visualize:
                single_env = envs.vec_envs[0].unwrapped.par_env
                frame = visualize_environment(single_env, ep_step)
                frames.append(frame.transpose((1, 0, 2)))
            
            with torch.no_grad():
                if args.ablate_message:
                    if args.ablate_type == "zero":
                        next_r_messages = torch.zeros_like(next_r_messages).to(device)
                    elif args.ablate_type == "noise":
                        next_r_messages = torch.randint(0, 10, next_r_messages.shape)
                    else:
                        raise Exception("only zero and noise are allowed")
                ######### Logging #################
                log_obs[ep_step] = next_obs
                log_locs[ep_step] = next_locs
                log_r_messages[ep_step] = next_r_messages.squeeze()
                ###################################
                action, action_logprob, _, s_message, message_logprob, _, value, next_lstm_state = agent.get_action_and_value((next_obs, next_locs, next_eners, next_r_messages), 
                                                                                                    next_lstm_state, next_done)
                # print(f"step {ep_step} agent_actions = {action}")
            env_action, env_message = action.cpu().numpy(), s_message.cpu().numpy()
            next_obs_dict, reward, terminations, truncations, infos = envs.step({"action": env_action, "message": env_message})
            next_obs, next_locs, next_eners, next_r_messages = extract_dict(next_obs_dict, device, use_message=True)

            next_done = np.logical_or(terminations, truncations)
            next_done = torch.tensor(terminations).to(device)
            next_obs = torch.Tensor(next_obs).to(device)
            next_r_messages = torch.tensor(next_r_messages).to(device)
            reward = torch.tensor(reward).to(device)

            ####### Logging #########
            log_actions[ep_step] = action
            log_s_messages[ep_step] = s_message.squeeze()
            log_rewards[ep_step] = reward
            ##########################
            
            ep_step+=1


        returns += infos[0]['episode']['r'] # torch.sum(reward).cpu()
        collected_items += infos[0]['episode']['success']
        running_length += infos[0]['episode']['l']
        
        # Logged Items:
        if args.log_trajectory:
            with torch.no_grad():
                log_r_message_embs = agent.message_encoder(log_r_messages).cpu().numpy()
                log_s_message_embs = agent.message_encoder(log_s_messages).cpu().numpy()
            log_obs = log_obs.cpu().numpy()
            log_locs = log_locs.cpu().numpy()
            log_r_messages = log_r_messages.cpu().numpy()
            log_actions = log_actions.cpu().numpy()
            log_s_messages = log_s_messages.cpu().numpy()
            log_rewards = log_rewards.cpu().numpy()
            import pickle

            # Combine all your data into a dictionary
            log_data[f"episode_{episode_id}"] = {
                "log_target_food_dict": log_target_food_dict,
                "log_dsitractor_food_dict": log_dsitractor_food_dict,
                "log_r_message_embs": log_r_message_embs,
                "log_s_message_embs": log_s_message_embs,
                "log_obs": log_obs,
                "log_locs": log_locs,
                "log_r_messages": log_r_messages,
                "log_actions": log_actions,
                "log_s_messages": log_s_messages,
                "log_rewards": log_rewards,
                "who_see_target": log_who_see_target
            }


        # Open the log file in append mode
        with open(os.path.join(args.saved_dir, "log.txt"), "a") as log_file:
            
            # Redirect the print statements to the log file
            print(f"EPISODE {episode_id}: {infos[0]['episode']['collect']}", file=log_file)
            print(f"Target Name: {infos[0]['episode']['target_name']}", file=log_file)
            print(f"Item Scores: \n {infos[0]['episode']['food_scores']}", file=log_file)
            print(f"Agent Item Score Observations \n {energy_obs}", file=log_file)
            print(f"Final Score Obs Agent0:  \n {next_obs_arr[0,1,:,:]}", file=log_file)
            print(f"Final Score Obs Agent1:  \n {next_obs_arr[1,1,:,:]}", file=log_file)
        
        running_rewards += returns

        if args.visualize: # and returns > 5:
            os.makedirs(args.video_save_dir, exist_ok=True)
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_videofile(os.path.join(args.video_save_dir, f"ep_{episode_id}_{infos[0]['episode']['target_name']}_r={infos[0]['episode']['r']}.mp4"), codec="libx264")
        
        if infos[0]['episode']['collect'] == len(single_env.foods):
            average_sr += 1

    with open(os.path.join(args.saved_dir, "score.txt"), "a") as log_file:
        print(f"Success Rate: {collected_items / args.total_episodes}", file=log_file)
        print(f"Average Reward {running_rewards / args.total_episodes}", file=log_file)
        print(f"Average Length: {running_length / args.total_episodes}", file=log_file)
    # Save the dictionary to a pickle file
    if args.log_trajectory:
        with open(os.path.join(args.saved_dir, "trajectory.pkl"), "wb") as f:
            pickle.dump(log_data, f)

    envs.close()
    # writer.close()