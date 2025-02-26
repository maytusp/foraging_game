# Edit: 20Dec2024
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
# CUDA_VISIBLE_DEIVCES=1 python -m scripts.goal_pickup.test_comm_ppo_separate
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
from environments.partial_goal_pickup import *
from utils.process_data import *
from models.goal_cond_models import PPOLSTMCommAgentGoal



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
    agent_visible = True
    fully_visible_score = False
    identical_item_obs = False
    zero_memory = False
    memory_transfer = False
    
    # Algorithm specific arguments
    env_id: str = "Foraging-Single-v1"
    total_episodes: int = 5000
    n_words = 16
    """vocab size"""
    image_size = 3
    """number of observation grid"""
    N_att = 4
    """number of attributes"""
    N_val = 10
    """number of values"""
    N_i = 2
    """number of items"""
    grid_size = 7
    """grid size"""
    mode = "train"
    agent_visible = False
    model_name = "dec_ppo_invisible_possig"
    model_step = "998400000"
    combination_name = f"grid{grid_size}_img{image_size}_ni{N_i}_natt{N_att}_nval{N_val}_nw{n_words}"
    ckpt_path = f"checkpoints/partial_goal_pickup/{model_name}/{combination_name}/seed{seed}/agent_0_step_{model_step}.pt"
    ckpt_path2 = f"checkpoints/partial_goal_pickup/{model_name}/{combination_name}/seed{seed}/agent_1_step_{model_step}.pt"
    saved_dir = f"logs/partial_goal_pickup/{model_name}/{combination_name}_{model_step}/seed{seed}/mode_{mode}"
    if ablate_message:
        saved_dir = os.path.join(saved_dir, ablate_type)
    else:
        saved_dir = os.path.join(saved_dir, "normal")
    print("saved at: ", saved_dir)
    video_save_dir = os.path.join(saved_dir, "vids")


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.visualize:
        from visualize import *
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
                        N_att=args.N_att,
                        N_val=args.N_val,
                        N_i = args.N_i,
                        grid_size=args.grid_size,
                        image_size=args.image_size,
                        mode=args.mode)

    num_channels = env.num_channels
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0])['action'].n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    # Vectorise env
    envs = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(envs, 1, num_cpus=0, base_class="gymnasium")

    agent0 = PPOLSTMCommAgentGoal(num_actions=num_actions, 
                                grid_size=args.grid_size, 
                                n_words=args.n_words, 
                                embedding_size=16, 
                                num_channels=num_channels, 
                                N_val=args.N_val, 
                                N_att=args.N_att, 
                                image_size=args.image_size,
                                ).to(device)
    agent0.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    agent0.eval()

    agent1 = PPOLSTMCommAgentGoal(num_actions=num_actions, 
                                grid_size=args.grid_size, 
                                n_words=args.n_words, 
                                embedding_size=16, 
                                num_channels=num_channels, 
                                N_val=args.N_val, 
                                N_att=args.N_att, 
                                image_size=args.image_size,
                                ).to(device)
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
    next_obs, next_locs, next_goals, next_r_messages = extract_dict_with_goal(next_obs_dict, device, use_message=True)

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

        log_r_message_embs = torch.zeros((env.max_steps, agent0.embedding_size, num_agents)).to(device) # obs: received message
        
    
        single_env = envs.vec_envs[0].unwrapped.par_env
        log_target_food_dict = {}
        log_foods = {}
        
        target_food_id = single_env.target_food_id
        target_food = single_env.foods[target_food_id]
        log_target_food_dict['location'] = target_food.position
        log_target_food_dict['attribute'] = target_food.attribute
        log_foods["attribute"] = [food.attribute for food in single_env.foods]
        log_foods["position"] = [food.position for food in single_env.foods]
        ############### Logging #########################

        while not next_done[0]:
            # print(f"step {ep_step}")
            next_obs_arr = next_obs.detach().cpu().numpy()
            energy_obs["agent0"] = energy_obs["agent0"].union(set(next_obs_arr[0,1,:,:].flatten()))
            energy_obs["agent1"] = energy_obs["agent1"].union(set(next_obs_arr[1,1,:,:].flatten()))

            if args.visualize and not(args.save_trajectory):
                single_env = envs.vec_envs[0].unwrapped.par_env
                frame = visualize_environment(single_env, ep_step)
                frames.append(frame.transpose((1, 0, 2)))

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


                action0, _, _, s_message0, _, _, _, (new_h0, new_c0) = agent0.get_action_and_value((next_obs[0].unsqueeze(0),
                                                                                            next_locs[0].unsqueeze(0), 
                                                                                            next_goals[0].unsqueeze(0),
                                                                                            next_r_messages[0].unsqueeze(0)),
                                                                                            (h0,c0), next_done[0])
                action1, _, _, s_message1, _, _, _, (new_h1, new_c1) = agent1.get_action_and_value((next_obs[1].unsqueeze(0),
                                                                                            next_locs[1].unsqueeze(0),
                                                                                            next_goals[1].unsqueeze(0),
                                                                                            next_r_messages[1].unsqueeze(0)),
                                                                                            (h1,c1), next_done[1])

                action = torch.cat((action0, action1), dim=0)
                s_message = torch.cat((s_message0, s_message1), dim=0)
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
            next_obs, next_locs, next_goals, next_r_messages = extract_dict_with_goal(next_obs_dict, device, use_message=True)
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
            log_masks = infos[0]['episode']['attribute_masks']
            log_attributes = infos[0]['episode']['food_attributes']
            log_goal = infos[0]['episode']['goal_attribute']
            with torch.no_grad():
                # agent0 embedding
                log_r_message_embs[:, :, 0] = agent0.message_encoder(log_r_messages[:, 0])
                # agent1 embedding
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
                "log_foods": log_foods,
                "log_target_food_dict": log_target_food_dict,
                "log_target_food_id" : target_food_id,
                "log_attributes" : log_attributes,
                "log_goal" : log_goal,
                "log_masks" : log_masks,
                "log_r_messages": log_r_messages,
                "log_s_messages": log_s_messages,
                "log_r_message_embs": log_r_message_embs,
                "log_obs": log_obs,
                "log_locs": log_locs,
                "log_actions": log_actions,
                "log_rewards": log_rewards,
            }

        if not(args.save_trajectory):
            # Open the log file in append mode
            with open(os.path.join(args.saved_dir, "log.txt"), "a") as log_file:
                
                # Redirect the print statements to the log file
                print(f"EPISODE {episode_id}: {infos[0]['episode']['collect']}", file=log_file)
                print(f"Target ID: {infos[0]['episode']['target_id']}", file=log_file)
                print(f"Goal Attribute: {infos[0]['episode']['goal_attribute']}", file=log_file)
                print(f"Food Attributes: \n {infos[0]['episode']['food_attributes']}", file=log_file)
                print(f"Agent Item Score Observations \n {energy_obs}", file=log_file)
                print(f"Final Score Obs Agent0:  \n {next_obs_arr[0,1,:,:]}", file=log_file)
                print(f"Final Score Obs Agent1:  \n {next_obs_arr[1,1,:,:]}", file=log_file)

        running_rewards += returns

        if args.visualize: # and returns > 5:
            os.makedirs(args.video_save_dir, exist_ok=True)
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_videofile(os.path.join(args.video_save_dir, f"ep_{episode_id}_r={infos[0]['episode']['r']}.mp4"), codec="libx264")

    # Save the dictionary to a pickle file
    if args.save_trajectory:
        with open(os.path.join(args.saved_dir, "trajectory.pkl"), "wb") as f:
            pickle.dump(log_data, f)

    with open(os.path.join(args.saved_dir, "score.txt"), "a") as log_file:
        print(f"Success Rate: {collected_items / args.total_episodes}", file=log_file)
        print(f"Average Reward {running_rewards / args.total_episodes}", file=log_file)
        print(f"Average Length: {running_length / args.total_episodes}", file=log_file)
        print(f"Average Success Length: {running_success_length / num_success_episodes}", file=log_file)
    
    envs.close()
    # writer.close()
