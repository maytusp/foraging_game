import pickle
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import sklearn.preprocessing
import pandas as pd

import matplotlib.pyplot as plt
import os
from utils import *

def extract_rendezvous_map(log_data, grid_size):
    max_message_length = 14
    rendezvous_map = np.zeros((grid_size, grid_size))
    for episode, data in log_data.items():
        log_r_message_embs = data["log_r_message_embs"]
        log_r_message_tokens = data["log_r_messages"]
        log_actions = data["log_actions"]
        log_food_dict = data["log_food_dict"]

        food_loc = log_food_dict["location"] # list
        food_time = log_food_dict["spawn_time"] # list
        episode_length = data["episode_length"]
        message_indices = get_message_indices(data["log_locs"][:, 0], data["log_locs"][:, 1], episode_length)
        agent_loc = data["log_locs"] #(num_steps, num_agents, 2)
        agent_action = data["log_actions"] # (num_steps, num_agents, 1)
        for t in range(len(agent_loc[:,0])):
            agent_loc0 = np.array(agent_loc[t, 0])
            agent_loc1 = np.array(agent_loc[t, 1])
            diff = agent_loc0 - agent_loc1
            if abs(diff[0]) <= 1 and abs(diff[1]) <= 1:
            # Check if agent is in adjacent grid
                rendezvous_map[int(agent_loc0[0]), int(agent_loc0[1])] += 1
                rendezvous_map[int(agent_loc1[0]), int(agent_loc1[1])] += 1
                break

    return rendezvous_map


if __name__ == "__main__":
    model_name = "pop_ppo_3net_invisible"
    checkpoints_dict = {
                    "pop_ppo_3net_invisible": {'seed1': 819200000, 'seed2': 819200000, 'seed3':819200000},
                    "pop_ppo_3net_ablate_message" : {'seed1': 460800000, 'seed2': 460800000, 'seed3':460800000},
                    "pop_ppo_3net_invisible_ablate_message" :  {'seed1': 460800000, 'seed2': 460800000, 'seed3':460800000},

                    }
    mode = "train"
    grid_size = 5
    message_mode = "normal"
    total_rendezvous_map = np.zeros((grid_size, grid_size))
    test_episodes = 1000
    pair = 0
    for seed in range(1,2):
        model_step = checkpoints_dict[model_name][f"seed{seed}"]
        combination_name = f"grid{grid_size}_img3_ni2_nw4_ms20_freeze_dur6_{model_step}"
        for i in range(3):
            for j in range(3):
                log_file_path = f"../../logs/pickup_temporal/ablate/{model_name}/{i}-{j}/{combination_name}/seed{seed}/mode_{mode}/{message_mode}/trajectory.pkl"
                # Load log data
                data = load_trajectory(log_file_path)
                pair_rendezvous_map = extract_rendezvous_map(data, grid_size)
                total_rendezvous_map += pair_rendezvous_map
                pair += 1
    total_rendezvous_map = np.round(100*(total_rendezvous_map) / (2*pair*test_episodes), 1)
    print("% of episode that agents meet", np.sum(total_rendezvous_map))
    print(total_rendezvous_map)