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
from transforms import *
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data
def count_separate_pickup(log_data):
    '''
    Count the number of episodes that two agents end up picking up the different objects
    '''
    attributes = {0:[], 1:[]}
    messages = {0:[], 1:[]}
    swap_target = {0:1, 1:0}
    count_ep = 0
    for episode, data in log_data.items():
        # Get sent messages and target food score
        log_locs = data["log_locs"] # (num_steps, num_agents, 2)
        log_actions = data["log_actions"] # (num_steps, num_agents, 1)
        log_rewards = data["log_rewards"] # (num_steps, num_agents, 1)
        
        max_steps = log_rewards.shape[0]
        
        # find the last time step
        last_step = 0
        for t in range(max_steps):
            if log_rewards[t, 0] != 0:
                last_step = t
                break

        if last_step==0:
            '''
            Sanity Check
            '''
            print("ERROR")

        agent0_last_loc = np.array(log_locs[last_step, 0, :])
        agent1_last_loc = np.array(log_locs[last_step, 1, :])

        diff = agent0_last_loc - agent1_last_loc
        if np.linalg.norm(diff) <= 3 and log_rewards[last_step,0] <0: 
        # the distance between two agents at the last time step is more than the maximum (sqrt(8)) that agents can pick up object
            count_ep += 1

            # print("-------")
            # print(f"episode {episode}")
            # print(agent0_last_loc)
            # print(agent1_last_loc)
        

    return count_ep

def extract_bump(log_data):
    '''
    Bump happens when two agents are 
    '''
    attributes = {0:[], 1:[]}
    messages = {0:[], 1:[]}
    swap_target = {0:1, 1:0}
    total_bump = 0 
    success_bump = 0 # bumps counted in success episodes
    failed_bump = 0 # bumps counted in failed episodes
    count_sucess = 0
    count_fail = 0
    for episode, data in log_data.items():
        # Get sent messages and target food score
        log_bump = data["log_bump"]
        log_success = data["log_success"]

        total_bump += log_bump
        if log_success:
            success_bump += log_bump
            count_sucess += 1
        else:
            failed_bump += log_bump
            count_fail += 1


    return total_bump, success_bump, failed_bump, count_sucess, count_fail

def extract_label(attributes_dict, agent_id=0):
    num_episodes = len(attributes_dict[agent_id])
    # extract labels
    item_score_arr = []
    item_loc_x_arr = []
    item_loc_y_arr = []
    for ep in range(num_episodes):
        ep_data = attributes_dict[agent_id][ep]
        item_score_arr.append(ep_data["item_score"])
        item_loc_x_arr.append(ep_data["item_location"][0])
        item_loc_y_arr.append(ep_data["item_location"][1])

    return {
            "item_score": item_score_arr,
            "item_loc_x": item_loc_x_arr,
            "item_loc_y": item_loc_y_arr,

            }
def visualise_class(agent_pos_x_arr):

    # Get unique position values and their frequencies
    unique_positions, counts = np.unique(agent_pos_x_arr, return_counts=True)

    # Visualize distribution
    plt.figure(figsize=(10, 5))
    plt.bar(unique_positions, counts, color='blue', alpha=0.7)
    plt.xlabel("Label Values")
    plt.ylabel("Frequency")
    plt.xticks(unique_positions)  # Ensure all unique values are displayed
    plt.show()


def save_classification_report_csv(report_dict, accuracy, filename):
    """Saves classification report as CSV."""
    df = pd.DataFrame(report_dict).transpose()
    df.loc["accuracy"] = ["", "", "", accuracy]  # Append accuracy as a row
    df.to_csv(filename, index=True)

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    label_list = ['item_score', 'item_loc_x', 'item_loc_y']
    model_name = "pop_ppo_3net_invisible"
    checkpoints_dict = {
                    "pop_ppo_3net_invisible_ablate_message": {'seed1': 358400000, 'seed2': 358400000, 'seed3':358400000},
                    "pop_ppo_3net_ablate_message": {'seed1': 358400000, 'seed2': 358400000, 'seed3':358400000},
                    "pop_ppo_3net_invisible" : {'seed1': 332800000, 'seed2': 332800000, 'seed3':332800000},

                    }
    avg_accuracy_dict = {'item_score':[], 
                        'item_loc_x':[], 
                        'item_loc_y':[]}
    decoding_mode = "embedding_decoding" # ["embedding_decoding", "token_decoding"]
    total_bump = 0
    success_bump = 0
    failed_bump = 0
    num_test_episodes = 1000
    total_test_episodes = 0
    total_no_pickup = 0
    for seed in range(1,4):
        model_step = checkpoints_dict[model_name][f"seed{seed}"]
        combination_name = f"grid5_img3_ni2_nw4_ms10_{model_step}"
        
        
        for i in range(3):
            for j in range(i+1):
                print(f"PAIR {i} {j}")
                log_file_path = f"../../logs/ablate_message_during_train/pickup_high_v1/{model_name}/{i}-{j}/{combination_name}/seed{seed}/mode_test/hard/trajectory.pkl"
                data = load_trajectory(log_file_path)

                count_ep = count_separate_pickup(data)
                total_no_pickup += count_ep
                total_test_episodes += num_test_episodes

                pair_total_bump, pair_success_bump, pair_failed_bump, count_sucess, count_fail = extract_bump(data) # attributes_dict[agent_id][episode_id] -> Dict
                total_bump += pair_total_bump #/ num_test_episodes
                success_bump += pair_success_bump #/ count_sucess
                failed_bump += pair_failed_bump #/ count_fail
                print("cross-check", count_sucess+count_fail)


    print(f"TOTAL BUMPS: {total_bump / total_test_episodes}")
    print(f"SUCESS BUMPS: {success_bump / total_test_episodes}")
    print(f"FAILED BUMPS: {failed_bump / total_test_episodes}")

    # print(f"PROPORTION OF NO-PICKUP EPS {total_no_pickup / total_test_episodes}")