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
from get_compos_score import load_trajectory, within_receptive_field, compute_first_seen_time_indices
import os

def extract_message(log_data, N_att=2, N_i=2, window_size=8, lag_time=0, use_all_message=False):
    attributes = {0:[], 1:[]}
    messages = {0:[], 1:[]}
    swap_target = {0:1, 1:0}
    
    neg_episode = 0
    for episode, data in log_data.items():
        # Get sent messages and target food score
        log_s_message = data["log_s_messages"]
        log_s_message_embs = data["log_s_message_embs"]
        log_masks = data["log_masks"]
        log_attributes = data["log_attributes"]
        log_goal = np.array(data["log_goal"])
        log_foods = data["log_foods"]
        log_locs = data["log_locs"] # (max_steps, num_agents, 2)
        log_target_id = data["log_target_food_id"]
        log_rewards = data["log_rewards"][:, 0]

        max_timesteps = log_locs.shape[0]
        num_agents = 2
        first_seen_time_indices = compute_first_seen_time_indices(log_locs, log_foods, receptive_field_size=5, N_i=N_i)
        
        # Check if any row contains -1
        # has_neg_one = np.any(first_seen_time_indices == -1, axis=1)
        has_neg_one = np.any(first_seen_time_indices == -1)
        check_window_size = np.any(19-first_seen_time_indices < window_size+lag_time)
        if not(has_neg_one or check_window_size):
            for agent_id in range(num_agents):
                start_idx_list = []
                for item_id in range(N_i):
                    
                    start_idx = int(first_seen_time_indices[agent_id, item_id])
                    start_idx_list.append(start_idx)

                    agent_pos = log_locs[start_idx, agent_id, :] # x_a, y_a
                    agent_goal = np.array(log_goal)  # g1,g2,...
                    agent_mask = log_masks[agent_id] # mask

                    item_pos = log_foods["position"][item_id] # x_i, y_i
                    item_att = np.array(log_attributes[item_id]) # a1,a2,...
                    diff_att = agent_goal - item_att # g1-a1, g2-a2,...
                    mse = np.expand_dims(np.mean(diff_att**2), axis=0)
                    mask_att = item_att * agent_mask
                    mask_att_diff = diff_att * agent_mask
                    mask_mse = np.expand_dims(np.mean(mask_att_diff**2), axis=0)

                    start_idx += lag_time
                    if use_all_message:
                        start_idx = 0
                        window_size=max_timesteps

                    extract_message = log_s_message[start_idx:start_idx+window_size, agent_id]
                    extract_attribute = {
                                        "agent_pos":agent_pos,  # agent position
                                        "agent_goal":agent_goal, # agent goal
                                        "agent_mask":agent_mask, # agent mask e.g., [1 0 0 1] 
                                        "item_pos":item_pos, # item position
                                        "att": item_att, # item's attribute
                                        "mask_att":mask_att, # item's masked attribute
                                        "mask_att_diff":mask_att_diff, # item's masked attribute difference
                                        "mse":mse,
                                        "mask_mse":mask_mse,
                                        }
                    
                    messages[agent_id].append(extract_message)  # Collect all time steps for the agent
                    attributes[agent_id].append(extract_attribute)
        else:
            neg_episode+=1

    print(f"Total unused episodes: {neg_episode}")
    return attributes, messages

def extract_label(attributes_dict, agent_id=0):
    num_episodes = len(attributes_dict[agent_id])
    # extract labels
    agent_pos_x_arr = []
    agent_pos_y_arr = []
    agent_mask_att0_arr = []
    agent_mask_att1_arr = []
    item_pos_x_arr = []
    item_pos_y_arr = []
    item_att0_arr = []
    item_att1_arr = []
    item_mask_att0_arr = []
    item_mask_att1_arr = []
    att0_diff_arr = []
    att1_diff_arr = []
    mse_arr = []
    mask_mse_arr = []
    for ep in range(num_episodes):
        ep_data = attributes_dict[agent_id][ep]
        agent_pos_x_arr.append(ep_data["agent_pos"][0])
        agent_pos_y_arr.append(ep_data["agent_pos"][1])
        agent_mask_att0_arr.append(ep_data["agent_mask"][0])
        agent_mask_att1_arr.append(ep_data["agent_mask"][1])
        item_pos_x_arr.append(ep_data["item_pos"][0])
        item_pos_y_arr.append(ep_data["item_pos"][1])
        item_att0_arr.append(ep_data["att"][0])
        item_att1_arr.append(ep_data["att"][1])
        item_mask_att0_arr.append(ep_data["mask_att"][0])
        item_mask_att1_arr.append(ep_data["mask_att"][1])
        att0_diff_arr.append(ep_data["mask_att_diff"][0])
        att1_diff_arr.append(ep_data["mask_att_diff"][1])
        mse_arr.append(ep_data["mse"])
        mask_mse_arr.append(ep_data["mask_mse"])
    return {
            "agent_pos_x": agent_pos_x_arr,
            "agent_pos_y": agent_pos_y_arr,
            "agent_mask_att0": agent_mask_att0_arr,
            "agent_mask_att1": agent_mask_att1_arr,
            "item_pos_x": item_pos_x_arr,
            "item_pos_y": item_pos_y_arr,
            "item_att0": item_att0_arr,
            "item_att1": item_att1_arr,
            "item_mask_att0": item_mask_att0_arr,
            "item_mask_att1": item_mask_att1_arr,
            "att0_diff": att0_diff_arr,
            "att1_diff": att1_diff_arr,
            "mse": mse_arr,
            "mask_mse": mask_mse_arr
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
    label_list = ['agent_pos_x', 'agent_pos_y', 'agent_mask_att0', 'agent_mask_att1', 'item_pos_x', 'item_pos_y', 'item_att0', 'item_att1', 'item_mask_att0', 'item_mask_att1', 'att0_diff', 'att1_diff'] # 'mse', 'mask_mse']
    seen_log_file_path = "../logs/goal_condition_pickup/dec_ppo_invisible_possig/grid5_img5_ni2_natt2_nval10_nw16_1B/seed1/mode_train/normal/trajectory.pkl"
    unseen_log_file_path = "../logs/goal_condition_pickup/dec_ppo_invisible_possig/grid5_img5_ni2_natt2_nval10_nw16_1B/seed1/mode_test/normal/trajectory.pkl"
    agent_id = 0
    label_encoder = sklearn.preprocessing.LabelEncoder()
    for k in label_list:
        groundtruth_name = k


        # Load log data
        seen_data = load_trajectory(seen_log_file_path)
        seen_attributes, seen_messages = extract_message(seen_data) # attributes_dict[agent_id][episode_id] -> Dict
        seen_label_dict = extract_label(seen_attributes, agent_id=agent_id)

        seen_message_arr = np.array(seen_messages[agent_id])
        seen_label_arr = np.array(seen_label_dict[groundtruth_name])
        # seen_label_arr = label_encoder.fit_transform(seen_label_arr)

        # Load log data
        unseen_data = load_trajectory(unseen_log_file_path)
        unseen_attributes, unseen_messages = extract_message(unseen_data) # attributes_dict[agent_id][episode_id] -> Dict
        unseen_label_dict = extract_label(unseen_attributes, agent_id=agent_id)
        
        unseen_message_arr = np.array(unseen_messages[agent_id])
        unseen_label_arr = np.array(unseen_label_dict[groundtruth_name])
        # unseen_label_arr = label_encoder.fit_transform(unseen_label_arr)

        # visualise_class(label_arr)
        # Split data into training and testing sets
        X_train, X_test_seen, y_train, y_test_seen = train_test_split(
            seen_message_arr, seen_label_arr, test_size=0.3, random_state=42
        )

        X_test_unseen, y_test_unseen = unseen_message_arr, unseen_label_arr
        # Train a linear classifier (Logistic Regression)
        # clf = DecisionTreeClassifier(random_state=0, class_weight='balanced')
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)

        # Evaluate on seen data
        y_pred_seen = clf.predict(X_test_seen)
        accuracy_seen = accuracy_score(y_test_seen, y_pred_seen)
        report_seen = classification_report(y_test_seen, y_pred_seen, output_dict=True)
        
        print(f"Seen Combinations {groundtruth_name}")
        print(f"Classification Accuracy: {accuracy_seen:.2f}")
        # print(pd.DataFrame(report_seen).transpose())

        # Save seen classification report as CSV
        seen_report_path = f"reports/{groundtruth_name}_seen.csv"
        save_classification_report_csv(report_seen, accuracy_seen, seen_report_path)

        # Evaluate on unseen data
        y_pred_unseen = clf.predict(X_test_unseen)
        accuracy_unseen = accuracy_score(y_test_unseen, y_pred_unseen)
        report_unseen = classification_report(y_test_unseen, y_pred_unseen, output_dict=True)
        
        print("Unseen Combinations")
        print(f"Classification Accuracy: {accuracy_unseen:.2f}")
        # print(pd.DataFrame(report_unseen).transpose())

        # Save unseen classification report as CSV
        unseen_report_path = f"reports/{groundtruth_name}_unseen.csv"
        save_classification_report_csv(report_unseen, accuracy_unseen, unseen_report_path)