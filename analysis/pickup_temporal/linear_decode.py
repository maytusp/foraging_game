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


# Extract and prepare data for t-SNE
def extract_message(log_data):
    message_data = {0: [], 1:[]}
    attributes =  {0: [], 1:[]}
    max_message_length = 32
    for episode, data in log_data.items():
        log_r_message_embs = data["log_r_message_embs"]
        log_r_message_tokens = data["log_r_messages"]
        log_food_dict = data["log_food_dict"]
        food_loc = log_food_dict["location"] # list
        food_time = log_food_dict["spawn_time"] # list
        episode_length = data["episode_length"]
        message_indices = get_message_indices(data["log_locs"][:, 0], data["log_locs"][:, 1], episode_length)

        for agent_id in range(2):
            reciever_id = {0:1, 1:0}[agent_id]
            agent_loc = data["log_locs"][:, agent_id] #(num_steps, 2)
            food_id = get_food_id(agent_loc, food_loc)
            # Get sent messages and target food score
            if log_r_message_embs.shape[2] == 2: # num_agents
                message_embs = log_r_message_embs[:, :, reciever_id]
            else:
                message_embs = log_r_message_embs[:, reciever_id, :]
            message_dim = message_embs.shape[1]
            padded_message_embs = np.zeros((max_message_length, message_dim))
            padded_message_tokens = np.zeros((max_message_length, 1))
            message_tokens = log_r_message_tokens[:, reciever_id]
            curr_message_length = len(message_indices)
            
            # print(f"curr_message_length {curr_message_length}")
            # print(f"message_embs {message_embs.shape}")
            # print(food_time[food_id])
            if curr_message_length <= max_message_length and len(message_indices) > 2:
                start_idx = message_indices[0]
                extracted_message_embs = message_embs[start_idx:]
                extracted_length = extracted_message_embs.shape[0]
                padded_message_embs[:extracted_length] = extracted_message_embs

                message_data[agent_id].append(padded_message_embs.flatten())  # Collect all time steps for the agent
                extract_attribute = {
                                    "item_times": food_time[food_id],
                                    "item_locations": list(food_loc[food_id]),
                                    }
                attributes[agent_id].append(extract_attribute)

    return message_data, attributes

def extract_label(attributes_dict, agent_id):
    num_episodes = len(attributes_dict[agent_id])
    # extract labels
    item_time_arr = []
    item_loc_x_arr = []
    item_loc_y_arr = []
    for ep in range(num_episodes):
        ep_data = attributes_dict[agent_id][ep]
        item_time_arr.append(ep_data["item_times"])
        item_loc_x_arr.append(ep_data["item_locations"][0])
        item_loc_y_arr.append(ep_data["item_locations"][1])

    return {
            "spawned_time": item_time_arr,
            "item_pos_x": item_loc_x_arr,
            "item_pos_y": item_loc_y_arr,

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
    label_list = ['spawned_time', 'item_pos_x', 'item_pos_y']
    model_name = "dec_ppo_invisible"
    combination_name = "grid8_img3_ni2_nw4_ms40_51200000"
    seed = 1
    mode = "test"
    seen_log_file_path = f"../../logs/pickup_temporal/{model_name}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
    agent_id = 0
    label_encoder = sklearn.preprocessing.LabelEncoder()

    # Load log data
    seen_data = load_trajectory(seen_log_file_path)
    messages, attributes_dict = extract_message(seen_data)
    seen_label_dict = extract_label(attributes_dict, agent_id)

    for k in label_list:
        groundtruth_name = k
        seen_message_arr = np.array(messages[agent_id])
        seen_label_arr = np.array(seen_label_dict[groundtruth_name])

        # Split data into training and testing sets
        X_train, X_test_seen, y_train, y_test_seen = train_test_split(
            seen_message_arr, seen_label_arr, test_size=0.4, random_state=1
        )
        
        clf = LogisticRegression(multi_class="multinomial", penalty="l2", C=1e-3,
                                solver="lbfgs", max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)

        # Evaluate on seen data
        y_pred_seen = clf.predict(X_test_seen)
        accuracy_seen = accuracy_score(y_test_seen, y_pred_seen)
        report_seen = classification_report(y_test_seen, y_pred_seen, output_dict=True)
        
        print(f"Seen Combinations {groundtruth_name}")
        print(f"Classification Accuracy: {accuracy_seen:.2f}")
        # print(pd.DataFrame(report_seen).transpose())
        # Save seen classification report as CSV
        seen_report_path = f"reports/see_target{groundtruth_name}_seen.csv"
        save_classification_report_csv(report_seen, accuracy_seen, seen_report_path)