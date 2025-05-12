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
def extract_message(log_data, decoding_mode):
    message_data = {0: [], 1:[]}
    attributes =  {0: [], 1:[]}
    max_message_length = 14
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

                extracted_message = message_tokens[start_idx:]
                padded_message_tokens[:extracted_length] = np.expand_dims(extracted_message, axis=1)
                if decoding_mode == "token_decoding":
                    message_data[agent_id].append(padded_message_tokens.flatten())  # Collect all time steps for the agent
                elif decoding_mode == "embedding_decoding":
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
    avg_accuracy_dict = {'spawned_time':[], 
                        'item_pos_x':[], 
                        'item_pos_y':[]}
    os.makedirs("reports", exist_ok=True)
    label_list = ['spawned_time', 'item_pos_x', 'item_pos_y']
    model_name = "pop_ppo_3net_invisible"
    checkpoints_dict = {
                    "pop_ppo_3net_invisible": {'seed1': 819200000, 'seed2': 819200000, 'seed3':819200000},
                    }
    mode = "train"
    decoding_mode = "embedding_decoding" # ["embedding_decoding", "token_decoding"]
    saved_dir = f"reports/{decoding_mode}"
    os.makedirs(saved_dir, exist_ok=True)
    for seed in range(1,4):
        model_step = checkpoints_dict[model_name][f"seed{seed}"]
        combination_name = f"grid5_img3_ni2_nw4_ms20_freeze_dur6_{model_step}"
        for i in range(3):
            for j in range(3):
                if i != j:
                    log_file_path = f"../../logs/linear_decode/pickup_temporal/{model_name}/{i}-{j}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
                    agent_id = 0
                    label_encoder = sklearn.preprocessing.LabelEncoder()

                    # Load log data
                    data = load_trajectory(log_file_path)
                    messages, attributes_dict = extract_message(data, decoding_mode)
                    label_dict = extract_label(attributes_dict, agent_id)

                    for k in label_list:
                        groundtruth_name = k
                        message_arr = np.array(messages[agent_id])
                        label_arr = np.array(label_dict[groundtruth_name])

                        # Split data into training and testing sets
                        X_train, X_test, y_train, y_test = train_test_split(
                            message_arr, label_arr, test_size=0.3, random_state=1
                        )
                        
                        clf = LogisticRegression(multi_class="multinomial", penalty="l2", C=1e-3,
                                                solver="lbfgs", max_iter=2000, class_weight="balanced")
                        clf.fit(X_train, y_train)

                        # Evaluate on seen data
                        y_pred = clf.predict(X_test)
                        pred_acc = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True)
                        
                        print(f"Seen Combinations {groundtruth_name}")
                        print(f"Classification Accuracy: {pred_acc:.2f}")
                        avg_accuracy_dict[groundtruth_name].append(pred_acc)

                        report_path = f"{saved_dir}/speaker{i}_listener{j}_{groundtruth_name}.csv"
                        save_classification_report_csv(report, pred_acc, report_path)
    print(avg_accuracy_dict)