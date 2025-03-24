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
from get_compos_score import load_trajectory
import os
from transforms import *

# Extract and prepare data for t-SNE
def extract_message(log_data):
    tsne_data = {"agent0": [], "agent1":[]}
    spawn_time = {"agent0": [], "agent1":[]}
    item_locs = {"agent0": [], "agent1":[]}
    item_times = {"agent0": [], "agent1":[]}
    for episode, data in log_data.items():
        log_s_message_embs = data["log_s_message_embs"]
        log_food_dict = data["log_food_dict"]
        food_loc = log_food_dict["location"] # list
        food_time = log_food_dict["spawn_time"] # list
        message_indices = get_message_indices(data["log_locs"][:, 0], data["log_locs"][:, 1])
        print(message_indices)
        for agent_id in range(2):
            agent_loc = data["log_locs"][:, agent_id] #(num_steps, 2)
            food_id = get_food_id(agent_loc, food_loc)
            # print(f"food id {food_id}")
            # Get sent messages and target food score
            if log_s_message_embs.shape[2] == 2: # num_agents
                message_embs = log_s_message_embs[:, :, agent_id].flatten()
            else:
                message_embs = log_s_message_embs[:, agent_id, :].flatten()
            # print(food_time[food_id])
            message_embs[f"agent{agent_id}"].append(message_embs)  # Collect all time steps for the agent
            item_times[f"agent{agent_id}"].append(food_time[food_id])
            item_locs[f"agent{agent_id}"].append(list(food_loc[food_id]))

    return message_embs, item_times, item_locs

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
            "item_time": item_score_arr,
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
    seen_log_file_path = "../../logs/pickup_high_v1/dec_ppo_invisible0-1/grid5_img3_ni2_nw16_ms10_307200000/seed1/mode_train/normal/trajectory.pkl"
    unseen_log_file_path = "../../logs/pickup_high_v1/dec_ppo_invisible0-1/grid5_img3_ni2_nw16_ms10_307200000/seed1/mode_test/normal/trajectory.pkl"
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
        if "score" in k:
            seen_label_arr = transform_to_range_class(seen_label_arr)
        # seen_label_arr = label_encoder.fit_transform(seen_label_arr)

        # visualise_class(label_arr)
        # Split data into training and testing sets
        X_train, X_test_seen, y_train, y_test_seen = train_test_split(
            seen_message_arr, seen_label_arr, test_size=0.3, random_state=42
        )

        
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
        seen_report_path = f"reports/see_target{groundtruth_name}_seen.csv"
        save_classification_report_csv(report_seen, accuracy_seen, seen_report_path)

        if k == "item_score":
            # Load log data
            unseen_data = load_trajectory(unseen_log_file_path)
            unseen_attributes, unseen_messages = extract_message(unseen_data) # attributes_dict[agent_id][episode_id] -> Dict
            unseen_label_dict = extract_label(unseen_attributes, agent_id=agent_id)

            unseen_message_arr = np.array(unseen_messages[agent_id])
            unseen_label_arr = np.array(unseen_label_dict[groundtruth_name])
            unseen_label_arr = transform_to_range_class(unseen_label_arr)

            X_test_unseen = unseen_message_arr
            y_test_unseen = unseen_label_arr
            # X_train_unseen, X_test_unseen, y_train_unseen, y_test_unseen = train_test_split(
            #     unseen_message_arr, unseen_label_arr, test_size=0.3, random_state=42
            # )
            # X_test_unseen, y_test_unseen = unseen_message_arr, unseen_label_arr

            # clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, class_weight="balanced")
            # clf.fit(X_train_unseen, y_train_unseen)
            # Evaluate on unseen data
            y_pred_unseen = clf.predict(X_test_unseen)
            accuracy_unseen = accuracy_score(y_test_unseen, y_pred_unseen)
            report_unseen = classification_report(y_test_unseen, y_pred_unseen, output_dict=True)
            
            print("Unseen Combinations")
            print(f"Classification Accuracy: {accuracy_unseen:.2f}")
            # print(pd.DataFrame(report_unseen).transpose())

            # Save unseen classification report as CSV
            unseen_report_path = f"reports/see_target{groundtruth_name}_unseen.csv"
            save_classification_report_csv(report_unseen, accuracy_unseen, unseen_report_path)