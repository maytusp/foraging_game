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

def extract_message(log_data):
    attributes = {0:[], 1:[]}
    messages = {0:[], 1:[]}
    swap_target = {0:1, 1:0}

    for episode, data in log_data.items():
        # Get sent messages and target food score
        log_s_messages = data["log_s_messages"]
        who_see_target = data["who_see_target"]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"] # (2,)
        distractor_score = data["log_distractor_food_dict"]["score"][0]
        distractor_loc = data["log_distractor_food_dict"]["location"][0] # (2,)

        max_timesteps = log_s_messages.shape[0]
        num_agents = 2

        for agent_id in range(num_agents):
            if agent_id == who_see_target:
                score = target_score
                item_loc = target_loc
            else:
                score = distractor_score
                item_loc = distractor_loc

            extract_message = log_s_messages[:, agent_id]
            extract_attribute = {
                                "item_score": score,
                                "item_location": item_loc,
                                }
            
            messages[agent_id].append(extract_message)  # Collect all time steps for the agent
            attributes[agent_id].append(extract_attribute)

    return attributes, messages

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
    label_list = ['item_score']# , 'item_loc_x', 'item_loc_y']
    seen_log_file_path = "../../logs/pickup_high_v1/dec_ppo_invisible/grid5_img3_ni2_nw16_ms10_307200000/seed1/mode_train/normal/trajectory.pkl"
    unseen_log_file_path = "../../logs/pickup_high_v1/dec_ppo_invisible/grid5_img3_ni2_nw16_ms10_307200000/seed1/mode_test/normal/trajectory.pkl"
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
        seen_report_path = f"reports/{groundtruth_name}_seen.csv"
        save_classification_report_csv(report_seen, accuracy_seen, seen_report_path)

        if k == "item_score":
            # Load log data
            unseen_data = load_trajectory(unseen_log_file_path)
            unseen_attributes, unseen_messages = extract_message(unseen_data) # attributes_dict[agent_id][episode_id] -> Dict
            unseen_label_dict = extract_label(unseen_attributes, agent_id=agent_id)

            unseen_message_arr = np.array(unseen_messages[agent_id])
            unseen_label_arr = np.array(unseen_label_dict[groundtruth_name])

            X_train_unseen, X_test_unseen, y_train_unseen, y_test_unseen = train_test_split(
                unseen_message_arr, unseen_label_arr, test_size=0.3, random_state=42
            )
            X_test_unseen, y_test_unseen = unseen_message_arr, unseen_label_arr

            clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, class_weight="balanced")
            clf.fit(X_train_unseen, y_train_unseen)
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