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
def extract_message(log_data, decoding_mode):
    attributes = {0:[], 1:[]}
    messages = {0:[], 1:[]}
    swap_target = {0:1, 1:0}

    for episode, data in log_data.items():
        # Get sent messages and target food score
        log_s_messages = data["log_s_messages"]
        log_s_message_embs = data["log_s_message_embs"]
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
            # if agent_id == who_see_target:
            if decoding_mode == "token_decoding":
                extract_message = log_s_messages[:, agent_id]
            elif decoding_mode == "embedding_decoding":
                extract_message = log_s_message_embs[:, :, agent_id].flatten()
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
    label_list = ['item_score', 'item_loc_x', 'item_loc_y']
    model_name = "dec_ppo_invisible"
    combination_name = "grid5_img3_ni2_nw4_ms6_38400000"
    seed = 1
    decoding_mode = "token_decoding" # ["embedding_decoding", "token_decoding"]

    seen_log_file_path = f"../../logs/pickup_rg/{model_name}/{combination_name}/seed{seed}/mode_test/normal/trajectory.pkl"
    unseen_log_file_path = f"../../logs/pickup_rg/{model_name}/{combination_name}/seed{seed}/mode_test/normal/trajectory.pkl"
    
    label_encoder = sklearn.preprocessing.LabelEncoder()
    saved_dir = f"reports/{decoding_mode}/{model_name}_{combination_name}_seed{seed}"
    os.makedirs(saved_dir, exist_ok=True)
    avg_accuracy_dict = {'item_score_seen':0, 
                        'item_score_unseen':0, 
                        'item_loc_x':0, 
                        'item_loc_y':0}

    for agent_id in range(2):
        for k in label_list:
            
            groundtruth_name = k

            # Load log data
            seen_data = load_trajectory(seen_log_file_path)
            seen_attributes, seen_messages = extract_message(seen_data, decoding_mode) # attributes_dict[agent_id][episode_id] -> Dict
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

            if k == "item_score":
                avg_accuracy_dict[groundtruth_name+"_seen"] += accuracy_seen/2
            else:
                avg_accuracy_dict[groundtruth_name] += accuracy_seen/2
            # print(pd.DataFrame(report_seen).transpose())
            # Save seen classification report as CSV
            seen_report_path = os.path.join(saved_dir, f"agent_id{agent_id}_{groundtruth_name}_seen.csv")
            save_classification_report_csv(report_seen, accuracy_seen, seen_report_path)

            if k == "item_score":
                # Load log data
                unseen_data = load_trajectory(unseen_log_file_path)
                unseen_attributes, unseen_messages = extract_message(unseen_data, decoding_mode) # attributes_dict[agent_id][episode_id] -> Dict
                unseen_label_dict = extract_label(unseen_attributes, agent_id=agent_id)

                unseen_message_arr = np.array(unseen_messages[agent_id])
                unseen_label_arr = np.array(unseen_label_dict[groundtruth_name])
                unseen_label_arr = transform_to_range_class(unseen_label_arr)

                X_test_unseen = unseen_message_arr
                y_test_unseen = unseen_label_arr

                # Evaluate on unseen data
                y_pred_unseen = clf.predict(X_test_unseen)
                accuracy_unseen = accuracy_score(y_test_unseen, y_pred_unseen)
                report_unseen = classification_report(y_test_unseen, y_pred_unseen, output_dict=True)
                
                print("Unseen Combinations")
                print(f"Classification Accuracy: {accuracy_unseen:.2f}")
                avg_accuracy_dict[groundtruth_name+"_unseen"] += accuracy_unseen / 2
                # print(pd.DataFrame(report_unseen).transpose())

                # Save unseen classification report as CSV
                unseen_report_path = os.path.join(saved_dir, f"agent_id{agent_id}_{groundtruth_name}_unseen.csv")
                save_classification_report_csv(report_unseen, accuracy_unseen, unseen_report_path)
    print(avg_accuracy_dict)