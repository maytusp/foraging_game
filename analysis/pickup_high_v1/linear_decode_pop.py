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
from collections import Counter

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
    item_loc_r_arr = [] # row posiiton
    item_loc_c_arr = [] # column position
    for ep in range(num_episodes):
        ep_data = attributes_dict[agent_id][ep]
        item_score_arr.append(ep_data["item_score"])
        item_loc_r_arr.append(ep_data["item_location"][0])
        item_loc_c_arr.append(ep_data["item_location"][1])

    return {
            "score": item_score_arr,
            "vertical position": item_loc_r_arr,
            "horizontal position": item_loc_c_arr,

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
    label_list = ['score', 'vertical position', 'horizontal position']
    model_name = "pop_ppo_3net_invisible"
    log_dir = "../../logs/pickup_high_v7/5k_test_eps/"
    saved_dir_prefix = "reports/pickup_high_v7/"
    checkpoints_dict = {
                    "pop_ppo_3net_invisible": {'seed1': 614400000, 'seed2': 614400000, 'seed3':614400000},
    }
    avg_accuracy_dict = {k:[] for k in label_list}
    decoding_mode = "embedding_decoding" # ["embedding_decoding", "token_decoding"]
    for seed in range(1,4):
        model_step = checkpoints_dict[model_name][f"seed{seed}"]
        # combination_name = f"grid5_img3_ni2_nw4_ms10_{model_step}" 
        combination_name = f"grid5_img5_ni2_nw4_ms10_{model_step}"  # for generalization across positions
        
        for i in range(3):
            for j in range(i+1):
                log_file_path = os.path.join(log_dir, f"{model_name}/{i}-{j}/{combination_name}/seed{seed}/mode_test/normal/trajectory.pkl")

                saved_dir = os.path.join(saved_dir_prefix, f"{decoding_mode}/{model_name}_{combination_name}_seed{seed}")
                os.makedirs(saved_dir, exist_ok=True)

                agent_id = 0
                for k in label_list:
                    
                    groundtruth_name = k

                    # Load log data
                    data = load_trajectory(log_file_path)
                    attributes, messages = extract_message(data, decoding_mode) # attributes_dict[agent_id][episode_id] -> Dict
                    label_dict = extract_label(attributes, agent_id=agent_id)

                    message_arr = np.array(messages[agent_id])
                    label_arr = np.array(label_dict[groundtruth_name])
                    if "score" in k:
                        label_arr = transform_to_range_class(label_arr)
                    else:
                        print(f"possible {k}: {Counter(label_arr)}")
                    # visualise_class(label_arr)
                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        message_arr, label_arr, test_size=0.3, random_state=42
                    )

                    
                    clf = LogisticRegression(multi_class="multinomial", penalty="l2", C=1e-3,
                        solver="lbfgs", max_iter=2000, class_weight="balanced")
                    clf.fit(X_train, y_train)

                    # Evaluate on seen data
                    y_pred = clf.predict(X_test)
                    pred_acc = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    print(f"Classification Accuracy: {pred_acc:.2f}")



                    avg_accuracy_dict[groundtruth_name].append(pred_acc)

                    report_path = os.path.join(saved_dir, f"speaker{i}_listener{j}_{groundtruth_name}.csv")
                    save_classification_report_csv(report, pred_acc, report_path)

    print(avg_accuracy_dict)