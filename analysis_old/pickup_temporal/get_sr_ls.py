# Language should be able to be understood by the language user
# Compare: DecTr, HybTr, PopTr

import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import editdistance

import os
import pandas as pd
from utils import *
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data


# Extract and prepare data for t-SNE
def extract_data(log_data):
    message_data = {0: [], 1:[]}
    attributes =  {0: [], 1:[]}
    last_step = 20
    max_message_length=20
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
            padded_message_tokens = np.zeros((max_message_length,)) - 1
            message_tokens = log_r_message_tokens[:, reciever_id]
            curr_message_length = len(message_indices)
            
            if len(message_indices) > 2:
                if message_indices[-1] < last_step:
                    padded_message_tokens[message_indices] = message_tokens[message_indices]
                    start_idx = message_indices[0]

                    message_data[agent_id].append(padded_message_tokens.flatten())  # Collect all time steps for the agent
                    extract_attribute = {
                                        "item_times": food_time[food_id],
                                        "item_locations": list(food_loc[food_id]),
                                        }
                    attributes[agent_id].append(extract_attribute)

    return message_data, attributes



def get_similarity(message_data, num_networks):
    sender_list = [i for i in range(num_networks)]
    data = []
    similarity_mat = np.zeros((num_networks,num_networks))
    n_samples = 1000000
    extracted_message = []
    receiver = 0
    for sender in sender_list:
        extracted_message.append(np.array(message_data[f"{sender}-{receiver}"][0]))
        n_samples = min(extracted_message[sender].shape[0], n_samples)


    for first_agent_id in range(len(sender_list)):
        for second_agent_id in range(len(sender_list)):
            for i in range(n_samples):          
                m1 = extracted_message[sender_list[first_agent_id]][i]
                m2 = extracted_message[sender_list[second_agent_id]][i]

                m1 = [i for i in m1 if i != -1]
                m2 = [i for i in m2 if i != -1]

                dist = editdistance.eval(m1, m2) / max(len(m1), len(m2))
               
                similarity_mat[first_agent_id,second_agent_id] += (1 - dist)

    similarity_mat = similarity_mat / n_samples
    mask = np.ones_like(similarity_mat)
    mask[np.triu_indices_from(mask, k=0)] = 0
    avg_sim = np.sum(similarity_mat * mask) / np.sum(mask)

    return similarity_mat , avg_sim


def plot_heatmap(similarity_mat, saved_fig_path):
    # mask = np.zeros_like(similarity_mat)
    # mask[np.triu_indices_from(mask)] = True
    mask = np.triu(np.ones_like(similarity_mat, dtype=bool), k=1)  # Mask only upper triangle (excluding diagonal)
    with sns.axes_style("white"):
        ax = sns.heatmap(similarity_mat, mask=mask, square=True,  cmap="YlGnBu")
        plt.savefig(saved_fig_path)
        plt.close()

def load_score(filename):
    scores = {}
    with open(filename, "r") as f:
        for line in f:
            x = line.strip().split(": ")
            if "Reward" in x[0]:
                continue
            scores[x[0].strip()] = float(x[1].strip())
    return scores

results_dict = {}  # Store metrics for each model


if __name__ == "__main__":
    checkpoints_dict = {"pop_ppo_3net_invisible" : {"seed1":819200000, "seed2":819200000, "seed3":819200000},
                        }

    for model_name in checkpoints_dict.keys():
        if "3net" in model_name:
            num_networks = 3
        else:
            num_networks = 2
        self_sr_list = []
        cross_sr_list = []
        ic_list = []
        ls_list = []
        for seed in range(1,4):
            ckpt_name = checkpoints_dict[model_name][f"seed{seed}"]
            combination_name = f"grid5_img3_ni2_nw4_ms20_freeze_dur6_{ckpt_name}"
            print(f"{model_name}/{combination_name}")
            saved_fig_dir = f"figs/population"
            saved_score_dir = f"../../logs/pickup_temporal/exp2/{model_name}/{combination_name}_seed{seed}"
            saved_fig_path_langsim = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_seed{seed}_similarity.png")
            saved_fig_path_sr = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_seed{seed}_sr.png")
            os.makedirs(saved_fig_dir, exist_ok=True)
            os.makedirs(saved_score_dir, exist_ok=True)
            mode = "train"
            network_pairs = [f"{i}-{j}" for i in range(num_networks) for j in range(num_networks)]
            # network_pairs = ["0-1", "1-1", "0-0"]

            log_file_path = {}
            sr_dict = {}
            sr_mat = np.zeros((num_networks, num_networks))
            message_data = {}
            attribute_data = {}
            
            # For Interchangeability
            ic_numerator = []
            ic_denominator = []

            for pair in network_pairs:
                row, col = pair.split("-")
                row, col = int(row), int(col)
                print(f"loading network pair {pair}")
                log_file_path[pair] =  f"../../logs/vary_grid_size/pickup_temporal/test_grid_size_5/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
                sr_dict[pair] = load_score(f"../../logs/vary_grid_size/pickup_temporal/test_grid_size_5/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/normal/score.txt")
                sr_mat[row, col] = sr_dict[pair]["Success Rate"]
                if row == col:
                    ic_numerator.append(sr_dict[pair]["Success Rate"])
                else:
                    ic_denominator.append(sr_dict[pair]["Success Rate"])
                # Load log data
                log_data = load_trajectory(log_file_path[pair])

                # Prepare data for t-SNE
                message_data[pair], _ = extract_data(log_data)

                # print(message_data[pair])

            self_sr = np.mean(ic_numerator)
            cross_sr = np.mean(ic_denominator)
            ic = self_sr / cross_sr
            
            similarity_mat, avg_sim = get_similarity(message_data, num_networks)

            # After computing each seed's metrics
            self_sr_list.append(self_sr)
            cross_sr_list.append(cross_sr)
            ic_list.append(ic)
            ls_list.append(avg_sim)

        # Save results
        results_dict[model_name] = {
            "self_sr_mean": np.mean(self_sr_list),
            "self_sr_std": np.std(self_sr_list),
            "cross_sr_mean": np.mean(cross_sr_list),
            "cross_sr_std": np.std(cross_sr_list),
            "ic_mean": np.mean(ic_list),
            "ic_std": np.std(ic_list),
            "ls_mean": np.mean(ls_list),
            "ls_std": np.std(ls_list),
        }

    # Format as DataFrame
    df_data = {
        "Model": [],
        "Self SR (mean ± std)": [],
        "Cross SR (mean ± std)": [],
        "IC (mean ± std)": [],
        "LS (mean ± std)" : [],
    }

    for model, metrics in results_dict.items():
        df_data["Model"].append(model)
        df_data["Cross SR (mean ± std)"].append(f"{metrics['cross_sr_mean']:.3f} ± {metrics['cross_sr_std']:.3f}")
        df_data["Self SR (mean ± std)"].append(f"{metrics['self_sr_mean']:.3f} ± {metrics['self_sr_std']:.3f}")
        df_data["IC (mean ± std)"].append(f"{metrics['ic_mean']:.3f} ± {metrics['ic_std']:.3f}")
        df_data["LS (mean ± std)"].append(f"{metrics['ls_mean']:.3f} ± {metrics['ls_std']:.3f}")

    df = pd.DataFrame(df_data)
    print(df)

    # Save to CSV
    df.to_csv("table1.csv", index=False)