import pickle
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import editdistance
from language_analysis import Disent, TopographicSimilarity
import os
from collections import Counter

# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data


def get_episode_length(log_s_messages: np.ndarray) -> int:
    """
    log_s_messages: (T, num_agents) NumPy array with -1 padding after the episode.
    Returns the number of valid time steps in the episode.
    """
    # indices where sentinel -1 appears in agent 0's messages
    end_idxs = np.where(log_s_messages[:, 0] == -1)[0]
    if end_idxs.size == 0:
        # no -1 at all: whole sequence is valid
        return log_s_messages.shape[0]
    else:
        # first -1 marks the end, so its index is the length
        return int(end_idxs[0])


def get_mode_episode_length(log_data) -> int:
    lengths = []
    for episode_id, data in log_data.items():
        log_s_messages = data["log_s_messages"]
        ep_len = get_episode_length(log_s_messages)
        lengths.append(ep_len)

    counts = Counter(lengths)
    mode_length = counts.most_common(1)[0][0]
    return mode_length

def extract_data(log_data):
    # Message length is the mode of the episode length
    mode_length = get_mode_episode_length(log_data)
    message_data = {"agent0": [], "agent1": []}
    attribute_data = []
    scores = {"agent0": [], "agent1": []}
    item_locs = {"agent0": [], "agent1": []}

    for episode_id, data in log_data.items():
        log_s_messages = data["log_s_messages"]
        log_rewards = data["log_rewards"]
        who_see_target = data["who_see_target"]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"]       # (2,)
        distractor_score = data["log_distractor_food_dict"]["score"][0]
        distractor_loc = data["log_distractor_food_dict"]["location"][0]  # (2,)
        # Compute this episode's actual length
        ep_len = get_episode_length(log_s_messages)

        # Option 1: only keep episodes whose length == mode_length
        if ep_len == mode_length and who_see_target == 0:
            messages = log_s_messages[:ep_len, 0].flatten()
            message_data["agent0"].append(messages)
            extract_attribute = [target_score, target_loc[0], target_loc[1]]
            attribute_data.append(extract_attribute)
            
    return message_data, attribute_data


def get_comp_scores(message_data, attribute_data, num_networks):
    '''
    Input: (messages, attributes)
    Output: topsim, posdis
    
    '''
    sender_list = [i for i in range(num_networks)]
    data = []
    n_samples = 1000000
    extracted_message = []
    extracted_attribute = []
    receiver = 0
    avg_topsim = 0
    avg_posdis = 0
    max_eval_episodes=1000
    if num_networks > 2:
        for sender in sender_list:
            extracted_message.append(np.array(message_data[f"{sender}-{receiver}"]["agent0"]))
            extracted_attribute.append(attribute_data[f"{sender}-{receiver}"])
            n_samples = min(extracted_message[sender].shape[0], n_samples)
    else:
        for sender in sender_list:
            receiver_map = {0:1, 1:0}
            receiver = receiver_map[sender]
            # in case of XP n_pop=2, agent cannot successfully play with itself, we need to gather info when it plays with its partner
            extracted_message.append(np.array(message_data[f"{sender}-{receiver}"]["agent0"]))
            extracted_attribute.append(attribute_data[f"{sender}-{receiver}"])
            n_samples = min(extracted_message[sender].shape[0], n_samples)


    for agent_id in range(len(sender_list)):
        messages = np.array(extracted_message[sender_list[agent_id]])
        print(f"message shape {messages.shape}")
        attributes = np.array(extracted_attribute[sender_list[agent_id]])
        topsim = TopographicSimilarity.compute_topsim(attributes[:max_eval_episodes], messages[:max_eval_episodes, :])
        torch_attributes = torch.tensor(attributes[:max_eval_episodes])
        torch_messages = torch.tensor(messages[:max_eval_episodes, :])
        posdis = Disent.posdis(torch_attributes, torch_messages)
        avg_topsim += topsim
        avg_posdis += posdis
        print(f"agent_id {agent_id} has topsim {topsim}, posdis {posdis}")
    avg_topsim /= num_networks
    avg_posdis /= num_networks

    return avg_topsim, avg_posdis

def get_similarity(message_data, num_networks):
    sender_list = [i for i in range(num_networks)]
    data = []
    similarity_mat = np.zeros((num_networks,num_networks))
    n_samples = 1000000
    extracted_message = []
    receiver = 0
    for sender in sender_list:
        extracted_message.append(np.array(message_data[f"{sender}-{receiver}"]["agent0"]))
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



def plot_heatmap(similarity_mat, saved_fig_path, cbar, vmin, vmax):
    # mask = np.zeros_like(similarity_mat)
    # mask[np.triu_indices_from(mask)] = True
    mask = np.triu(np.ones_like(similarity_mat, dtype=bool), k=1)  # Mask only upper triangle (excluding diagonal)
    with sns.axes_style("white"):
        ax = sns.heatmap(
            similarity_mat,
            mask=mask,
            square=True,
            cmap="YlGnBu",
            vmin=vmin,       # Set your desired minimum value
            vmax=vmax,       # Set your desired maximum value
            cbar=cbar,     # Show the color bar
            xticklabels=np.arange(1, similarity_mat.shape[1] + 1),
            yticklabels=np.arange(1, similarity_mat.shape[0] + 1)
        )
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


if __name__ == "__main__":


    checkpoints_dict = {
                        "dec_ppo_invisible" : {"seed1":204800000, "seed2":204800000, "seed3":204800000},
                        "pop_ppo_3net_invisible": {'seed1': 204800000, 'seed2': 204800000, 'seed3':204800000},
                        "pop_ppo_6net_invisible": {'seed1': 460800000, 'seed2': 460800000, 'seed3':460800000},
                        "pop_ppo_9net_invisible": {'seed1': 512000000, 'seed2': 512000000, 'seed3':512000000},
                        "pop_ppo_12net_invisible": {'seed1': 768000000, 'seed2': 768000000, 'seed3':768000000},
                        "pop_ppo_15net_invisible": {'seed1': 819200000, 'seed2': 819200000, 'seed3':819200000},
                        "dec_sp_ppo_invisible" : {'seed1': 204800000, 'seed2': 204800000, 'seed3':204800000},
                        "pop_sp_ppo_3net_invisible": {'seed1': 204800000, 'seed2': 204800000, 'seed3':204800000},
                        "pop_sp_ppo_6net_invisible": {'seed1': 460800000, 'seed2': 460800000, 'seed3':460800000},
                        "pop_sp_ppo_9net_invisible": {'seed1': 512000000, 'seed2': 512000000, 'seed3':512000000},
                        "pop_sp_ppo_12net_invisible": {'seed1': 768000000, 'seed2': 768000000, 'seed3':768000000},
                        "pop_sp_ppo_15net_invisible": {'seed1': 819200000, 'seed2': 819200000, 'seed3':819200000},
                        }
    model2numnet = {
        "dec_ppo_invisible": 2,
        "pop_ppo_3net_invisible": 3,
        "pop_ppo_6net_invisible": 6,
        "pop_ppo_9net_invisible": 9,
        "pop_ppo_12net_invisible": 12,
        "pop_ppo_15net_invisible": 15,
        "dec_sp_ppo_invisible": 2,
        "pop_sp_ppo_3net_invisible": 3,
        "pop_sp_ppo_6net_invisible": 6,
        "pop_sp_ppo_9net_invisible": 9,
        "pop_sp_ppo_12net_invisible": 12,
        "pop_sp_ppo_15net_invisible": 15,
    }
    compute_topsim = True
    cbar = False
    for model_name in checkpoints_dict.keys():
        num_networks = model2numnet[model_name]
        avg_similarity_mat = np.zeros((num_networks,num_networks))
        avg_sr_mat = np.zeros((num_networks,num_networks))
        for seed in range(1,4):
            ckpt_name = checkpoints_dict[model_name][f"seed{seed}"]
            combination_name = f"grid5_img3_ni2_nw4_ms10_{ckpt_name}"

            print(f"{model_name}/{combination_name}")
            saved_fig_dir = f"plots/population/fc/sr_lang_sim"
            saved_score_dir = f"../../logs/vary_n_pop/msg_len_mode/{model_name}/{combination_name}_seed{seed}"
            saved_fig_path_langsim = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_seed{seed}_similarity.pdf")
            saved_fig_path_sr = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_seed{seed}_sr.pdf")
            os.makedirs(saved_fig_dir, exist_ok=True)
            os.makedirs(saved_score_dir, exist_ok=True)
            mode = "test"
            if num_networks <= 2:
                network_pairs = [f"{i}-{j}" for i in range(num_networks) for j in range(num_networks)]
            else:
                network_pairs = [f"{i}-{j}" for i in range(num_networks) for j in range(i+1)]
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
                log_file_path[pair] =  f"../../logs/vary_n_pop/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
                sr_dict[pair] = load_score(f"../../logs/vary_n_pop/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/normal/score.txt")
                sr_mat[row, col] = sr_dict[pair]["Success Rate"]
                if row == col:
                    ic_numerator.append(sr_dict[pair]["Success Rate"])
                else:
                    ic_denominator.append(sr_dict[pair]["Success Rate"])
                # Load log data
                log_data = load_trajectory(log_file_path[pair])

                # Prepare data for t-SNE
                message_data[pair], attribute_data[pair] = extract_data(log_data)


            ic = np.mean(ic_numerator) / np.mean(ic_denominator)
            similarity_mat, avg_sim = get_similarity(message_data, num_networks)
            print(f"Similarity score: {avg_sim} \n matrix: {similarity_mat}")
            print(f"Interchangeability: {ic}")
            # plot_heatmap(similarity_mat, saved_fig_path_langsim)
            # plot_heatmap(sr_mat, saved_fig_path_sr)
            
            if compute_topsim:
                avg_topsim, avg_posdis = get_comp_scores(message_data, attribute_data, num_networks)
                print(f"avg topsim = {avg_topsim}")
                print(f"avg posdis = {avg_posdis}")

                # Save the variables
                np.savez(os.path.join(saved_score_dir, "sim_scores.npz"), similarity_mat=similarity_mat, 
                                                                        avg_sim=avg_sim, 
                                                                        avg_topsim=avg_topsim, 
                                                                        avg_posdis=avg_posdis,
                                                                        sr_mat=sr_mat,
                                                                        ic=ic)
            avg_similarity_mat += similarity_mat
            avg_sr_mat += sr_mat

        avg_similarity_mat /= 3 # 3 seeds
        avg_sr_mat /= 3 # 3 seeds
        np.savez(os.path.join(saved_fig_dir, "avg_sim_sr_mat.npz"), 
                                            avg_similarity_mat=avg_similarity_mat, 
                                            avg_sr_mat=avg_sr_mat)
        plot_heatmap(avg_similarity_mat, saved_fig_path_langsim, cbar, vmin=0.2, vmax=0.6)
        plot_heatmap(avg_sr_mat, saved_fig_path_sr, cbar, vmin=0.3, vmax=1.0)