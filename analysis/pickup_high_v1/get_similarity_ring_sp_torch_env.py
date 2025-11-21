import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import editdistance
from language_analysis import Disent, TopographicSimilarity
import os
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

# Extract and prepare data for t-SNE
def extract_data(log_data):
    message_data = {"agent0": [], "agent1":[]}
    attribute_data = []
    scores = {"agent0": [], "agent1":[]}
    item_locs = {"agent0": [], "agent1":[]}
    for id, (episode, data) in enumerate(log_data.items()):
        log_s_messages = data["log_s_messages"]

        for agent_id in range(2):
            messages = log_s_messages[:, agent_id].flatten()
            message_data[f"agent{agent_id}"].append(messages)  # Collect all time steps for the agent
        
    return message_data, attribute_data


def get_topsim(message_data,attribute_data, num_networks):
    sender_list = [i for i in range(num_networks)]
    data = []
    n_samples = 1000000
    extracted_message = []
    extracted_attribute = []
    receiver = 0
    avg_topsim = 0
    max_eval_episodes=1000
    max_message_length=5
    for sender in sender_list:
        extracted_message.append(np.array(message_data[f"{sender}-{receiver}"]["agent0"]))
        extracted_attribute.append(attribute_data[f"{sender}-{receiver}"])
        n_samples = min(extracted_message[sender].shape[0], n_samples)


    for agent_id in range(len(sender_list)):
        messages = np.array(extracted_message[sender_list[agent_id]])
        attributes = np.array(extracted_attribute[sender_list[agent_id]])
        topsim = TopographicSimilarity.compute_topsim(attributes[:max_eval_episodes], messages[:max_eval_episodes, :max_message_length])     
        avg_topsim += topsim
        print(f"agent_id {agent_id} has topsim {topsim}")
    avg_topsim /= num_networks

    return avg_topsim

def get_similarity(message_data, num_networks):
    sender_list = [i for i in range(num_networks)]
    data = []
    similarity_mat = np.zeros((num_networks,num_networks))
    n_samples = 1000000
    extracted_message = []

    for sender in sender_list:
        if sender == num_networks-1:
            receiver = 0
        else:
            receiver = sender + 1
        print(f"{receiver}-{sender}")
        extracted_message.append(np.array(message_data[f"{receiver}-{sender}"]["agent0"]))
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
"gpt_200k_sp_ring_ppo_15net_invisible" : {'seed1':599998464, 'seed2':599998464, 'seed3':599998464}

}
    cbar = False
    for num_networks in [15]: # [3,6,9,12,15]:
        avg_similarity_mat = np.zeros((num_networks,num_networks))
        avg_sr_mat = np.zeros((num_networks,num_networks))
        for seed in range(1,4):
            # model_name = f"ring_ppo_{num_networks}net_invisible"
            model_name = f"gpt_200k_sp_ring_ppo_{num_networks}net_invisible"
            ckpt_name = checkpoints_dict[model_name][f"seed{seed}"]
            combination_name = f"grid5_img3_ni2_nw4_ms10_{ckpt_name}"

            print(f"{model_name}/{combination_name}")
            saved_fig_dir = f"plots/population/ring/sr_lang_sim"
            saved_score_dir = f"../../logs/ring_pop/torch_pickup_high_v1/{model_name}/{combination_name}_seed{seed}"
            saved_fig_path_langsim = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_similarity.pdf")
            saved_fig_path_sr = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_sr.pdf")
            os.makedirs(saved_fig_dir, exist_ok=True)
            os.makedirs(saved_score_dir, exist_ok=True)
            mode = "test"
            network_pairs = [f"{i}-{j}" for i in range(num_networks) for j in range(num_networks)]
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
                log_file_path[pair] =  f"../../logs/torch_pickup_high_v1/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/trajectory.pkl"
                sr_dict[pair] = load_score(f"../../logs/torch_pickup_high_v1/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/score.txt")
                sr_mat[row, col] = sr_dict[pair]["Success Rate"]
                if row == col:
                    ic_numerator.append(sr_dict[pair]["Success Rate"])
                elif min(abs(row-col), num_networks-abs(row-col)) == 1:
                    ic_denominator.append(sr_dict[pair]["Success Rate"])
                # Load log data
                log_data = load_trajectory(log_file_path[pair])

                # Prepare data for t-SNE
                message_data[pair], attribute_data[pair] = extract_data(log_data)
                

            ic = np.mean(ic_numerator) / np.mean(ic_denominator)
            similarity_mat, avg_sim = get_similarity(message_data, num_networks)
            print(f"Similarity score: {avg_sim} \n matrix: {similarity_mat}")
            print(f"Interchangeability: {ic}")


            avg_similarity_mat += similarity_mat
            avg_sr_mat += sr_mat

        avg_similarity_mat /= 3 # 3 seeds
        avg_sr_mat /= 3 # 3 seeds
        np.savez(os.path.join(saved_fig_dir, "avg_sim_sr_mat.npz"), 
                                            avg_similarity_mat=avg_similarity_mat, 
                                            avg_sr_mat=avg_sr_mat)
        plot_heatmap(avg_similarity_mat, saved_fig_path_langsim, cbar, vmin=0.2, vmax=0.6)
        plot_heatmap(avg_sr_mat, saved_fig_path_sr, cbar, vmin=0.3, vmax=1.0)