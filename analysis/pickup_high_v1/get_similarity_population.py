import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import editdistance
import os
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

# Extract and prepare data for t-SNE
def extract_data(log_data):
    message_data = {"agent0": [], "agent1":[]}
    scores = {"agent0": [], "agent1":[]}
    item_locs = {"agent0": [], "agent1":[]}
    switch_agent = {0:1, 1:0}
    for id, (episode, data) in enumerate(log_data.items()):

        log_s_messages = data["log_s_messages"]
        who_see_target = data["who_see_target"]
        another_agent = switch_agent[who_see_target]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"] # (2,)
        distractor_score = data["log_distractor_food_dict"]["score"][0]
        distractor_loc = data["log_distractor_food_dict"]["location"][0] # (2,)

        for agent_id in range(2):
            messages = log_s_messages[:, agent_id].flatten()
            message_data[f"agent{agent_id}"].append(messages)  # Collect all time steps for the agent


    return message_data



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


def plot_heatmap(similarity_mat, saved_fig_path):
    # mask = np.zeros_like(similarity_mat)
    # mask[np.triu_indices_from(mask)] = True
    mask = np.triu(np.ones_like(similarity_mat, dtype=bool), k=1)  # Mask only upper triangle (excluding diagonal)
    with sns.axes_style("white"):
        ax = sns.heatmap(similarity_mat, mask=mask, square=True,  cmap="YlGnBu")
        plt.savefig(saved_fig_path)
        plt.show()
        

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    
    model_name = "pop_ppo_2net_selfplay_invisible"
    combination_name = "grid5_img3_ni2_nw16_ms10_307200000"
    seed = 1
    saved_fig_dir = f"figs"
    saved_fig_path = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_seed{seed}_similarity.png")
    os.makedirs(saved_fig_dir, exist_ok=True)
    mode = "train"
    num_networks = 2
    network_pairs = [f"{i}-{j}" for i in range(num_networks) for j in range(num_networks)]
    log_file_path = {}
    message_data = {}
    for pair in network_pairs:
        print(f"loading network pair {pair}")
        log_file_path[pair] =  f"../../logs/pickup_high_v1/{model_name}{pair}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
    
        
        # Load log data
        log_data = load_trajectory(log_file_path[pair])

        # Prepare data for t-SNE
        message_data[pair] = extract_data(log_data)


    similarity_mat, avg_sim = get_similarity(message_data, num_networks)
    print(f"Similarity score: {avg_sim} \n matrix: {similarity_mat}")
    plot_heatmap(similarity_mat, saved_fig_path)
    

