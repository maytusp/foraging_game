import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

# Extract and prepare data for t-SNE
def prepare_tsne_data(log_data):
    tsne_data = {"agent0": [], "agent1":[]}
    scores = {"agent0": [], "agent1":[]}
    item_locs = {"agent0": [], "agent1":[]}
    switch_agent = {0:1, 1:0}
    for id, (episode, data) in enumerate(log_data.items()):

        log_s_message_embs = data["log_s_message_embs"]
        who_see_target = data["who_see_target"]
        another_agent = switch_agent[who_see_target]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"] # (2,)
        distractor_score = data["log_distractor_food_dict"]["score"][0]
        distractor_loc = data["log_distractor_food_dict"]["location"][0] # (2,)
        # if id<10:
        #     print(f"Episode {id}")
        #     print(f"who_see_target {who_see_target}")
        #     print(f"target_score loc {target_score} {target_loc}")
        #     print(f"distractor score loc {distractor_score} {distractor_loc}")
        for agent_id in range(2):
            # Get sent messages and target food score
            message_embs = log_s_message_embs[:, :, agent_id].flatten()
            # agent_loc = data["log_locs"][:, agent_id] #(num_steps, 2)
            # if agent_id == who_see_target:
            #     score = target_score
            #     item_loc = target_loc
            # else:
            #     score = distractor_score
            #     item_loc = distractor_loc
            # if agent_id == who_see_target:
            tsne_data[f"agent{agent_id}"].append(message_embs)  # Collect all time steps for the agent
            # scores[f"agent{agent_id}"].append(score)  # Same score for all time steps
            # item_locs[f"agent{agent_id}"].append([item_loc[0], item_loc[1]])

    return tsne_data



def plot_tsne_by_agent(tsne_data):
    sender_list = [0, 1]
    data = []
    agent_ids = []

    for sender in sender_list:
        message_emb = np.array(tsne_data[f"{sender}-1"]["agent0"])
        n_samples = message_emb.shape[0]
        data.append(message_emb)
        agent_ids.append(np.array([sender] * n_samples))

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    data = np.vstack(data)
    agent_ids = np.hstack(agent_ids)  # Flatten agent ID array
    tsne_results = tsne.fit_transform(data)

    # Define colors and markers for each class
    colors = ["red", "blue"]
    labels = ["Agent 0", "Agent 1"]

    plt.figure(figsize=(10, 8))

    # Plot each class separately
    for i, agent_id in enumerate(sender_list):
        mask = agent_ids == agent_id
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                    color=colors[i], label=labels[i], alpha=0.7)

    # Add legend
    plt.legend(title="Sent by Agent")
    plt.title("t-SNE of Message Embeddings by Agents")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    model_name = "dec_ppo_invisible"
    combination_name = "grid5_img3_ni2_nw16_ms10_204800000"
    seed = 1
    mode = "train"
    network_pairs = ["0-0", "0-1", "1-1"]
    log_file_path = {}
    receiver = "agent1"
    tsne_data = {}
    for pair in network_pairs:
        if receiver[-1] in pair[-1]:
            print(f"loading network pair {pair}")
            log_file_path[pair] =  f"../../logs/pickup_high_v1/{model_name}{pair}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
        
            
            # Load log data
            log_data = load_trajectory(log_file_path[pair])

            # Prepare data for t-SNE
            tsne_data[pair] = prepare_tsne_data(log_data)
    # print(np.array(tsne_data["0-1"]["agent0"]).shape)
    plot_tsne_by_agent(tsne_data)

