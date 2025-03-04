import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import distance
import os
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

# Extract and prepare data for t-SNE
def extract_data(log_data):
    agent_value = 127
    message_data = {"agent0": [], "agent1":[]}
    scores = {"agent0": [], "agent1":[]}
    item_locs = {"agent0": [], "agent1":[]}
    switch_agent = {0:1, 1:0}
    for id, (episode, data) in enumerate(log_data.items()):
        log_obs = data["log_obs"]
        log_s_messages = data["log_s_messages"]
        who_see_target = data["who_see_target"]
        another_agent = switch_agent[who_see_target]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"] # (2,)
        distractor_score = data["log_distractor_food_dict"]["score"][0]
        distractor_loc = data["log_distractor_food_dict"]["location"][0] # (2,)
        image_obs = log_obs[:,:, 0,:,:] # (steps, num_agents, channels, img_size, img_size)
        image_size = image_obs.shape[3]
        center = image_size // 2

        # remove the observation of itself
        image_obs[:,:,center,center] = 0
        time_steps = image_obs.shape[0]

        for agent_id in range(2):
            obs_position = []
            for t in range(time_steps):
                # print(f"time {t}")
                agent_obs = image_obs[t, agent_id, :, :]
                another_agent_pos = np.where(agent_obs.flatten() == agent_value)
                if len(another_agent_pos[0]) > 0:
                    obs_position.append(another_agent_pos[0][0]+1)
                else:
                    obs_position.append(0)
            if agent_id == who_see_target:
                score = distractor_score
            else:
                score = target_score

            print(len(obs_position))
            message_data[f"agent{agent_id}"].append(obs_position)  # Collect all time steps for the agent
            scores[f"agent{agent_id}"].append(score)
    return message_data, scores
    
# Plot t-SNE
def plot_tsne(tsne_data, scores, plot_agent="agent0"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_data = np.vstack(tsne_data[plot_agent])
    scores = scores[plot_agent]
    print(tsne_data.shape)
    tsne_results = tsne.fit_transform(tsne_data)

    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=scores, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Item's Score")
    plt.title(f"t-SNE of another agent position")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    model_name = "dec_ppo"
    combination_name = "grid5_img5_ni2_nw16_ms10_204800000"
    seed = 1
    mode = "train"

    log_file_path =  f"../../logs/pickup_high_v2/{model_name}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
    
    # Load log data
    log_data = load_trajectory(log_file_path)

    # Prepare data for t-SNE
    message_data, scores = extract_data(log_data)

    plot_tsne(message_data, scores)