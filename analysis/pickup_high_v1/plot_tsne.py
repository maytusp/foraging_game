import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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
    for episode, data in log_data.items():
        log_s_message_embs = data["log_s_message_embs"]
        who_see_target = data["who_see_target"]
        another_agent = switch_agent[who_see_target]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"] # (2,)
        distractor_score = data["log_distractor_food_dict"]["score"][0]
        distractor_loc = data["log_distractor_food_dict"]["location"][0] # (2,)
        for agent_id in range(2):
            # Get sent messages and target food score
            message_embs = log_s_message_embs[:, :, agent_id].flatten()
            agent_loc = data["log_locs"][:, agent_id] #(num_steps, 2)
            if agent_id == who_see_target:
                score = target_score
                item_loc = target_loc
            else:
                score = distractor_score
                item_loc = distractor_loc
            if agent_id==who_see_target:
                tsne_data[f"agent{agent_id}"].append(message_embs)  # Collect all time steps for the agent
                scores[f"agent{agent_id}"].append(score)  # Same score for all time steps
                item_locs[f"agent{agent_id}"].append([item_loc[0], item_loc[1]])

    return tsne_data, scores, item_locs

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
    plt.title(f"t-SNE of message embeddings sent by {plot_agent}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()

# Plot t-SNE
def plot_tsne_loc(tsne_data, item_locs, plot_agent="agent0", axis_name="y"):
    axis_map = {"x":0, "y":1}
    axis = axis_map[axis_name]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_data = np.vstack(tsne_data[plot_agent])
    item_locs = np.array(item_locs[plot_agent])[:, axis]

    tsne_results = tsne.fit_transform(tsne_data)
    
    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=item_locs, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label=f"Item's position {axis}")
    plt.title(f"t-SNE of message embeddings sent by {plot_agent}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Path to the trajectory .pkl file
    model_name = "dec_ppo_zero_ent_invisible"
    combination_name = "grid5_img3_ni2_nw16_ms10_204800000"
    seed = 1
    mode = "train"
    log_file_path =  f"../../logs/pickup_high_v1/{model_name}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
    plot_agent = "agent0"
    if os.path.exists(log_file_path):
        # Load log data
        log_data = load_trajectory(log_file_path)

        # Prepare data for t-SNE
        tsne_data, scores, item_locs = prepare_tsne_data(log_data)

        # Plot t-SNE
        # plot_tsne(tsne_data, scores , plot_agent)
        plot_tsne_loc(tsne_data, item_locs , plot_agent)
    else:
        print(f"Log file not found: {log_file_path}")
