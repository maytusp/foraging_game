import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
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
    see_target = {"agent0": [], "agent1":[]}
    switch_agent = {0:1, 1:0}
    max_timesteps = 6
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
            if log_s_message_embs.shape[2] == 2: # num_agents
                message_embs = log_s_message_embs[:max_timesteps, :, agent_id].flatten()
            else:
                message_embs = log_s_message_embs[:max_timesteps, agent_id, :].flatten()
            # message_embs = data["log_s_messages"][:, agent_id].flatten()
            agent_loc = data["log_locs"][:, agent_id] #(num_steps, 2)
            if agent_id == who_see_target:
                score = target_score
                item_loc = target_loc
            else:
                score = distractor_score
                item_loc = distractor_loc

            tsne_data[f"agent{agent_id}"].append(message_embs)  # Collect all time steps for the agent
            scores[f"agent{agent_id}"].append(score)  # Same score for all time steps
            item_locs[f"agent{agent_id}"].append([item_loc[0], item_loc[1]])
            see_target[f"agent{agent_id}"].append(agent_id==who_see_target)
    return tsne_data, scores, item_locs, see_target

# Plot t-SNE
def plot_tsne(tsne_results, scores, plot_agent="agent0", saved_fig_path=None):
    scores = scores[plot_agent]
    print(tsne_data.shape)
    
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24
    })
    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=scores, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Item's Score")
    plt.title(f"t-SNE of message embeddings sent by {plot_agent}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig(saved_fig_path+"_score")
    plt.show()


# Plot t-SNE
def plot_tsne_loc(tsne_results, item_locs, plot_agent="agent0", saved_fig_path=None):
    item_locs = np.int32(np.array(item_locs[plot_agent]))

    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24
    })
    
    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=item_locs[:, 0], palette='tab10', s=50, alpha=0.7)
    plt.legend(title="item's position y")
    plt.title(f"t-SNE of message embeddings sent by {plot_agent}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig(saved_fig_path+"_position_y")
    # plt.show()

    
    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))

    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=item_locs[:, 1], palette='tab10', s=50, alpha=0.7)
    plt.legend(title="item's position x")
    plt.title(f"t-SNE of message embeddings sent by {plot_agent}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig(saved_fig_path+"_position_x")
    plt.show()


if __name__ == "__main__":
    # Path to the trajectory .pkl file
    model_name = f"pop_ppo_3net_invisible"
    combination_name = "grid5_img3_ni2_nw4_ms10_332800000"
    seed = 1
    mode = "test"
    log_file_path =   f"../../logs/linear_decode/pickup_high_v1/{model_name}/0-0/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
    plot_agent = "agent0"
    saved_fig_dir = f"plots/tsne_scores/"
    os.makedirs(saved_fig_dir, exist_ok=True)
    saved_fig_path = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_seed{seed}.pdf")

    if os.path.exists(log_file_path):
        # Load log data
        log_data = load_trajectory(log_file_path)

        # Prepare data for t-SNE
        tsne_data, scores, item_locs, see_target = prepare_tsne_data(log_data)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_data = np.vstack(tsne_data[plot_agent])
        tsne_results = tsne.fit_transform(tsne_data)
        # Plot t-SNE
        plot_tsne(tsne_results, scores , plot_agent, saved_fig_path)
        plot_tsne_loc(tsne_results, item_locs , plot_agent, saved_fig_path)
        # plot_tsne_see_target(tsne_results, see_target, plot_agent)
        # plot_tsne_loc_see_target(tsne_results, item_locs, see_target, plot_agent)
    else:
        print(f"Log file not found: {log_file_path}")
