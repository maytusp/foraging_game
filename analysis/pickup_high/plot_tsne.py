import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

plot_teacher_agent = False # plot agent that see the target
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

# Extract and prepare data for t-SNE
def prepare_tsne_data(log_data):
    tsne_data = []
    scores = []
    switch_agent = {0:1, 1:0}
    for episode, data in log_data.items():
        # Get sent messages and target food score
        log_s_message_embs = data["log_s_message_embs"]
        who_see_target = data["who_see_target"]
        another_agent = switch_agent[who_see_target]
        if plot_teacher_agent:
            plot_agent = who_see_target
            score = data["log_target_food_dict"]["score"]
            target_loc = data["log_target_food_dict"]["location"] # (2,)
        else:
            plot_agent = another_agent
            score = data["log_distractor_food_dict"]["score"][0]
            target_loc = data["log_distractor_food_dict"]["location"][0] # (2,)
        
        
        agent_locs = data["log_locs"][:,plot_agent] #(num_steps, 2)

        # Calculate the start_idx where the agent first sees the target
        start_idx = None
        for t in range(agent_locs.shape[0]):
            agent_loc = agent_locs[t]
            # Check if the target is within a 5x5 grid centered at the agent's location
            if (target_loc[0] >= agent_loc[0] - 2 and target_loc[0] <= agent_loc[0] + 2 and
                target_loc[1] >= agent_loc[1] - 2 and target_loc[1] <= agent_loc[1] + 2):
                start_idx = t
                break


        # print(f"start_idx{start_idx}")
        if start_idx is None or start_idx > 10:
            print("start_idx none")
            # If the agent never sees the target, skip this episode
            continue

        sent_message_embs = log_s_message_embs[start_idx:start_idx+10, plot_agent].flatten()
        tsne_data.append(sent_message_embs)  # Collect all time steps for the agent
        scores.append(score)  # Same score for all time steps

    # Flatten and convert to NumPy
    tsne_data = np.vstack(tsne_data)
    scores = np.array(scores)
    print(f"tsne_data {tsne_data.shape}")
    return tsne_data, scores

# Plot t-SNE
def plot_tsne(tsne_data, scores):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(tsne_data)

    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=scores, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Target Food Score")
    plt.title("t-SNE of Messages Sent by Agents Seeing Target")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    log_file_path =  "../../logs/pickup_high_moderate_debug/ppo_ps_comm_294M/normal/trajectory.pkl"

    if os.path.exists(log_file_path):
        # Load log data
        log_data = load_trajectory(log_file_path)

        # Prepare data for t-SNE
        tsne_data, scores = prepare_tsne_data(log_data)

        # Plot t-SNE
        plot_tsne(tsne_data, scores)
    else:
        print(f"Log file not found: {log_file_path}")
