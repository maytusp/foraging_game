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
    tsne_data = []
    locations = []

    for episode, data in log_data.items():
        # Get sent messages and target food score
        log_s_message_embs = data["log_s_message_embs"]
        who_see_target = data["who_see_target"]

        location = data["log_target_food_dict"]["location"]
        location = f"x={location[0]}, y={location[1]}"
        print(location)
        target_loc = data["log_target_food_dict"]["location"] # (2,)
        agent_locs = data["log_locs"][:,who_see_target] #(num_steps, 2)
        # Calculate the start_idx where the agent first sees the target
        start_idx = None
        for t in range(agent_locs.shape[0]):
            agent_loc = agent_locs[t]
            # Check if the target is within a 5x5 grid centered at the agent's location
            if (target_loc[0] >= agent_loc[0] - 2 and target_loc[0] <= agent_loc[0] + 2 and
                target_loc[1] >= agent_loc[1] - 2 and target_loc[1] <= agent_loc[1] + 2):
                start_idx = t
                break


        print(f"start_idx{start_idx}")
        if start_idx is None or start_idx > 10:
            print("start_idx none")
            # If the agent never sees the target, skip this episode
            continue

        sent_message_embs = log_s_message_embs[start_idx+2:start_idx+5, who_see_target].flatten()
        tsne_data.append(sent_message_embs)  # Collect all time steps for the agent
        locations.append(location)  # Same score for all time steps

    # Flatten and convert to NumPy
    tsne_data = np.vstack(tsne_data)
    locations = np.array(locations)
    print(f"tsne_data {tsne_data.shape}")
    return tsne_data, locations

# Plot t-SNE
def plot_tsne(tsne_data, locations):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(tsne_data)

    # Map string locations to unique numeric values
    unique_locations = list(set(locations))
    location_to_num = {loc: i for i, loc in enumerate(unique_locations)}
    numeric_locations = np.array([location_to_num[loc] for loc in locations])

    # Scatter plot with grouping by location
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=numeric_locations, cmap="viridis", alpha=0.7)
    colorbar = plt.colorbar(scatter, ticks=range(len(unique_locations)))
    colorbar.set_label("Target Location")
    colorbar.set_ticks(range(len(unique_locations)))
    colorbar.set_ticklabels(unique_locations)
    plt.title("t-SNE of Messages Sent by Agents Seeing Target")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Path to the trajectory .pkl file
    log_file_path = "../logs/pickup_high_easy/ppo_ps_comm_550M/normal/trajectory.pkl"

    if os.path.exists(log_file_path):
        # Load log data
        log_data = load_trajectory(log_file_path)

        # Prepare data for t-SNE
        tsne_data, locations = prepare_tsne_data(log_data)

        # Plot t-SNE
        plot_tsne(tsne_data, locations)
    else:
        print(f"Log file not found: {log_file_path}")
