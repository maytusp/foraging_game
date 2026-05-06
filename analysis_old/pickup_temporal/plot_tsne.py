import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import *


plt.rcParams.update({'font.size': 16})
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data
        
# Extract and prepare data for t-SNE
def prepare_tsne_data(log_data):
    max_message_length = 5
    count_uselsss =  0
    message_emb_dict = {"agent0": [], "agent1":[]}
    message_token_dict = {"agent0": [], "agent1":[]}
    spawn_time = {"agent0": [], "agent1":[]}
    item_locs = {"agent0": [], "agent1":[]}
    item_times = {"agent0": [], "agent1":[]}
    for episode, data in log_data.items():
        log_r_message_embs = data["log_r_message_embs"]
        log_r_message_tokens = data["log_r_messages"]
        
        log_r_message_tokens = data["log_r_messages"]
        log_rewards = data["log_rewards"]
        episode_length = data["episode_length"]
        success = np.max(log_rewards) >= 1
        log_food_dict = data["log_food_dict"]
        food_loc = log_food_dict["location"] # list
        food_time = log_food_dict["spawn_time"] # list
        message_indices = get_message_indices(data["log_locs"][:, 0], data["log_locs"][:, 1], episode_length)

        for agent_id in range(2):
            receiver_id =  {0:1, 1:0}[agent_id]
            agent_loc = data["log_locs"][:, agent_id] #(num_steps, 2)
            food_id = get_food_id(agent_loc, food_loc)
            
            if log_r_message_embs.shape[2] == 2: # num_agents
                message_embs = log_r_message_embs[:, :, receiver_id]
            else:
                message_embs = log_r_message_embs[:, receiver_id, :]
            message_dim = message_embs.shape[1]
            padded_message_embs = np.zeros((max_message_length, message_dim))
            padded_message_tokens = np.zeros((max_message_length, 1))
            message_tokens = log_r_message_tokens[:, receiver_id]
            curr_message_length = len(message_indices)
            if curr_message_length <= max_message_length:
                padded_message_embs[:curr_message_length] = message_embs[message_indices]
                padded_message_embs = padded_message_embs.flatten()
            
            
                # print(food_time[food_id])
                message_emb_dict[f"agent{agent_id}"].append(message_embs.flatten())  # Collect all time steps for the agent
                message_token_dict[f"agent{agent_id}"].append(message_tokens.flatten()) 
                item_times[f"agent{agent_id}"].append(food_time[food_id])
                item_locs[f"agent{agent_id}"].append(list(food_loc[food_id]))
            else:
                count_uselsss += 1
            

    print(f"Too Long Message: {count_uselsss}")
    return message_emb_dict, message_token_dict, item_times, item_locs

# Plot t-SNE
def plot_tsne(tsne_results, scores, plot_agent="agent0"):
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
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=scores, palette='tab10', s=50, alpha=0.7)
    plt.legend(title="item's spawned time")
    plt.title(f"t-SNE of message embeddings")
    plt.xlabel("Dim.1")
    plt.ylabel("Dim.2")
    plt.grid(True)
    
    plt.savefig("plots/tsne_time.pdf")
    plt.show()


# Plot t-SNE
def plot_tsne_loc(tsne_results, item_locs, plot_agent="agent0"):
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
    plt.legend(title="vertical position")
    plt.title(f"t-SNE of message embeddings")
    plt.xlabel("Dim. 1")
    plt.ylabel("Dim. 2")
    plt.grid(True)
    plt.savefig("plots/pickup_time_tsne_vertical.pdf")
    # plt.show()

    
    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=item_locs[:, 1], palette='tab10', s=50, alpha=0.7)
    plt.legend(title="horizontal position")
    plt.title(f"t-SNE of message embeddings")
    plt.xlabel("Dim.1")
    plt.ylabel("Dim.2")
    plt.grid(True)
    plt.savefig("plots/pickup_time_tsne_horizontal.pdf")
    plt.show()


if __name__ == "__main__":
    # Path to the trajectory .pkl file
    model_name = "pop_ppo_3net_invisible"
    combination_name = "grid5_img3_ni2_nw4_ms20_freeze_dur6_819200000"
    seed = 1
    mode = "train"
    log_file_path =  f"../../logs/linear_decode/pickup_temporal/{model_name}/0-1/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
    plot_agent = "agent0"
    if os.path.exists(log_file_path):
        # Load log data
        log_data = load_trajectory(log_file_path)

        # Prepare data for t-SNE
        message_emb_dict, message_token_dict, item_times, item_locs = prepare_tsne_data(log_data)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_data = np.vstack(message_emb_dict[plot_agent])
        tsne_results = tsne.fit_transform(tsne_data)
        # Plot t-SNE
        plot_tsne(tsne_results, item_times , plot_agent)
        plot_tsne_loc(tsne_results, item_locs , plot_agent)
        # plot_tsne_see_target(tsne_results, see_target, plot_agent)
        # plot_tsne_loc_see_target(tsne_results, item_locs, see_target, plot_agent)
    else:
        print(f"Log file not found: {log_file_path}")
