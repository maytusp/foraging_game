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

# check what item id the agent see
def get_food_id(agent_loc, food_loc_list):
    dist_list = []
    for food_loc in food_loc_list:
        food_loc = np.array(food_loc)
        curr_dist = np.linalg.norm(food_loc - agent_loc)
        dist_list.append(curr_dist)
        # print(curr_dist)
    return np.argmin(np.array(dist_list))

def check_comm_range(agent0_pos, agent1_pos, comm_range=1):
    '''
    check whether agents can communicate or not
    '''
    if (agent0_pos[0] >= agent1_pos[0] - comm_range and agent0_pos[0] <= agent1_pos[0] + comm_range and
        agent0_pos[1] >= agent1_pos[1] - comm_range and agent0_pos[1] <= agent1_pos[1] + comm_range):
        return 1
    else:
        return 0

def get_message_indices(agent0_pos_list, agent1_pos_list):
    '''
    get message indices where agents are nearby
    '''
    T = len(agent0_pos_list)
    time_indices = []
    for t in range(T):
        if check_comm_range(agent0_pos_list[t], agent1_pos_list[t]):
            time_indices.append(t)
    return time_indices

        
# Extract and prepare data for t-SNE
def prepare_tsne_data(log_data):
    max_message_length = 12
    count_uselsss =  0
    message_emb_dict = {"agent0": [], "agent1":[]}
    message_token_dict = {"agent0": [], "agent1":[]}
    spawn_time = {"agent0": [], "agent1":[]}
    item_locs = {"agent0": [], "agent1":[]}
    item_times = {"agent0": [], "agent1":[]}
    for episode, data in log_data.items():
        log_s_message_embs = data["log_s_message_embs"]
        log_s_message_tokens = data["log_s_messages"]
        log_r_message_tokens = data["log_r_messages"]
        log_rewards = data["log_rewards"]
        success = np.max(log_rewards) >= 1
        log_food_dict = data["log_food_dict"]
        food_loc = log_food_dict["location"] # list
        food_time = log_food_dict["spawn_time"] # list
        message_indices = get_message_indices(data["log_locs"][:, 0], data["log_locs"][:, 1])
        # print(f"success={success} with inds {message_indices}")
        # print(log_r_message_tokens)
        for agent_id in range(2):
            agent_loc = data["log_locs"][:, agent_id] #(num_steps, 2)
            food_id = get_food_id(agent_loc, food_loc)
            # print(f"food id {food_id}")
            # Get sent messages and target food score
            
            if log_s_message_embs.shape[2] == 2: # num_agents
                message_embs = log_s_message_embs[:, :, agent_id]
            else:
                message_embs = log_s_message_embs[:, agent_id, :]
            message_dim = message_embs.shape[1]
            padded_message_embs = np.zeros((max_message_length, message_dim))
            padded_message_tokens = np.zeros((max_message_length, 1))
            message_tokens = log_s_message_tokens[:, agent_id]
            curr_message_length = len(message_indices)
            if curr_message_length <= max_message_length:
                padded_message_embs[:curr_message_length] = message_embs[message_indices]
                padded_message_tokens[:curr_message_length] = np.expand_dims(message_tokens[message_indices], axis=1)
                padded_message_embs = padded_message_embs.flatten()
                padded_message_tokens = padded_message_tokens.flatten()
            
                # print(food_time[food_id])
                message_emb_dict[f"agent{agent_id}"].append(padded_message_embs.flatten())  # Collect all time steps for the agent
                message_token_dict[f"agent{agent_id}"].append(padded_message_tokens.flatten()) 
                item_times[f"agent{agent_id}"].append(food_time[food_id])
                item_locs[f"agent{agent_id}"].append(list(food_loc[food_id]))
            else:
                count_uselsss += 1
        if episode == 500:
            break
            

    print(f"Too Long Message: {count_uselsss}")
    return message_emb_dict, message_token_dict, item_times, item_locs

# Plot t-SNE
def plot_tsne(tsne_results, scores, plot_agent="agent0"):
    scores = scores[plot_agent]
    print(tsne_data.shape)
    

    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=scores, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="spawned time")
    plt.title(f"t-SNE of message embeddings sent by {plot_agent}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    
    plt.savefig("plot_by_time")
    plt.show()


# Plot t-SNE
def plot_tsne_loc(tsne_results, item_locs, plot_agent="agent0"):
    item_locs = np.int32(np.array(item_locs[plot_agent]))

    
    
    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=item_locs[:, 0], palette='tab10', s=50, alpha=0.7)
    plt.legend(title="item's position y")
    plt.title(f"t-SNE of message embeddings sent by {plot_agent}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig("plot_by_y")
    # plt.show()

    
    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))

    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=item_locs[:, 1], palette='tab10', s=50, alpha=0.7)
    plt.legend(title="item's position x")
    plt.title(f"t-SNE of message embeddings sent by {plot_agent}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig("plot_by_x")
    plt.show()


# Plot t-SNE
def plot_tsne_loc_see_target(tsne_results, item_locs, see_target, plot_agent="agent0"):
    item_locs = np.int32(np.array(item_locs[plot_agent]))
    see_target = np.int32(np.array(see_target[plot_agent]))  # Ensure see_target is an array
    
    # Encode each data point as a binary string "VHST" (V=vertical, H=horizontal, S=see_target)
    group_v_labels = [f"v{v}t{s}" for v, s in zip(item_locs[:, 0], see_target)]
    group_h_labels = [f"h{h}t{s}" for h, s in zip(item_locs[:, 1], see_target)]
    
    
    
    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=group_v_labels, palette='tab10', s=50, alpha=0.7)
    plt.legend(title="item's vertical position")
    plt.title(f"t-SNE of message embeddings sent by {plot_agent}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    # plt.show()

    
    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))

    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=group_h_labels, palette='tab10', s=50, alpha=0.7)
    plt.legend(title="item's horizontal position")
    plt.title(f"t-SNE of message embeddings sent by {plot_agent}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    model_name = "dec_ppo_invisible"
    combination_name = "grid5_img3_ni2_nw4_ms20_332800000"
    seed = 1
    mode = "train"
    log_file_path =  f"../../logs/pickup_temporal/{model_name}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
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
