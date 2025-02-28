import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from linear_decode import extract_label
from get_compos_score import load_trajectory, within_receptive_field, compute_first_seen_time_indices

def extract_message_emb(log_data, N_att=2, N_i=2, window_size=8, lag_time=0, use_all_message=True):
    attributes = {0:[], 1:[]}
    messages = {0:[], 1:[]}
    swap_target = {0:1, 1:0}
    
    neg_episode = 0
    for episode, data in log_data.items():
        # Get sent messages and target food score
        log_s_message_embs = data["log_s_message_embs"] # (timesteps, emb_size, num_agents)
        log_masks = data["log_masks"]
        log_attributes = data["log_attributes"]
        log_goal = np.array(data["log_goal"])
        log_foods = data["log_foods"]
        log_locs = data["log_locs"] # (max_steps, num_agents, 2)
        log_target_id = data["log_target_food_id"]
        log_rewards = data["log_rewards"][:, 0]

        max_timesteps = log_locs.shape[0]
        num_agents = 2
        first_seen_time_indices = compute_first_seen_time_indices(log_locs, log_foods, receptive_field_size=5, N_i=N_i)
        
        # Check if any row contains -1
        # has_neg_one = np.any(first_seen_time_indices == -1, axis=1)
        has_neg_one = np.any(first_seen_time_indices == -1)
        check_window_size = np.any(19-first_seen_time_indices < window_size+lag_time)
        if not has_neg_one: # not(has_neg_one or check_window_size):
            for agent_id in range(num_agents):
                start_idx_list = []
                for item_id in range(N_i):
                    
                    start_idx = int(first_seen_time_indices[agent_id, item_id])
                    start_idx_list.append(start_idx)

                    agent_pos = log_locs[start_idx, agent_id, :] # x_a, y_a
                    agent_goal = np.array(log_goal)  # g1,g2,...
                    agent_mask = log_masks[agent_id] # mask

                    item_pos = log_foods["position"][item_id] # x_i, y_i
                    item_att = np.array(log_attributes[item_id]) # a1,a2,...
                    diff_att = agent_goal - item_att # g1-a1, g2-a2,...
                    mse = np.expand_dims(np.mean(diff_att**2), axis=0)
                    mask_att = item_att * agent_mask
                    mask_att_diff = diff_att * agent_mask
                    mask_mse = np.expand_dims(np.mean(mask_att_diff**2), axis=0)

                    start_idx += lag_time
                    if use_all_message:
                        start_idx = 0
                        window_size=max_timesteps

                    extract_message = log_s_message_embs[start_idx:start_idx+window_size, :, agent_id].flatten()
                    extract_attribute = {
                                        "agent_pos":agent_pos,  # agent position
                                        "agent_goal":agent_goal, # agent goal
                                        "agent_mask":agent_mask, # agent mask e.g., [1 0 0 1] 
                                        "item_pos":item_pos, # item position
                                        "att": item_att, # item's attribute
                                        "mask_att":mask_att, # item's masked attribute
                                        "mask_att_diff":mask_att_diff, # item's masked attribute difference
                                        "mse":mse,
                                        "mask_mse":mask_mse,
                                        }
                    
                    messages[agent_id].append(extract_message)  # Collect all time steps for the agent
                    attributes[agent_id].append(extract_attribute)
        else:
            neg_episode+=1

    print(f"Total unused episodes: {neg_episode}")
    print(f"messages dim {np.array(messages[0]).shape}")
    return attributes, messages

# Plot t-SNE
def plot_tsne(message_data, labels, label_name):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(message_data)

    # Scatter plot with grouping by score
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label=label_name)
    plt.title("t-SNE of Message Embedding")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    log_file_path = "../logs/pickup_high_easy/ppo_ps_comm_550M/normal/trajectory.pkl"

    if os.path.exists(log_file_path):
        seen_log_file_path = "../logs/goal_condition_pickup/dec_ppo_invisible_possig/grid5_img5_ni2_natt2_nval10_nw16_1B/seed1/mode_train/normal/trajectory.pkl"
        unseen_log_file_path = "../logs/goal_condition_pickup/dec_ppo_invisible_possig/grid5_img5_ni2_natt2_nval10_nw16_1B/seed1/mode_test/normal/trajectory.pkl"
        agent_id = 0
        # label_encoder = sklearn.preprocessing.LabelEncoder()
        groundtruth_name = "agent_pos_x"

        # Load log data
        seen_data = load_trajectory(seen_log_file_path)
        seen_attributes, seen_messages = extract_message_emb(seen_data, window_size=2, lag_time=0, use_all_message=False) # attributes_dict[agent_id][episode_id] -> Dict
        seen_label_dict = extract_label(seen_attributes, agent_id=agent_id)
        
        seen_message_arr = np.array(seen_messages[agent_id])
        seen_label_arr = np.array(seen_label_dict[groundtruth_name])

        # Plot t-SNE
        plot_tsne(seen_message_arr, seen_label_arr, label_name=groundtruth_name)
    else:
        print(f"Log file not found: {log_file_path}")
