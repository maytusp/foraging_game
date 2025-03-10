from language_analysis import Disent, TopographicSimilarity
import numpy as np
import pickle
import os
import torch
image_size = 5
visible_range = image_size // 2
target_item_only = True
use_distractor = True

# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

def within_receptive_field(receptive_field_size, agent_location, item_location):
    visible_range = receptive_field_size // 2 # visible_range(3x3) = 1, (5x5) = 2, (7x7) =3
    if (item_location[0] >= agent_location[0] - visible_range and item_location[0] <= agent_location[0] + visible_range and
        item_location[1] >= agent_location[1] - visible_range and item_location[1] <= agent_location[1] + visible_range):
        return True
    else:
        return False

def compute_first_seen_time_indices(log_locs, log_foods, receptive_field_size, N_i):
    first_seen_time_indices = np.zeros((2, N_i))
    for agent_id in range(2):
        for item_id in range(N_i):
            item_loc = log_foods["position"][item_id]
            for t in range(log_locs.shape[0]):
                agent_loc = log_locs[t, agent_id, :]
                seen = within_receptive_field(receptive_field_size, agent_loc, item_loc)
                if seen:
                    first_seen_time_indices[agent_id, item_id] = t
                    break
                if t == log_locs.shape[0]-1: # Item is not seen
                    first_seen_time_indices[agent_id, item_id] = -1
    return first_seen_time_indices



def extract_message(log_data, N_att=2, N_i=2, window_size=8, lag_time=0):
    attributes = {0:[], 1:[]}
    messages = {0:[], 1:[]}
    swap_target = {0:1, 1:0}
    
    neg_episode = 0
    for episode, data in log_data.items():
        
        # Get sent messages and target food score
        log_s_message = data["log_s_messages"]
        log_masks = data["log_masks"]
        log_attributes = data["log_attributes"]
        log_goal = np.array(data["log_goal"])
        log_foods = data["log_foods"]
        log_locs = data["log_locs"] # (max_steps, num_agents, 2)
        log_target_id = data["log_target_food_id"]
        log_rewards = data["log_rewards"][:, 0]
        print("log_s_message agent0:", log_s_message[:, 0])
        print("log_s_message agent1:", log_s_message[:, 1])
        # print("target dist: ", np.sum((log_attributes[log_target_id]-log_goal)**2))
        # print("distactor dist: ", np.sum((log_attributes[swap_target[log_target_id]]-log_goal)**2))
        
        if use_distractor:
            log_target_id = swap_target[log_target_id]
        max_timesteps = log_locs.shape[0]
        num_agents = 2
        first_seen_time_indices = compute_first_seen_time_indices(log_locs, log_foods, receptive_field_size=5, N_i=N_i)
        
        # Check if any row contains -1
        # has_neg_one = np.any(first_seen_time_indices == -1, axis=1)
        has_neg_one = np.any(first_seen_time_indices == -1)
        check_window_size = np.any(19-first_seen_time_indices < window_size+lag_time)
        if not(has_neg_one or check_window_size):
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

                    mask_att = item_att * agent_mask
                    mask_att_diff = diff_att * agent_mask
                    mse = np.expand_dims(np.mean(mask_att_diff**2), axis=0)

                    start_idx += lag_time

                    extract_message = log_s_message[start_idx:start_idx+window_size, agent_id]

                    extract_attribute = np.concatenate((mse, agent_pos))
                    # extract_attribute = mask_att
                    messages[agent_id].append(extract_message)  # Collect all time steps for the agent
                    attributes[agent_id].append(extract_attribute)

        else:
            neg_episode+=1

    print(f"Total unused episodes: {neg_episode}")
    return attributes, messages

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    log_file_path = "../logs/goal_condition_pickup/dec_ppo_invisible/grid5_img5_ni2_natt2_nval10_nw16_716800000/seed1/mode_train/normal/trajectory.pkl"
    num_episodes = 2000
    if os.path.exists(log_file_path):
        # Load log data
        log_data = load_trajectory(log_file_path)
        attributes_dict, messages_dict = extract_message(log_data)
        for agent_id in range(2):
            print(f"agent{agent_id}")
            attributes = np.array(attributes_dict[agent_id])
            messages = np.array(messages_dict[agent_id])
            print(messages.shape)
            attributes, messages = torch.Tensor(attributes[:num_episodes, :]), torch.Tensor(messages[:num_episodes])

            topsim = TopographicSimilarity.compute_topsim(attributes, messages)
            posdis = Disent.posdis(attributes, messages)
            bosdis = Disent.bosdis(attributes, messages, vocab_size=16)
            
            print(f"topsim {topsim}")
            print(f"posdis: {posdis}")
            print(f"bosdis: {bosdis}")
        
    else:
        print(f"Log file not found: {log_file_path}")
