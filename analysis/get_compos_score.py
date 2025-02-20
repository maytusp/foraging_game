from language_analysis import Disent, TopographicSimilarity
import numpy as np
import pickle
import os
import torch
image_size = 5
visible_range = image_size // 2
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

def extract_high_score_message(log_data, N_att=2, N_i=2):
    attributes = {0:[], 1:[]}
    messages = {0:[], 1:[]}

    for episode, data in log_data.items():
        
        # Get sent messages and target food score
        log_s_message = data["log_s_messages"]
        log_masks = data["log_masks"]
        log_attributes = data["log_attributes"]
        for agent_id in range(2):
            seen_attribute = np.zeros((N_i, N_att))
            for item_id in range(N_i):
                seen_attribute[item_id, :] = log_attributes[item_id]*log_masks[agent_id]
            # target_loc = data["log_target_food_dict"]["location"] # (2,)
            # agent_locs = data["log_locs"][:, plot_agent] #(num_steps, 2)
            # num_steps = agent_locs.shape[0]
            # print(f"agent {agent_id} sees {seen_attribute.flatten()}")
            messages[agent_id].append(log_s_message[:, agent_id])  # Collect all time steps for the agent
            attributes[agent_id].append(seen_attribute.flatten())

    return attributes, messages

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    log_file_path = "../logs/goal_condition_pickup/dec_ppo/grid5_img5_ni2_natt2_nval10_nw16/seed42/mode_train/trajectory.pkl"
    num_episodes = 2000
    if os.path.exists(log_file_path):
        # Load log data
        log_data = load_trajectory(log_file_path)
        attributes_dict, messages_dict = extract_high_score_message(log_data)
        for agent_id in range(2):
            print(f"agent{agent_id}")
            attributes = np.array(attributes_dict[agent_id])
            messages = np.array(messages_dict[agent_id])
            attributes, messages = torch.Tensor(attributes[:num_episodes, :]), torch.Tensor(messages[:num_episodes, 2:4])


            topsim = TopographicSimilarity.compute_topsim(attributes, messages)
            posdis = Disent.posdis(attributes, messages)
            bosdis = Disent.bosdis(attributes, messages, vocab_size=16)
            
            print(f"topsim {topsim}")
            print(f"posdis: {posdis}")
            print(f"bosdis: {bosdis}")
        
    else:
        print(f"Log file not found: {log_file_path}")
