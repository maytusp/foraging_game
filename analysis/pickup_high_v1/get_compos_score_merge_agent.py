from language_analysis import Disent, TopographicSimilarity
import numpy as np
import pickle
import os
import torch
from transforms import *
image_size = 5
visible_range = image_size // 2
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

def extract_message_attribute(log_data):
    attributes = []
    messages = []

    for episode, data in log_data.items():
        # Get sent messages and target food score
        log_s_messages = data["log_s_messages"]
        who_see_target = data["who_see_target"]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"] # (2,)
        distractor_score = data["log_distractor_food_dict"]["score"][0]
        distractor_loc = data["log_distractor_food_dict"]["location"][0] # (2,)

        max_timesteps = log_s_messages.shape[0]
        num_agents = 2


        extract_message = log_s_messages[:5].flatten("F")


        extract_attribute = [target_score, target_loc[0], target_loc[1], 
                                distractor_score, distractor_loc[0], distractor_loc[1]]
        messages.append(extract_message)  # Collect all time steps for the agent
        attributes.append(extract_attribute)

    return attributes, messages

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    log_file_path = "../../logs/pickup_high_v1/dec_ppo_invisible0-1/grid5_img3_ni2_nw16_ms10_307200000/seed1/mode_train/normal/trajectory.pkl"
    num_episodes = 2000
    if os.path.exists(log_file_path):
        # Load log data
        log_data = load_trajectory(log_file_path)
        attributes_list, messages_list = extract_message_attribute(log_data)
        attributes = np.array(attributes_list)
        messages = np.array(messages_list)
        attributes, messages = torch.Tensor(attributes[:num_episodes, :]), torch.Tensor(messages[:num_episodes, :])

        topsim = TopographicSimilarity.compute_topsim(attributes, messages)
        posdis = Disent.posdis(attributes, messages)
        bosdis = Disent.bosdis(attributes, messages, vocab_size=16)
        
        print(f"topsim {topsim}")
        print(f"posdis: {posdis}")
        print(f"bosdis: {bosdis}")
        
    else:
        print(f"Log file not found: {log_file_path}")
