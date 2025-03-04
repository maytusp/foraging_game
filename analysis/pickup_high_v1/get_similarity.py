import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import distance
import os
# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

# Extract and prepare data for t-SNE
def extract_data(log_data):
    message_data = {"agent0": [], "agent1":[]}
    scores = {"agent0": [], "agent1":[]}
    item_locs = {"agent0": [], "agent1":[]}
    switch_agent = {0:1, 1:0}
    for id, (episode, data) in enumerate(log_data.items()):

        log_s_messages = data["log_s_messages"]
        who_see_target = data["who_see_target"]
        another_agent = switch_agent[who_see_target]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"] # (2,)
        distractor_score = data["log_distractor_food_dict"]["score"][0]
        distractor_loc = data["log_distractor_food_dict"]["location"][0] # (2,)

        for agent_id in range(2):
            messages = log_s_messages[:, agent_id].flatten()
            message_data[f"agent{agent_id}"].append(messages)  # Collect all time steps for the agent


    return message_data



def get_similarity(message_data):
    sender_list = [0, 1]
    data = []
    agent_ids = []
    similarity_score = 0
    n_samples = 1000000
    extracted_message = []
    receiver = 1
    for sender in sender_list:
        extracted_message.append(np.array(message_data[f"{sender}-{receiver}"]["agent0"]))
        n_samples = min(extracted_message[sender].shape[0], n_samples)

    print(f"compute similarity on {n_samples} samples")

    for i in range(n_samples):
        m1 = extracted_message[sender_list[0]][i]
        m2 = extracted_message[sender_list[1]][i]
        similarity_score += (1 - distance.hamming(m1, m2))

    return similarity_score / n_samples
    

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    model_name = "dec_ppo_selfplay_invisible"
    combination_name = "grid5_img3_ni2_nw16_ms10_204800000"
    seed = 1
    mode = "test"
    network_pairs = ["0-0", "0-1", "1-1", "1-0"]
    log_file_path = {}

    message_data = {}
    for pair in network_pairs:
        print(f"loading network pair {pair}")
        log_file_path[pair] =  f"../../logs/pickup_high_v1/{model_name}{pair}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl"
    
        
        # Load log data
        log_data = load_trajectory(log_file_path[pair])

        # Prepare data for t-SNE
        message_data[pair] = extract_data(log_data)


    similarity_score = get_similarity(message_data)
    print(f"Similarity score: {similarity_score}")

