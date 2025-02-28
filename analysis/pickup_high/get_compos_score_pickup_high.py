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

# Extract and prepare data for t-SNE
def extract_high_score_message(log_data):
    tsne_data = []
    attributes = []
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
            target_loc = data["log_distractor_food_dict"]["location"] # (2,)
        
        
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


        print(f"start_idx{start_idx}")
        if start_idx is None or start_idx > 10:
            print("start_idx none")
            # If the agent never sees the target, skip this episode
            continue

        sent_message_embs = log_s_message_embs[start_idx:start_idx+5, plot_agent].flatten()
        tsne_data.append(sent_message_embs)  # Collect all time steps for the agent
        concat_att = np.concatenate((agent_locs, score))
        attributes.append()  # Same score for all time steps

    # Flatten and convert to NumPy
    tsne_data = np.vstack(tsne_data)
    scores = np.array(scores)
    print(f"tsne_data {tsne_data.shape}")
    return tsne_data, scores

if __name__ == "__main__":
    # Path to the trajectory .pkl file
    log_file_path = "../logs/goal_condition_pickup/dec_ppo/grid5_img5_ni2_natt2_nval10_nw16/seed0/mode_train/trajectory.pkl"
    num_episodes = 1000
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
