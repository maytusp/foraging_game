# This file is for storing datasets for computing repr. comp. score from ICML'25 paper.

import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import editdistance
from language_analysis import Disent, TopographicSimilarity
import os
import torch  # <-- NEW: for saving .pt files

# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

# Extract and prepare data for t-SNE
def extract_data(log_data, max_length=3):
    message_data = {"agent0": [], "agent1": []}
    attribute_data = []
    scores = {"agent0": [], "agent1": []}
    item_locs = {"agent0": [], "agent1": []}
    for id, (episode, data) in enumerate(log_data.items()):
        log_s_messages = data["log_s_messages"]
        who_see_target = data["who_see_target"]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"]  # (2,)
        distractor_score = data["log_distractor_food_dict"]["score"][0]
        distractor_loc = data["log_distractor_food_dict"]["location"][0]  # (2,)
        if log_s_messages[max_length-1,0] != -1:
            for agent_id in range(2):
                messages = log_s_messages[:max_length, agent_id].flatten()
                message_data[f"agent{agent_id}"].append(
                    messages
                )  # Collect all time steps for the agent
            extract_attribute = [
                target_score,
                target_loc[0],
                target_loc[1],
                distractor_score,
                distractor_loc[0],
                distractor_loc[1],]

            attribute_data.append(extract_attribute)
    return message_data, attribute_data



# ---------- NEW: save datasets for prequential training ----------
def save_prequential_datasets(
    message_data,
    attribute_data,
    num_networks,
    seed,
    model_name,
    combination_name,
    root_dir="prequential_datasets",
):
    """
    For each agent_id, save a dataset under:
      {root_dir}/{model_name}/{combination_name}/seed{seed}/agent{agent_id}/
        - z.pt : normalized attributes tensor [N, 6], float32 in [0, 1]
        - w.pt : messages tensor           [N, msg_dim], int64

    Normalization (per dimension):
      z[:, 0], z[:, 3] -> divided by 255.0
      z[:, 1], z[:, 2], z[:, 4], z[:, 5] -> divided by 4.0
    """
    receiver = 0

    for agent_id in range(num_networks):
        pair_key = f"{agent_id}-{receiver}"
        if pair_key not in message_data:
            print(f"[WARN] pair {pair_key} not found in message_data, skipping agent {agent_id}")
            continue

        # messages: list of 1D arrays
        msgs = np.array(message_data[pair_key]["agent0"])
        attrs = np.array(attribute_data[pair_key])
        print("attr", attrs.shape)

        # ---- build tensors ----
        # z as float for continuous Gaussian model
        z = torch.as_tensor(attrs, dtype=torch.float32)   # [N, 6]
        w = torch.as_tensor(msgs, dtype=torch.long)       # [N, msg_dim]

        # sanity check on z shape
        # if z.ndim != 2 or z.shape[1] < 6:
        #     raise ValueError(f"Expected z to have shape [N, 6], got {z.shape}")
        print("message shape", msgs.shape)
        # ---- normalization ----
        z_norm = z.clone()
        # dims 0 and 3: range {5, 10, ..., 255}
        z_norm[:, 0] = z_norm[:, 0] / 255.0 # max score is 255
        z_norm[:, 3] = z_norm[:, 3] / 255.0
        # dims 1, 2, 4, 5: range {0, 1, 2, 3, 4}
        z_norm[:, 1] = z_norm[:, 1] / 4.0 # max position is 4
        z_norm[:, 2] = z_norm[:, 2] / 4.0
        z_norm[:, 4] = z_norm[:, 4] / 4.0
        z_norm[:, 5] = z_norm[:, 5] / 4.0

        agent_dir = os.path.join(
            root_dir,
            model_name,
            combination_name,
            f"seed{seed}",
            f"agent{agent_id}",
        )
        os.makedirs(agent_dir, exist_ok=True)

        # save normalized z
        torch.save(z_norm, os.path.join(agent_dir, "z.pt"))
        torch.save(w, os.path.join(agent_dir, "w.pt"))

        print(
            f"Saved prequential dataset for {model_name}, {combination_name}, "
            f"seed{seed}, agent{agent_id} at {agent_dir}"
        )


# -----------------------------------------------------------------


if __name__ == "__main__":
    checkpoints_dict = {
        "dec_ppo_invisible": {
            "seed1": 204800000,
            "seed2": 204800000,
            "seed3": 204800000,
        },
        "pop_ppo_3net_invisible": {
            "seed1": 204800000,
            "seed2": 204800000,
            "seed3": 204800000,
        },
        # "pop_ppo_6net_invisible": {
        #     "seed1": 460800000,
        #     "seed2": 460800000,
        #     "seed3": 460800000,
        # },
        # "pop_ppo_9net_invisible": {
        #     "seed1": 512000000,
        #     "seed2": 512000000,
        #     "seed3": 512000000,
        # },
        # "pop_ppo_12net_invisible": {
        #     "seed1": 768000000,
        #     "seed2": 768000000,
        #     "seed3": 768000000,
        # },
        # "pop_ppo_15net_invisible": {
        #     "seed1": 819200000,
        #     "seed2": 819200000,
        #     "seed3": 819200000,
        # },
        # "dec_sp_ppo_invisible": {
        #     "seed1": 204800000,
        #     "seed2": 204800000,
        #     "seed3": 204800000,
        # },
        # "pop_sp_ppo_3net_invisible": {
        #     "seed1": 204800000,
        #     "seed2": 204800000,
        #     "seed3": 204800000,
        # },
        # "pop_sp_ppo_6net_invisible": {
        #     "seed1": 460800000,
        #     "seed2": 460800000,
        #     "seed3": 460800000,
        # },
        # "pop_sp_ppo_9net_invisible": {
        #     "seed1": 512000000,
        #     "seed2": 512000000,
        #     "seed3": 512000000,
        # },
        # "pop_sp_ppo_12net_invisible": {
        #     "seed1": 768000000,
        #     "seed2": 768000000,
        #     "seed3": 768000000,
        # },
        # "pop_sp_ppo_15net_invisible": {
        #     "seed1": 819200000,
        #     "seed2": 819200000,
        #     "seed3": 819200000,
        # },
    }
    model2numnet = {
        "dec_ppo_invisible": 2,
        "pop_ppo_3net_invisible": 3,
        "pop_ppo_6net_invisible": 6,
        "pop_ppo_9net_invisible": 9,
        "pop_ppo_12net_invisible": 12,
        "pop_ppo_15net_invisible": 15,
        "dec_sp_ppo_invisible": 2,
        "pop_sp_ppo_3net_invisible": 3,
        "pop_sp_ppo_6net_invisible": 6,
        "pop_sp_ppo_9net_invisible": 9,
        "pop_sp_ppo_12net_invisible": 12,
        "pop_sp_ppo_15net_invisible": 15,
    }

    compute_topsim = True
    cbar = False
    max_length = 4 # max message length
    for model_name in checkpoints_dict.keys():
        num_networks = model2numnet[model_name]
        avg_similarity_mat = np.zeros((num_networks, num_networks))
        avg_sr_mat = np.zeros((num_networks, num_networks))
        for seed in range(1, 4):

            ckpt_name = checkpoints_dict[model_name][f"seed{seed}"]
            combination_name = f"grid5_img3_ni2_nw4_ms10_{ckpt_name}"

            mode = "test"
            network_pairs = [
                f"{i}-{j}" for i in range(num_networks) for j in range(i + 1)
            ]
            log_file_path = {}
            sr_dict = {}
            sr_mat = np.zeros((num_networks, num_networks))
            message_data = {}
            attribute_data = {}

            for pair in network_pairs:
                row, col = pair.split("-")
                row, col = int(row), int(col)
                print(f"loading network pair {pair}")
                log_file_path[pair] = (
                    f"../../logs/vary_n_pop/{model_name}/{pair}/{combination_name}"
                    f"/seed{seed}/mode_{mode}/normal/trajectory.pkl"
                )

                # Load log data
                log_data = load_trajectory(log_file_path[pair])

                # Prepare data for t-SNE / topsim / datasets
                message_data[pair], attribute_data[pair] = extract_data(log_data, max_length=max_length)



            # ---------- NEW: save datasets per agent & seed ----------
            save_prequential_datasets(
                message_data=message_data,
                attribute_data=attribute_data,
                num_networks=num_networks,
                seed=seed,
                model_name=model_name,
                combination_name=combination_name,
                root_dir="../../logs/repcom_dataset",  # change if you like
            )