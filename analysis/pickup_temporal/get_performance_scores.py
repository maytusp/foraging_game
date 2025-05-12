import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import os
import pandas as pd


def load_score(filename):
    scores = {}
    with open(filename, "r") as f:
        for line in f:
            x = line.strip().split(": ")
            if "Reward" in x[0]:
                continue
            scores[x[0].strip()] = float(x[1].strip())
    return scores

model_name_map = {
                  "pop_ppo_3net_invisible/grid5_img3_ni2_nw4_ms20_freeze_dur6_819200000": "default",
                }
mode = "train"
num_networks = 3
test_conditions = ["normal", "hard"] 
metrics = ["sr", "length"]
model_score_dict = {test_cond:{"sr":{}, "length":{}} for test_cond in test_conditions}
saved_fig_dir = f"plots/message_ablation/"
save_fig_path = os.path.join(saved_fig_dir, f"message_ablation")
os.makedirs(saved_fig_dir, exist_ok=True)
for test_condition in test_conditions:
    for setting_name in model_name_map:
        model_name, combination_name  = setting_name.split("/")
        print(f"{model_name}/{combination_name}")
        sr_list = [] # sucess rates of different seeds and pairs
        l_list = [] # episode lengths of different seeds and pairs
        for seed in [1,2,3]:
            network_pairs = [f"{i}-{j}" for i in range(num_networks) for j in range(i+1)]
            # network_pairs = ["0-1"]
            score_dict = {}
            sr_mat = np.zeros((num_networks, num_networks))
            
            for pair in network_pairs:
                row, col = pair.split("-")
                row, col = int(row), int(col)
                print(f"loading network pair {pair}")
                
                score_dict[pair] = load_score(f"../../logs/ablate_message_during_train/pickup_high_v1/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/{test_condition}/score.txt")
                sr_list.append(score_dict[pair]["Success Rate"])
                l_list.append(score_dict[pair]["Average Success Length"])
        model_score_dict[test_condition]["sr"][setting_name] = sr_list
        model_score_dict[test_condition]["length"][setting_name] = l_list
