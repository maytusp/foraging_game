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
            line = line.strip()
            if not line:
                continue

            if ":" in line:  # colon-separated (e.g., Length)
                key, val = line.split(":", 1)
                scores[key.strip()] = float(val.strip())
            else:  # space-separated (e.g., Reward Agent0 2.11)
                parts = line.split()
                # Last part is the number, everything before is the key
                key = " ".join(parts[:-1])
                val = parts[-1]
                scores[key] = float(val)
    return scores


model_name_map = {
                "pop_ppo_3net_invisible/grid6_img5_ni2_nw4_ms20_1M",
                }
mode = "test"
num_networks = 3
test_conditions =  ["normal","noise","zero"] 
metrics = ["Average Reward Agent0", "Average Reward Agent1", "length"]
model_score_dict = {test_cond:{"Average Reward Agent0":{}, "Average Reward Agent1":{}, "length":{}} for test_cond in test_conditions}

for test_condition in test_conditions:
    for setting_name in model_name_map:
        model_name, combination_name  = setting_name.split("/")
        print(f"{model_name}/{combination_name}")
        sr_list_0 = [] # sucess rates of different seeds and pairs
        sr_list_1 = [] # sucess rates of different seeds and pairs
        l_list = [] # episode lengths of different seeds and pairs
        for seed in [1]:
            network_pairs = [f"{i}-{j}" for i in range(num_networks) for j in range(i+1)]
            # network_pairs = ["0-1"]
            score_dict = {}
            sr_mat = np.zeros((num_networks, num_networks))
            
            for pair in network_pairs:
                row, col = pair.split("-")
                row, col = int(row), int(col)
                print(f"loading network pair {pair}")
                
                score_dict[pair] = load_score(f"../../logs/pickup_ind/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/{test_condition}/score.txt")
                # print(score_dict)
                sr_list_0.append(score_dict[pair]["Average Reward Agent0"])
                sr_list_1.append(score_dict[pair]["Average Reward Agent1"])
                l_list.append(score_dict[pair]["Average Length"])
        model_score_dict[test_condition]["Average Reward Agent0"][setting_name] = sr_list_0
        model_score_dict[test_condition]["Average Reward Agent1"][setting_name] = sr_list_1
        model_score_dict[test_condition]["length"][setting_name] = l_list


# PLOT HERE
# Prepare data for plotting
plot_data = {
    "condition": [], "Average Reward Agent0": [], "Average Reward Agent1":[], "length": []
}


for model_key in model_name_map:

    for test_condition in test_conditions:
        plot_data["condition"].append(test_condition)
        for metric in metrics:
            # Handle model_name part only (strip path after slash if needed)
            base_model_key = model_key

            score_list = model_score_dict[test_condition][metric].get(base_model_key, None)
            if score_list is not None:
                mean = np.mean(score_list)
                std = np.std(score_list)
            else:
                mean = np.nan
                std = np.nan

            # report_score = f"{mean} ± {std}"
            # print(report_score)
            plot_data[metric].append(f"{mean:.3f} ± {std:.3f}")

print(plot_data)
df = pd.DataFrame(plot_data)
print(df)