# This code is for plotting LS, topsim, IC over population size 3,6,9,12,15
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import editdistance
from language_analysis import Disent, TopographicSimilarity
import os

def compute_avg_sr_ls(sr_mat, ls_mat):
    M = sr_mat.shape[0]
    sr_list = []
    ls_list = []
    dist_list = []
    
    max_distance = M // 2  # Maximum meaningful distance in the ring
    distance_sums = {d: [] for d in range(1, max_distance+1)}
    ls_sums = {d: [] for d in range(1, max_distance+1)}

    for row in range(1, M):  # Start from 1 to avoid diagonal
        for col in range(row):  # Only consider the lower triangle (col < row)
            distance = min(abs(row - col), M - abs(row - col))  # Ring topology distance

            distance_sums[distance].append(sr_mat[row, col])  # Only lower triangle
            ls_sums[distance].append(ls_mat[row, col])

    for d in range(1, max_distance+1):
        if distance_sums[d]:  # Avoid division by zero
            avg_sr = np.mean(distance_sums[d])
            avg_ls = np.mean(ls_sums[d])
        else:
            avg_sr = 0  # If no pairs at this distance (unlikely in a ring)
            avg_ls = 0

        sr_list.append(avg_sr)
        ls_list.append(avg_ls)
        dist_list.append(d)
    return sr_list, ls_list, dist_list



saved_fig_dir = f"plots/population/ring/15net/"
os.makedirs(saved_fig_dir, exist_ok=True)
model_name_list = []
checkpoints_dict = {
    "ring_sp_ppo_15net_invisible": {'seed1': 870400000, 'seed2': 870400000, 'seed3':870400000},
    "ring_ppo_15net_invisible": {'seed1': 870400000, 'seed2': 870400000, 'seed3':870400000},
                    }

total_sr = {}
total_ls = {}

for model_name in checkpoints_dict.keys():
    print(f"model {model_name}")
    total_sr[model_name] = []
    total_ls[model_name] = []
    for seed in range(1,2):
        ckpt_name = checkpoints_dict[model_name][f"seed{seed}"]
        combination_name = f"grid5_img3_ni2_nw4_ms10_{ckpt_name}"
        
        scores_path = f"../../logs/pickup_high_v1/exp2/{model_name}/{combination_name}_seed{seed}/sim_scores.npz"
        
        scores = np.load(scores_path)
        sr_list, ls_list, distance_list = compute_avg_sr_ls(scores['sr_mat'], scores['similarity_mat'])
        total_sr[model_name].append(sr_list)
        total_ls[model_name].append(ls_list)
    

    total_sr[model_name] = np.mean(np.array(total_sr[model_name]), axis=0)
    total_ls[model_name] = np.mean(np.array(total_ls[model_name]), axis=0)


import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn style and global font settings
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 20, 
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 14})

# Plot 1: Success Rate vs Distance
plt.figure(figsize=(6, 5))
sns.lineplot(x=distance_list, y=total_sr['ring_sp_ppo_15net_invisible'], marker="o", label='self-play')
sns.lineplot(x=distance_list, y=total_sr['ring_ppo_15net_invisible'], marker="o", label='w/o self-play')
plt.xlabel('Distance')
plt.ylabel('Success Rate')
plt.xticks(distance_list)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(saved_fig_dir, "sr_vs_distance.png"))
plt.close()

# Plot 2: Language Similarity vs Distance
plt.figure(figsize=(6, 5))
sns.lineplot(x=distance_list, y=total_ls['ring_sp_ppo_15net_invisible'], marker="o", label='self-play')
sns.lineplot(x=distance_list, y=total_ls['ring_ppo_15net_invisible'], marker="o", label='w/o self-play')
plt.xlabel('Distance')
plt.ylabel('Language Similarity')
plt.xticks(distance_list)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(saved_fig_dir, "ls_vs_distance.png"))
plt.close()

