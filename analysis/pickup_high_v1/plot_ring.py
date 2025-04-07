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



saved_fig_dir = f"figs/population/plots"
os.makedirs(saved_fig_dir, exist_ok=True)
model_name_list = []
checkpoints_dict = {
                    "ring_sp_ppo_9net_invisible": {'seed1': 460800000, 'seed2': 460800000, 'seed3':460800000},
                    "ring_ppo_9net_invisible": {'seed1': 460800000, 'seed2': 460800000, 'seed3':460800000},
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


sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.lineplot(x=distance_list, y=total_sr['ring_sp_ppo_9net_invisible'], marker="o", ax=axes[0], label='self-play')
sns.lineplot(x=distance_list, y=total_sr['ring_ppo_9net_invisible'], marker="o", ax=axes[0], label='w/o self-play')
# axes[0].fill_between(distance_list, np.array(mean_ls) - np.array(std_ls), np.array(mean_ls) + np.array(std_ls), alpha=0.2)
axes[0].set_xlabel('Distance')
axes[0].set_ylabel('Success Rate')
axes[0].set_title('Success Rate vs Distance')
axes[0].set_xticks(distance_list)
axes[0].legend()


sns.lineplot(x=distance_list, y=total_ls['ring_sp_ppo_9net_invisible'], marker="o", ax=axes[1], label='self-play')
sns.lineplot(x=distance_list, y=total_ls['ring_ppo_9net_invisible'], marker="o", ax=axes[1], label='w/o self-play')
# axes[0].fill_between(distance_list, np.array(mean_ls) - np.array(std_ls), np.array(mean_ls) + np.array(std_ls), alpha=0.2)
axes[1].set_xlabel('Distance')
axes[1].set_ylabel('Language Similarity')
axes[1].set_title('Language Similarity vs Distance')
axes[1].set_xticks(distance_list)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(saved_fig_dir, "plot_ring.png"))
plt.show()
# print(mean_sr, std_sr)
