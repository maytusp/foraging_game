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

def compute_avg_sr(sr_mat):
    """
    Compute the average of the lower-triangle elements of an MxM matrix,
    excluding the diagonal.
    
    Args:
        sr_mat (np.ndarray): An MxM matrix.

    Returns:
        float: The average of the lower-triangle elements.
    """
    M = sr_mat.shape[0]
    lower_triangle = np.tril(sr_mat, k=-1)  # Get lower-triangle excluding diagonal
    return lower_triangle.sum() / (M * (M - 1) / 2)  # Divide by the number of elements



saved_fig_dir = f"figs/population/plots"
os.makedirs(saved_fig_dir, exist_ok=True)
num_networks_list = [3,6,9,12,15]
model_name_list = []
checkpoints_dict = {"pop_ppo_3net_invisible": {'seed1': 204800000, 'seed2': 204800000, 'seed3':204800000},
                    "pop_ppo_6net_invisible": {'seed1': 486400000, 'seed2': 486400000, 'seed3':486400000},
                    "pop_ppo_9net_invisible": {'seed1': 691200000, 'seed2': 691200000, 'seed3':691200000},
                    "pop_ppo_12net_invisible": {'seed1': 896000000, 'seed2': 819200000, 'seed3':819200000},
                    "pop_ppo_15net_invisible": {'seed1': 947200000, 'seed2': 819200000, 'seed3':819200000},
                    }

load_score = {}
mean_ls = []
std_ls = []
mean_topsim = []
std_topsim = []
mean_ic = []
std_ic = []
mean_sr = []
std_sr = []

for num_networks in num_networks_list:
    ls = []
    topsim = []
    ic = []
    sr = []

    for seed in range(1,4):
        if seed == 3 and num_networks == 9:
            continue
        model_name = f"pop_ppo_{num_networks}net_invisible"
        ckpt_name = checkpoints_dict[model_name][f"seed{seed}"]
        combination_name = f"grid5_img3_ni2_nw4_ms10_{ckpt_name}"
        
        scores_path = f"../../logs/pickup_high_v1/exp2/{model_name}/{combination_name}_seed{seed}/sim_scores.npz"
        
        scores = np.load(scores_path)
        avg_sr = compute_avg_sr(scores['sr_mat'])
        ic.append(scores['ic'])
        ls.append(scores['avg_sim'])
        topsim.append(scores['avg_topsim'])
        sr.append(avg_sr)

    mean_ls.append(np.mean(ls))
    std_ls.append(np.std(ls))

    mean_topsim.append(np.mean(topsim))
    std_topsim.append(np.std(topsim))

    mean_ic.append(np.mean(ic))
    std_ic.append(np.std(ic))

    mean_sr.append(np.mean(sr))
    std_sr.append(np.std(sr))
print(mean_sr, std_sr)
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Plot Language Similarity
sns.lineplot(x=num_networks_list, y=mean_ls, marker="o", ax=axes[0], label='Language Similarity')
axes[0].fill_between(num_networks_list, np.array(mean_ls) - np.array(std_ls), np.array(mean_ls) + np.array(std_ls), alpha=0.2)
axes[0].set_xlabel('Population Size')
axes[0].set_ylabel('Language Similarity')
axes[0].set_title('Language Similarity vs Population Size')
axes[0].set_xticks(num_networks_list)
axes[0].legend()

# Plot Topographic Similarity
sns.lineplot(x=num_networks_list, y=mean_topsim, marker="s", ax=axes[1], label='Topographic Similarity', color='r')
axes[1].fill_between(num_networks_list, np.array(mean_topsim) - np.array(std_topsim), np.array(mean_topsim) + np.array(std_topsim), alpha=0.2, color='r')
axes[1].set_xlabel('Population Size')
axes[1].set_ylabel('Topographic Similarity')
axes[1].set_title('Topographic Similarity vs Population Size')
axes[1].set_xticks(num_networks_list)
axes[1].legend()

# Plot Interchangeability
sns.lineplot(x=num_networks_list, y=mean_ic, marker="^", ax=axes[2], label='Interchangeability', color='g')
axes[2].fill_between(num_networks_list, np.array(mean_ic) - np.array(std_ic), np.array(mean_ic) + np.array(std_ic), alpha=0.2, color='g')
axes[2].set_xlabel('Population Size')
axes[2].set_ylabel('Interchangeability')
axes[2].set_title('Interchangeability vs Population Size')
axes[2].set_xticks(num_networks_list)
axes[2].legend()

# Plot Success Rate
sns.lineplot(x=num_networks_list, y=mean_sr, marker="^", ax=axes[3], label='Success Rate', color='g')
axes[3].fill_between(num_networks_list, np.array(mean_sr) - np.array(std_sr), np.array(mean_sr) + np.array(std_sr), alpha=0.2, color='g')
axes[3].set_xlabel('Population Size')
axes[3].set_ylabel('Success Rate')
axes[3].set_title('Success Rate vs Population Size')
axes[3].set_xticks(num_networks_list)
axes[3].legend()

plt.tight_layout()
plt.savefig(os.path.join(saved_fig_dir, "plot.png"))
plt.show()
