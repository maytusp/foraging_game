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



saved_fig_dir = f"plots/population/fc/"
os.makedirs(saved_fig_dir, exist_ok=True)
num_networks_list = [2, 3,6,9,12,15]
model_name_list = []
checkpoints_dict = {
    "dec_ppo_invisible" : {"seed1":307200000, "seed2":307200000, "seed3":307200000},
    "pop_ppo_3net_invisible": {'seed1': 204800000, 'seed2': 204800000, 'seed3':204800000},
    "pop_ppo_6net_invisible": {'seed1': 332800000, 'seed2': 332800000, 'seed3':332800000},
    "pop_ppo_9net_invisible": {'seed1': 486400000, 'seed2': 486400000, 'seed3':486400000},
    "pop_ppo_12net_invisible": {'seed1': 768000000, 'seed2': 768000000, 'seed3':768000000},
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
        if num_networks >= 3:
            model_name = f"pop_ppo_{num_networks}net_invisible"
        else:
            model_name = "dec_ppo_invisible"
        ckpt_name = checkpoints_dict[model_name][f"seed{seed}"]
        combination_name = f"grid5_img3_ni2_nw4_ms10_{ckpt_name}"
        
        scores_path = f"../../logs/vary_n_pop/exp2/{model_name}/{combination_name}_seed{seed}/sim_scores.npz"
        
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


# Set global style and font for publication-quality plots
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 20, 
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 14})

# Plot 1: Language Similarity
plt.figure(figsize=(6, 5))
sns.lineplot(x=num_networks_list, y=mean_ls, marker="o", linewidth=2.5, markersize=10)
plt.fill_between(num_networks_list, np.array(mean_ls) - np.array(std_ls), np.array(mean_ls) + np.array(std_ls), alpha=0.2)
plt.xlabel('Population Size')
plt.ylabel('Language Similarity')
plt.xticks(num_networks_list)
plt.tight_layout()
plt.savefig(os.path.join(saved_fig_dir, "ls_vs_pop.pdf"))
plt.close()

# Plot 2: Topographic Similarity
plt.figure(figsize=(6, 5))
sns.lineplot(x=num_networks_list, y=mean_topsim, marker="o", color='r', linewidth=2.5, markersize=10)
plt.fill_between(num_networks_list, np.array(mean_topsim) - np.array(std_topsim), np.array(mean_topsim) + np.array(std_topsim), alpha=0.2, color='r')
plt.xlabel('Population Size')
plt.ylabel('Topographic Similarity')
plt.xticks(num_networks_list)
plt.tight_layout()
plt.savefig(os.path.join(saved_fig_dir, "topsim_vs_pop.pdf"))
plt.close()

# Plot 3: Interchangeability
plt.figure(figsize=(6, 5))
sns.lineplot(x=num_networks_list, y=mean_ic, marker="o", color='g', linewidth=2.5, markersize=10)
plt.fill_between(num_networks_list, np.array(mean_ic) - np.array(std_ic), np.array(mean_ic) + np.array(std_ic), alpha=0.2, color='g')
plt.xlabel('Population Size')
plt.ylabel('Interchangeability')
plt.xticks(num_networks_list)
plt.tight_layout()
plt.savefig(os.path.join(saved_fig_dir, "ic_vs_pop.pdf"))
plt.close()


# Plot 4: SR
plt.figure(figsize=(6, 5))
sns.lineplot(x=num_networks_list, y=mean_sr, marker="o", color='m', linewidth=2.5, markersize=10)
plt.fill_between(num_networks_list, np.array(mean_sr) - np.array(std_sr), np.array(mean_sr) + np.array(std_sr), alpha=0.2, color='m')
plt.xlabel('Population Size')
plt.ylabel('Success Rate')
plt.xticks(num_networks_list)
plt.tight_layout()
plt.savefig(os.path.join(saved_fig_dir, "sr_vs_pop.pdf"))
plt.close()

# # Plot Success Rate
# sns.lineplot(x=num_networks_list, y=mean_sr, marker="o", ax=axes[3], color='m', linewidth=2.5, markersize=10)
# axes[3].fill_between(num_networks_list, np.array(mean_sr) - np.array(std_sr), np.array(mean_sr) + np.array(std_sr), alpha=0.2, color='m')
# axes[3].set_xlabel('Population Size', fontsize=18)
# axes[3].set_ylabel('Success Rate', fontsize=18)
# # axes[3].set_title('Success Rate vs Population Size')
# axes[3].set_xticks(num_networks_list)
# # axes[3].legend()
