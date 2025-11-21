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
message_length = "mode"
model_name_list = []
checkpoints_dict = {
    "dec_ppo_invisible" : {"seed1":204800000, "seed2":204800000, "seed3":204800000},
    "pop_ppo_3net_invisible": {'seed1': 204800000, 'seed2': 204800000, 'seed3':204800000},
    "pop_ppo_6net_invisible": {'seed1': 460800000, 'seed2': 460800000, 'seed3':460800000},
    "pop_ppo_9net_invisible": {'seed1': 512000000, 'seed2': 512000000, 'seed3':512000000},
    "pop_ppo_12net_invisible": {'seed1': 768000000, 'seed2': 768000000, 'seed3':768000000},
    "pop_ppo_15net_invisible": {'seed1': 819200000, 'seed2': 819200000, 'seed3':819200000},}

checkpoints_dict_sp = {
    "dec_sp_ppo_invisible" : {"seed1":204800000, "seed2":204800000, "seed3":204800000},
    "pop_sp_ppo_3net_invisible": {'seed1': 204800000, 'seed2': 204800000, 'seed3':204800000},
    "pop_sp_ppo_6net_invisible": {'seed1': 460800000, 'seed2': 460800000, 'seed3':460800000},
    "pop_sp_ppo_9net_invisible": {'seed1': 512000000, 'seed2': 512000000, 'seed3':512000000},
    "pop_sp_ppo_12net_invisible": {'seed1': 768000000, 'seed2': 768000000, 'seed3':768000000},
    "pop_sp_ppo_15net_invisible": {'seed1': 819200000, 'seed2': 819200000, 'seed3':819200000},
        }


# Output directory
saved_fig_dir = f"plots/population/fc/msg_len_{message_length}"
os.makedirs(saved_fig_dir, exist_ok=True)

num_networks_list = [2, 3, 6, 9, 12, 15]


def gather_stats(checkpoints_dict, label):
    mean_ls, std_ls = [], []
    mean_topsim, std_topsim = [], []
    mean_posdis, std_posdis = [], []
    mean_ic, std_ic = [], []
    mean_sr, std_sr = [], []
    corr_list = []

    for num_networks in num_networks_list:
        ls, topsim, posdis, ic, sr = [], [], [], [], []

        for seed in range(1, 4):
            if num_networks >= 3:
                model_name = f"pop{'_sp' if label == 'SP' else ''}_ppo_{num_networks}net_invisible"
            else:
                model_name = "dec_sp_ppo_invisible" if label == "SP" else "dec_ppo_invisible"

            ckpt_name = checkpoints_dict[model_name][f"seed{seed}"]
            combination_name = f"grid5_img3_ni2_nw4_ms10_{ckpt_name}"
            scores_path = f"../../logs/vary_n_pop/msg_len_{message_length}/{model_name}/{combination_name}_seed{seed}/sim_scores.npz"

            scores = np.load(scores_path)
            avg_sr = compute_avg_sr(scores['sr_mat'])

            ic.append(scores['ic'])
            ls.append(scores['avg_sim'])
            topsim.append(scores['avg_topsim'])
            posdis.append(scores['avg_posdis'])
            sr.append(avg_sr)
            corr_list.append([num_networks, scores['avg_topsim']])

        mean_ls.append(np.mean(ls))
        std_ls.append(np.std(ls))
        mean_topsim.append(np.mean(topsim))
        std_topsim.append(np.std(topsim))
        mean_posdis.append(np.mean(posdis))
        std_posdis.append(np.std(posdis))
        mean_ic.append(np.mean(ic))
        std_ic.append(np.std(ic))
        mean_sr.append(np.mean(sr))
        std_sr.append(np.std(sr))
        
    corr_list = np.array(corr_list)
    return mean_ls, std_ls, mean_topsim, std_topsim, mean_ic, std_ic, mean_sr, std_sr, mean_posdis, std_posdis, corr_list

# Collect stats for both conditions
sp_stats = gather_stats(checkpoints_dict_sp, label="SP")
base_stats = gather_stats(checkpoints_dict, label="Base")

sns.set(style="whitegrid")
# Helper for plotting
def plot_metric(y1, std1, y2, std2, ylabel, filename, loc='lower right'):
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 20, 
                        'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20})
    plt.figure(figsize=(6, 5))

    sns.lineplot(x=num_networks_list, y=y2, marker="s", label='XP+SP')
    plt.fill_between(num_networks_list, np.array(y2) - np.array(std2), np.array(y2) + np.array(std2), alpha=0.2)

    sns.lineplot(x=num_networks_list, y=y1, marker="o", label='XP')
    plt.fill_between(num_networks_list, np.array(y1) - np.array(std1), np.array(y1) + np.array(std1), alpha=0.2)

    plt.xlabel('Population Size')
    plt.ylabel(ylabel)
    plt.xticks(num_networks_list)
    plt.legend(loc=loc)
    plt.tight_layout()
    plt.savefig(os.path.join(saved_fig_dir, filename))
    plt.close()

# Compute Correlation between population size and topsim
corr_sp = np.corrcoef(sp_stats[10].T)
corr_base = np.corrcoef(base_stats[10].T)
print("SP's corr(topsim, size)", corr_sp)
print("Base's corr(topsim, size)",corr_base)

# Plot all metrics
plot_metric(base_stats[0], base_stats[1], sp_stats[0], sp_stats[1], 'Language Similarity', 'ls_vs_pop.pdf')
plot_metric(base_stats[2], base_stats[3], sp_stats[2], sp_stats[3], 'Topographic Similarity', 'topsim_vs_pop.pdf')
plot_metric(base_stats[4], base_stats[5], sp_stats[4], sp_stats[5], 'Interchangeability', 'ic_vs_pop.pdf')
plot_metric(base_stats[6], base_stats[7], sp_stats[6], sp_stats[7], 'Success Rate', 'sr_vs_pop.pdf')
plot_metric(base_stats[8], base_stats[9], sp_stats[8], sp_stats[9], 'Positional Disentanglement', 'posdis_vs_pop.pdf', 'upper right')