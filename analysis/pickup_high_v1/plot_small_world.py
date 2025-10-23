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

with open("plots/population/small_world/sr_lang_sim/mean_std_by_model.pkl", "rb") as f:
    log_data = pickle.load(f)
saved_fig_dir = f"plots/small_world"

ckptname2label = {
            "wsk4p02_ppo_15net_invisible": "WS",
            "optk2e30_ppo_15net_invisible": "LRC",
                # "ccnet_sp_ppo_15net_invisible": "XP+SP",
                "ccnet_ppo_15net_invisible": "Clq",
                
                }
distance_list = [i for i in range(1,6)]

# Set seaborn style and global font settings
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 20, 
                    'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20})

# Plot 1: Success Rate vs Distance
plt.figure(figsize=(6, 5))
sns.set_palette(sns.color_palette()[2:])
for model_name in ckptname2label.keys():
    mean_sr = np.array(log_data[model_name]['mean_sr'])
    std_sr = np.array(log_data[model_name]['std_sr'])
    cur_x = distance_list[:len(mean_sr)]
    sns.lineplot(x=cur_x, y=mean_sr, marker="o", label=ckptname2label[model_name])
    plt.fill_between(cur_x, 
                    mean_sr - std_sr, 
                    mean_sr + std_sr, 
                    alpha=0.2)

plt.xlabel('Distance')
plt.ylabel('Success Rate')
plt.ylim(ymin=0.4)
plt.xticks(distance_list)
plt.legend(loc='upper right', labelspacing=0.2, handlelength=1.5, handleheight=0.4, borderpad=0.2) 
plt.tight_layout()
plt.savefig(os.path.join(saved_fig_dir, "sm_sr_vs_distance.pdf"))
plt.close()

# Plot 2: Language Similarity vs Distance
plt.figure(figsize=(6, 5))
for model_name in ckptname2label.keys():
    mean_ls = np.array(log_data[model_name]['mean_ls'])
    std_ls = np.array(log_data[model_name]['std_ls'])
    cur_x = distance_list[:len(mean_ls)]
    sns.lineplot(x=cur_x, y=mean_ls, marker="o", label=ckptname2label[model_name])
    plt.fill_between(cur_x, 
                    mean_ls - std_ls, 
                    mean_ls + std_ls, 
                    alpha=0.2)

plt.xlabel('Distance')
plt.ylabel('Language Similarity')
plt.ylim()
plt.xticks(distance_list)
plt.legend(loc='upper right', labelspacing=0.2, handlelength=1.5, handleheight=0.4, borderpad=0.2) #, framealpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(saved_fig_dir, "sm_ls_vs_distance.pdf"))
plt.close()

