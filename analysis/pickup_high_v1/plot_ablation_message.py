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
    "pop_ppo_3net_invisible/grid5_img3_ni2_nw4_ms10_332800000": "Inv-Com",
    "pop_ppo_3net_invisible_ablate_message/grid5_img3_ni2_nw4_ms10_358400000" : "Inv-NoCom",
    "pop_ppo_3net_ablate_message/grid5_img3_ni2_nw4_ms10_358400000": "Vis-NoCom",
    # "pop_ppo_3net_ablate_message/grid5_img5_ni2_nw4_ms10_358400000": "Vis-5x5-NoM",
}
mode = "test"
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

# PLOT HERE
# Prepare data for plotting
plot_data = {
    "Model": [], "Condition": [], "Metric": [], "Mean": [], "Std": []
}

for cond in test_conditions:
    for metric in metrics:
        for model_key, label in model_name_map.items():
            # Handle model_name part only (strip path after slash if needed)
            base_model_key = model_key

            score_list = model_score_dict[cond][metric].get(base_model_key, None)
            if score_list is not None:
                mean = np.mean(score_list)
                std = np.std(score_list)
            else:
                mean = np.nan
                std = np.nan

            plot_data["Model"].append(label)
            plot_data["Condition"].append(cond)
            plot_data["Metric"].append("Success Rate" if metric == "sr" else "Avg. Episode Length")
            plot_data["Mean"].append(mean)
            plot_data["Std"].append(std)

df = pd.DataFrame(plot_data)
print(df)

# Plotting
# Map proper filenames
metric_to_filename = {
    "Success Rate": "success_rate",
    "Avg. Episode Length": "episode_length"
}
sns.set(style="whitegrid")

for metric_name in ["Success Rate", "Avg. Episode Length"]:
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 14
    })
    plt.figure(figsize=(5, 5))

    data = df[df["Metric"] == metric_name]

    ax = sns.barplot(
        data=data,
        x="Model", y="Mean", hue="Condition",
        palette="Paired", ci=None,
        errorbar=None
    )

    # Add manual error bars
    for i, bar in enumerate(ax.patches):
        # Only add to the top row of bars (the first n groups)
        if i >= len(data):
            break
        row = data.iloc[i]
        height = bar.get_height()
        err = row["Std"]
        if not np.isnan(height) and not np.isnan(err):
            ax.errorbar(
                x=bar.get_x() + bar.get_width() / 2,
                y=height,
                yerr=err,
                fmt='none',
                ecolor='black',
                capsize=4,
                linewidth=1
            )

    ax.set_title("")  # Remove title
    ax.set_ylabel(metric_name)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    if "Success Rate" in metric_name:
        ax.legend_.remove()
    else:

        if ax.get_legend() is not None:
            ax.legend(loc="lower right", title="Eval. Cond.")

    plt.tight_layout()
    save_path = os.path.join(saved_fig_dir, f"message_ablation_{metric_to_filename[metric_name]}.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
