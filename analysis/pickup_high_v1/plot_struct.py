# Plot 4 columns × 2 rows
# Columns: population structures (WS, Opt, Clq, Rng)
# Rows: Language Similarity (LS) on top, Success Rate (SR) on bottom
# Each subplot shows two conditions: XP and XP+SP

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ----------------------------
# Load data
# ----------------------------
with open("plots/population/struct_pop/sr_lang_sim/mean_std_by_model.pkl", "rb") as f:
    log_data = pickle.load(f)

saved_fig_dir = "plots/struct_pop"
os.makedirs(saved_fig_dir, exist_ok=True)
print(log_data.keys())
# Map model checkpoint names to Structure_Condition labels
ckptname2label = {
    "ring_ppo_15net_invisible" : "Ring_XP",
    # "wsk4p02_ppo_15net_invisible":      "WS_XP",
    "optk2e30_ppo_15net_invisible":     "Opt_XP",
    "ccnet_ppo_15net_invisible":        "Clq_XP",

    "ring_sp_ppo_15net_invisible" : "Ring_XP+SP",
    # "wsk4p02_sp_ppo_15net_invisible":   "WS_XP+SP",
    "optk2e30_sp_ppo_15net_invisible":  "Opt_XP+SP",
    "ccnet_sp_ppo_15net_invisible":     "Clq_XP+SP",
    # If you have ring/other structures, add them here similarly
}

distance_list = [i for i in range(1, 8)]

# ----------------------------
# Styling
# ----------------------------
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 18,
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16
})

# Fixed order for the 4 columns (the last column will be blank if not present)
structures_order = ["WS", "Opt", "Clq", "Ring"]
structure_titles = {"WS": "WS", "Opt": "Opt", "Clq": "Clq", "Ring": "Ring"}

# Two conditions to show per subplot
conditions_order = ["XP", "XP+SP"]

# Consistent condition styling across all subplots
palette = sns.color_palette("deep", n_colors=2)
cond2style = {
    "XP":    {"color": palette[0], "linestyle": "-",  "marker": "o", "label": "XP"},
    "XP+SP": {"color": palette[1], "linestyle": "--", "marker": "s", "label": "XP+SP"},
}

# Build (structure -> {condition -> list of model_names})
by_struct_cond = {}
for model_name, label in ckptname2label.items():
    if "_" not in label:
        continue
    struct, cond = label.split("_", 1)  # e.g., "WS", "XP" or "XP+SP"
    by_struct_cond.setdefault(struct, {}).setdefault(cond, []).append(model_name)

# ----------------------------
# Create 2×4 grid
# ----------------------------
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 10), sharex=False, sharey=False)

# Helper to plot one metric row for a given structure/axis
def plot_metric(ax, struct, metric_key_mean, metric_key_std, ylabel=None):
    any_plotted = False
    for cond in conditions_order:
        model_list = by_struct_cond.get(struct, {}).get(cond, [])
        if not model_list:
            continue

        # If multiple models per (struct, cond) exist, plot each or aggregate – here we plot each model.
        # If you prefer aggregating across models, you can average mean/std here instead.
        for model_name in model_list:
            mean_vals = np.array(log_data[model_name][metric_key_mean])
            std_vals  = np.array(log_data[model_name][metric_key_std])
            cur_x = distance_list[:len(mean_vals)]

            style = cond2style[cond]
            ax.plot(cur_x, mean_vals,
                    linestyle=style["linestyle"], marker=style["marker"],
                    label=style["label"], color=style["color"])
            ax.fill_between(cur_x, mean_vals - std_vals, mean_vals + std_vals,
                            alpha=0.2, color=style["color"])
            any_plotted = True

    ax.set_xticks(distance_list)
    ax.set_xlabel("Distance")
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Only show a small legend per subplot if something was plotted
    if any_plotted:
        handles = [
            Line2D([0], [0], linestyle=cond2style["XP"]["linestyle"], marker=cond2style["XP"]["marker"],
                   color=cond2style["XP"]["color"], label="XP"),
            Line2D([0], [0], linestyle=cond2style["XP+SP"]["linestyle"], marker=cond2style["XP+SP"]["marker"],
                   color=cond2style["XP+SP"]["color"], label="XP+SP"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False, handlelength=1.5)

    return any_plotted

# Plot each column (structure)
for col, struct in enumerate(structures_order):
    # Top row: Language Similarity
    ax_ls = axes[0, col]
    has_ls = plot_metric(ax_ls, struct, "mean_ls", "std_ls", ylabel="Language Similarity" if col == 0 else None)
    ax_ls.set_title(structure_titles[struct])

    # Bottom row: Success Rate
    ax_sr = axes[1, col]
    has_sr = plot_metric(ax_sr, struct, "mean_sr", "std_sr", ylabel="Success Rate" if col == 0 else None)
    # Optional: enforce lower bound like original code
    if has_sr:
        ymin, ymax = ax_sr.get_ylim()
        ax_sr.set_ylim(max(0.4, ymin), ymax)

    # If nothing plotted in this column, blank both axes
    if not (has_ls or has_sr):
        ax_ls.axis("off")
        ax_sr.axis("off")
        ax_ls.set_title(f"{structure_titles[struct]} (No data)")

# Global legend (conditions) at the top (optional; we already have per-subplot legends)
# Uncomment to add a single shared legend:
# shared_handles = [
#     Line2D([0], [0], linestyle=cond2style["XP"]["linestyle"], marker=cond2style["XP"]["marker"],
#            color=cond2style["XP"]["color"], label="XP"),
#     Line2D([0], [0], linestyle=cond2style["XP+SP"]["linestyle"], marker=cond2style["XP+SP"]["marker"],
#            color=cond2style["XP+SP"]["color"], label="XP+SP"),
# ]
# fig.legend(handles=shared_handles, loc="upper center", ncol=2, frameon=False)

plt.tight_layout()
outpath = os.path.join(saved_fig_dir, "grid_ls_sr_by_structure.png")
plt.savefig(outpath)
plt.close()

print(f"Saved figure to: {outpath}")
