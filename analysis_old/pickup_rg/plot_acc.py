import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import pandas as pd
import numpy as np
import os

pickup_high_avg_accuracy_dict_emb = {'item_score': [0.5513333333333333, 0.602, 0.61, 0.562, 0.566, 0.5586666666666666, 0.5546666666666666, 0.5626666666666666, 0.574, 0.39266666666666666, 0.4106666666666667, 0.406, 0.448, 0.6026666666666667, 0.594, 0.5513333333333333, 0.5346666666666666, 0.5446666666666666], 
'item_loc_x': [0.5186666666666667, 0.494, 0.4886666666666667, 0.514, 0.516, 0.514, 0.526, 0.5286666666666666, 0.5193333333333333, 0.5633333333333334, 0.552, 0.554, 0.49, 0.5306666666666666, 0.5206666666666667, 0.5153333333333333, 0.5173333333333333, 0.5146666666666667], 
'item_loc_y': [0.20933333333333334, 0.18666666666666668, 0.18733333333333332, 0.198, 0.20066666666666666, 0.20266666666666666, 0.19933333333333333, 0.216, 0.21666666666666667, 0.20666666666666667, 0.20466666666666666, 0.2, 0.202, 0.20733333333333334, 0.20733333333333334, 0.19066666666666668, 0.18866666666666668, 0.198]}


pickup_high_avg_accuracy_dict_token = {'item_score': [0.43333333333333335, 0.4186666666666667, 0.41533333333333333, 0.4106666666666667, 0.426, 0.41933333333333334, 0.3426666666666667, 0.37, 0.36733333333333335, 0.3293333333333333, 0.328, 0.33066666666666666, 0.3446666666666667, 0.4166666666666667, 0.41333333333333333, 0.394, 0.37933333333333336, 0.38466666666666666], 'item_loc_x': [0.5186666666666667, 0.5113333333333333, 0.5066666666666667, 0.5166666666666667, 0.514, 0.522, 0.5113333333333333, 0.5173333333333333, 0.5173333333333333, 0.5373333333333333, 0.5406666666666666, 0.5453333333333333, 0.5093333333333333, 0.5006666666666667, 0.5046666666666667, 0.5093333333333333, 0.49733333333333335, 0.5026666666666667], 'item_loc_y': [0.20066666666666666, 0.184, 0.18733333333333332, 0.19266666666666668, 0.20933333333333334, 0.19666666666666666, 0.212, 0.21733333333333332, 0.21866666666666668, 0.20733333333333334, 0.21133333333333335, 0.208, 0.21333333333333335, 0.20066666666666666, 0.20066666666666666, 0.20133333333333334, 0.20333333333333334, 0.206]}
# Define your chance levels
chance_levels = {
    'pickup_high': {
        'score': 0.1,
        'horizontal position': 0.20,
        'vertical position': 0.50
    },
    'pickup_temporal': {
        'spawn time': 0.167,
        'horizontal position': 0.20,
        'vertical position': 0.25
    }
}

saved_dir = "plots/decoding"
os.makedirs(saved_dir, exist_ok=True)

key2var = { "item_score" : "score",
            "item_loc_x" : "vertical position",
             "item_loc_y" : "horizontal position"}

# Display-friendly names
variable_display_names = {
    'score': 'Score',
    'spawn time': 'Time',
    'horizontal position': 'H-Pos',
    'vertical position': 'V-Pos'
}

# Create records
records = []
for task_type in ['pickup_high']:
    for input_type, label in [('emb', 'Msg Embedding'), ('token', ' Integer Msg')]:
        acc_dict = globals()[f'{task_type}_avg_accuracy_dict_{input_type}']
        for key, accs in acc_dict.items():
            var = key2var[key]
            records.append({
                'Task': task_type,
                'TaskLabel': task_type.replace('_', ' ').title(),
                'Variable': variable_display_names.get(var, var.title()),
                'Input': label,
                'Mean': np.mean(accs),
                'Std': np.std(accs),
                'Chance': chance_levels[task_type][var]
            })

df = pd.DataFrame(records)
df['Group'] = df['TaskLabel'] + '\n' + df['Variable']
sns.set(style="whitegrid")



task_order = ['pickup_high']
task_titles = {'pickup_high': 'Pickup High Score', 'pickup_temporal': 'Pickup Temporal Order'}
# Load the Paired palette
original_palette = sns.color_palette("Paired")
color_palette = original_palette[2:] + original_palette[:2] # Green

for i, task in enumerate(task_order):
    # ax = axes[i]
    # Create FacetGrid-style subplots

    plt.figure(figsize=(6, 6))
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24
    })
    subdf = df[df['Task'] == task].copy()
    subdf['X'] = subdf['Variable']
    # Draw barplot without error bars
    ax = sns.barplot(
        data=subdf,
        x="X", y="Mean", hue="Input",
        palette=color_palette, ci=None,
        errorbar=None
    )
    # Compute correct positions for error bars
    # Match each patch (bar) to a Mean/Std
    patches = ax.patches
    means = subdf['Mean'].values
    stds = subdf['Std'].values

    # Only take visible bars (exclude legend handles etc.)
    visible_patches = [p for p in patches if p.get_height() > 0]

    # Ensure lengths match
    assert len(visible_patches) == len(means), \
        f"Expected {len(means)} bars, but found {len(visible_patches)}"

    # Plot each error bar
    for patch, mean, std in zip(visible_patches, means, stds):
        x = patch.get_x() + patch.get_width() / 2.
        ax.errorbar(x, mean, yerr=std, fmt='none', c='black', capsize=5, linewidth=1)
                
    # Fix the chance lines: add one line per variable across both bars (Token + Emb)
    variables = subdf['Variable'].unique()
    x_ticks = ax.get_xticks()
    for xtick, var in zip(x_ticks, variables):
        chance = subdf[subdf['Variable'] == var]['Chance'].iloc[0]
        # Adjust x range slightly to match the width of grouped bars
        ax.hlines(y=chance, xmin=xtick - 0.4, xmax=xtick + 0.4, 
                  color='red', linestyle='--', linewidth=1)

    ax.set_title("")
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    # if i == 0:
    #     ax.legend_.remove()
    # else:

    # Create legend handles (Input types from seaborn)
    handles, labels = ax.get_legend_handles_labels()

    # Add red dashed line for chance level
    chance_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Chance Level')
    handles.append(chance_handle)
    labels.append('Chance Level')

    # Add legend to top right of each subplot
    ax.legend(handles, labels, loc='upper right', frameon=True, fontsize=20, title='Input Type')

    # plt.suptitle('Decoding Accuracy per Task and Variable', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(saved_dir, f"{task}_rg.pdf"))
    plt.close()
