import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import os


# v1: standard setting where agents spawn on different sides
# pickup_high_avg_accuracy_dict_emb = {'score': [0.2986666666666667, 0.296, 0.31, 0.26266666666666666, 0.25533333333333336, 0.252, 0.25933333333333336, 0.274, 0.2733333333333333, 0.376, 0.3393333333333333, 0.3313333333333333, 0.414, 0.3913333333333333, 0.38333333333333336, 0.38133333333333336, 0.36, 0.35733333333333334, 0.37333333333333335, 0.408, 0.37133333333333335, 0.454, 0.45466666666666666, 0.402, 0.3973333333333333, 0.42, 0.438], 'vertical position': [0.6073333333333333, 0.6126666666666667, 0.6093333333333333, 0.5826666666666667, 0.5926666666666667, 0.5986666666666667, 0.6413333333333333, 0.6473333333333333, 0.6526666666666666, 0.8646666666666667, 0.8633333333333333, 0.8666666666666667, 0.8526666666666667, 0.8533333333333334, 0.8446666666666667, 0.724, 0.724, 0.7353333333333333, 0.8486666666666667, 0.872, 0.8546666666666667, 0.8366666666666667, 0.83, 0.834, 0.8346666666666667, 0.8213333333333334, 0.846], 'horizontal position': [0.30866666666666664, 0.31533333333333335, 0.31933333333333336, 0.30466666666666664, 0.31066666666666665, 0.296, 0.30133333333333334, 0.30866666666666664, 0.32, 0.244, 0.25266666666666665, 0.23533333333333334, 0.2833333333333333, 0.2793333333333333, 0.2966666666666667, 0.2773333333333333, 0.27266666666666667, 0.2853333333333333, 0.3, 0.2633333333333333, 0.25666666666666665, 0.2966666666666667, 0.282, 0.2846666666666667, 0.30933333333333335, 0.2966666666666667, 0.276]}
# pickup_high_avg_accuracy_dict_token = {'score': [0.16333333333333333, 0.162, 0.162, 0.16266666666666665, 0.16133333333333333, 0.15933333333333333, 0.17733333333333334, 0.186, 0.17666666666666667, 0.24333333333333335, 0.19866666666666666, 0.20266666666666666, 0.23866666666666667, 0.21333333333333335, 0.222, 0.28, 0.25066666666666665, 0.26066666666666666, 0.35733333333333334, 0.32466666666666666, 0.2886666666666667, 0.36533333333333334, 0.38, 0.332, 0.26066666666666666, 0.24333333333333335, 0.254], 'vertical position': [0.576, 0.556, 0.576, 0.5613333333333334, 0.5506666666666666, 0.5433333333333333, 0.5773333333333334, 0.582, 0.5753333333333334, 0.6233333333333333, 0.6806666666666666, 0.652, 0.676, 0.6773333333333333, 0.692, 0.5846666666666667, 0.608, 0.602, 0.676, 0.6586666666666666, 0.648, 0.5393333333333333, 0.558, 0.5386666666666666, 0.7453333333333333, 0.7433333333333333, 0.7533333333333333], 'horizontal position': [0.25266666666666665, 0.25066666666666665, 0.25466666666666665, 0.25533333333333336, 0.25466666666666665, 0.24066666666666667, 0.26, 0.266, 0.2753333333333333, 0.24866666666666667, 0.25733333333333336, 0.244, 0.25866666666666666, 0.248, 0.274, 0.25666666666666665, 0.2713333333333333, 0.27066666666666667, 0.24333333333333335, 0.24333333333333335, 0.252, 0.256, 0.242, 0.23733333333333334, 0.25133333333333335, 0.23266666666666666, 0.21866666666666668]}

# v7: agents and items spawn randomly. item_position_train is disjointed with item_position_test
pickup_high_posgen_avg_accuracy_dict_emb = {'score': [0.4033333333333333, 0.404, 0.4093333333333333, 0.4026666666666667, 0.4086666666666667, 0.41533333333333333, 0.234, 0.23133333333333334, 0.23733333333333334, 0.22133333333333333, 0.23466666666666666, 0.238, 0.4026666666666667, 0.42333333333333334, 0.42866666666666664, 0.44133333333333336, 0.43133333333333335, 0.42933333333333334], 'vertical position': [0.3, 0.2846666666666667, 0.274, 0.30866666666666664, 0.25666666666666665, 0.284, 0.30133333333333334, 0.2786666666666667, 0.314, 0.2946666666666667, 0.3, 0.29533333333333334, 0.2693333333333333, 0.25866666666666666, 0.26266666666666666, 0.282, 0.26, 0.272], 'horizontal position': [0.354, 0.328, 0.322, 0.3273333333333333, 0.31533333333333335, 0.3293333333333333, 0.3333333333333333, 0.32, 0.30533333333333335, 0.29533333333333334, 0.2926666666666667, 0.30133333333333334, 0.3233333333333333, 0.304, 0.304, 0.282, 0.29533333333333334, 0.3]}
pickup_high_posgen_avg_accuracy_dict_token = {'score': [0.186, 0.184, 0.20466666666666666, 0.19733333333333333, 0.19733333333333333, 0.182, 0.19133333333333333, 0.18266666666666667, 0.18466666666666667, 0.17533333333333334, 0.18266666666666667, 0.158, 0.246, 0.24533333333333332, 0.258, 0.26066666666666666, 0.25333333333333335, 0.23333333333333334], 'vertical position': [0.26666666666666666, 0.24266666666666667, 0.24266666666666667, 0.24466666666666667, 0.22133333333333333, 0.23666666666666666, 0.25, 0.22666666666666666, 0.24133333333333334, 0.252, 0.24866666666666667, 0.24866666666666667, 0.232, 0.22266666666666668, 0.238, 0.224, 0.234, 0.23], 'horizontal position': [0.30133333333333334, 0.2753333333333333, 0.28933333333333333, 0.238, 0.25066666666666665, 0.26, 0.246, 0.18133333333333335, 0.21333333333333335, 0.20933333333333334, 0.21933333333333332, 0.23666666666666666, 0.22333333333333333, 0.22733333333333333, 0.208, 0.21133333333333335, 0.208, 0.23333333333333334]}
eval_tasks = [
            # 'pickup_high', 
            # 'pickup_temporal', 
            'pickup_high_posgen',
            ]
# Define your chance levels
chance_levels = {
    'pickup_high': {
        'score': 0.1,
        'horizontal position': 0.20,
        'vertical position': 0.50
    },
    'pickup_high_posgen': {
        'score': 0.1,
        'horizontal position': 0.20,
        'vertical position': 0.20
    },
    'pickup_temporal': {
        'spawn time': 0.167,
        'horizontal position': 0.20,
        'vertical position': 0.25
    }
}


# Display-friendly names
variable_display_names = {
    'score': 'Score',
    'spawn time': 'Time',
    'horizontal position': 'H-Pos',
    'vertical position': 'V-Pos'
}

# Create records
records = []
for task_type in eval_tasks:
    for input_type, label in [('emb', 'Msg Embedding'), ('token', ' Integer Msg')]:
        acc_dict = globals()[f'{task_type}_avg_accuracy_dict_{input_type}']
        for var, accs in acc_dict.items():
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




task_titles = {'pickup_high': 'Pickup High Score', 'pickup_temporal': 'Pickup Temporal Order'}
# Load the Paired palette
original_palette = sns.color_palette("Paired")
color_palette = original_palette[2:] + original_palette[:2] # Green

for i, task in enumerate(eval_tasks):
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
    if i == 0 and len(eval_tasks) > 1:
        ax.legend_.remove()
    else:

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
    os.makedirs("plots/decoding", exist_ok=True)
    plt.savefig(f"plots/decoding/acc_{task}.pdf")
    plt.close()