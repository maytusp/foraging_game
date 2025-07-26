import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.lines import Line2D


pickup_temporal_avg_accuracy_dict_token = {'spawn time': [0.2473404255319149, 0.21242653232577666, 0.21206743566992015, 0.2405821917808219, 0.2652885443583118, 0.21788194444444445, 0.23465703971119134, 0.20765027322404372, 0.2206148282097649, 0.20127504553734063, 0.18297101449275363, 0.21066907775768534, 0.234860883797054, 0.21735467565290648, 0.21056977704376548, 0.21688741721854304, 0.20100502512562815, 0.24796084828711257], 'vertical position': [0.33687943262411346, 0.3408900083963056, 0.3762200532386868, 0.4006849315068493, 0.3350559862187769, 0.3550347222222222, 0.3601083032490975, 0.4107468123861566, 0.39511754068716093, 0.4599271402550091, 0.36141304347826086, 0.3969258589511754, 0.4099836333878887, 0.4077506318449874, 0.37572254335260113, 0.39983443708609273, 0.41457286432160806, 0.37520391517128876], 'horizontal position': [0.2898936170212766, 0.2938706968933669, 0.27151730257320317, 0.2636986301369863, 0.2739018087855297, 0.3046875, 0.26714801444043323, 0.3005464480874317, 0.2739602169981917, 0.30327868852459017, 0.3016304347826087, 0.3010849909584087, 0.24959083469721768, 0.21651221566975568, 0.22378199834847234, 0.2541390728476821, 0.24371859296482412, 0.2601957585644372]}

pickup_temporal_avg_accuracy_dict_emb = {'spawn time': [0.2916666666666667, 0.2770780856423174, 0.26353149955634425, 0.261986301369863, 0.30749354005167956, 0.2526041666666667, 0.26534296028880866, 0.2641165755919854, 0.27757685352622063, 0.24408014571949, 0.26902173913043476, 0.2468354430379747, 0.3076923076923077, 0.3091828138163437, 0.2898431048720066, 0.30960264900662254, 0.3224455611390285, 0.31239804241435565], 'vertical position' : [0.39095744680851063, 0.42317380352644834, 0.4392191659272405, 0.4511986301369863, 0.4142980189491817, 0.4609375, 0.40974729241877256, 0.4371584699453552, 0.46835443037974683, 0.48633879781420764, 0.3795289855072464, 0.4258589511754069, 0.45499181669394434, 0.44566133108677336, 0.40214698596201487, 0.4420529801324503, 0.4564489112227806, 0.4070146818923328], 'horizontal position': [0.325354609929078, 0.3131821998320739, 0.3070097604259095, 0.2970890410958904, 0.32558139534883723, 0.328125, 0.2833935018050541, 0.30783242258652094, 0.27757685352622063, 0.31602914389799636, 0.29528985507246375, 0.3065099457504521, 0.3044189852700491, 0.2898062342038753, 0.3129644921552436, 0.33195364238410596, 0.29731993299832493, 0.31158238172920066]}



pickup_high_avg_accuracy_dict_emb = {'score': [0.2986666666666667, 0.296, 0.31, 0.26266666666666666, 0.25533333333333336, 0.252, 0.25933333333333336, 0.274, 0.2733333333333333, 0.376, 0.3393333333333333, 0.3313333333333333, 0.414, 0.3913333333333333, 0.38333333333333336, 0.38133333333333336, 0.36, 0.35733333333333334, 0.37333333333333335, 0.408, 0.37133333333333335, 0.454, 0.45466666666666666, 0.402, 0.3973333333333333, 0.42, 0.438], 'vertical position': [0.6073333333333333, 0.6126666666666667, 0.6093333333333333, 0.5826666666666667, 0.5926666666666667, 0.5986666666666667, 0.6413333333333333, 0.6473333333333333, 0.6526666666666666, 0.8646666666666667, 0.8633333333333333, 0.8666666666666667, 0.8526666666666667, 0.8533333333333334, 0.8446666666666667, 0.724, 0.724, 0.7353333333333333, 0.8486666666666667, 0.872, 0.8546666666666667, 0.8366666666666667, 0.83, 0.834, 0.8346666666666667, 0.8213333333333334, 0.846], 'horizontal position': [0.30866666666666664, 0.31533333333333335, 0.31933333333333336, 0.30466666666666664, 0.31066666666666665, 0.296, 0.30133333333333334, 0.30866666666666664, 0.32, 0.244, 0.25266666666666665, 0.23533333333333334, 0.2833333333333333, 0.2793333333333333, 0.2966666666666667, 0.2773333333333333, 0.27266666666666667, 0.2853333333333333, 0.3, 0.2633333333333333, 0.25666666666666665, 0.2966666666666667, 0.282, 0.2846666666666667, 0.30933333333333335, 0.2966666666666667, 0.276]}


pickup_high_avg_accuracy_dict_token = {'score': [0.16333333333333333, 0.162, 0.162, 0.16266666666666665, 0.16133333333333333, 0.15933333333333333, 0.17733333333333334, 0.186, 0.17666666666666667, 0.24333333333333335, 0.19866666666666666, 0.20266666666666666, 0.23866666666666667, 0.21333333333333335, 0.222, 0.28, 0.25066666666666665, 0.26066666666666666, 0.35733333333333334, 0.32466666666666666, 0.2886666666666667, 0.36533333333333334, 0.38, 0.332, 0.26066666666666666, 0.24333333333333335, 0.254], 'vertical position': [0.576, 0.556, 0.576, 0.5613333333333334, 0.5506666666666666, 0.5433333333333333, 0.5773333333333334, 0.582, 0.5753333333333334, 0.6233333333333333, 0.6806666666666666, 0.652, 0.676, 0.6773333333333333, 0.692, 0.5846666666666667, 0.608, 0.602, 0.676, 0.6586666666666666, 0.648, 0.5393333333333333, 0.558, 0.5386666666666666, 0.7453333333333333, 0.7433333333333333, 0.7533333333333333], 'horizontal position': [0.25266666666666665, 0.25066666666666665, 0.25466666666666665, 0.25533333333333336, 0.25466666666666665, 0.24066666666666667, 0.26, 0.266, 0.2753333333333333, 0.24866666666666667, 0.25733333333333336, 0.244, 0.25866666666666666, 0.248, 0.274, 0.25666666666666665, 0.2713333333333333, 0.27066666666666667, 0.24333333333333335, 0.24333333333333335, 0.252, 0.256, 0.242, 0.23733333333333334, 0.25133333333333335, 0.23266666666666666, 0.21866666666666668]}

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


# Display-friendly names
variable_display_names = {
    'score': 'Score',
    'spawn time': 'Time',
    'horizontal position': 'H-Pos',
    'vertical position': 'V-Pos'
}

# Create records
records = []
for task_type in ['pickup_high', 'pickup_temporal']:
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



task_order = ['pickup_high', 'pickup_temporal']
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
    if i == 0:
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
    plt.savefig(f"plots/decoding/acc_{task}.png")
    plt.close()
