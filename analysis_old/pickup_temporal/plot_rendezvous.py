import matplotlib.pyplot as plt
import numpy as np
import os

# Rendezvous map data
inv_com = np.array([
    [4.6, 5.5, 5.1, 4.6, 0.0],
    [0.0, 6.2, 13.9, 4.7, 0.0],
    [0.0, 8.5, 12.6, 6.7, 0.0],
    [0.0, 4.7, 10.8, 4.9, 0.0],
    [0.0, 2.3, 2.5, 2.4, 0.0]
])

vis_nocom = np.array([
    [0.0, 2.5, 2.3, 2.2, 0.0],
    [0.0, 4.7, 13.1, 5.8, 0.0],
    [0.0, 6.0, 15.7, 8.3, 0.0],
    [0.0, 6.4, 12.9, 6.4, 0.0],
    [0.0, 3.7, 6.6, 3.4, 0.0]
])

inv_nocom = np.array([
    [1.7, 0.6, 0.7, 0.5, 0.0],
    [0.0, 0.6, 1.2, 0.6, 0.0],
    [0.0, 0.3, 0.3, 0.3, 0.0],
    [0.0, 0.6, 0.8, 0.4, 0.0],
    [0.0, 0.4, 0.5, 0.4, 0.0]
])
maps = {
    "Inv-Com (100%)": inv_com,
    "Vis-NoCom (100%)": vis_nocom,
    "Inv-NoCom (9.9%)": inv_nocom
}

# Find global min and max for consistent scaling
all_values = np.concatenate([arr.flatten() for arr in maps.values()])
vmin, vmax = all_values.min(), all_values.max()

# Plot heatmaps with fixed color scale
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})
for ax, (title, data) in zip(axes, maps.items()):
    im = ax.imshow(data, cmap="YlOrRd", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    # Annotate values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", color="black", fontsize=8)

# Add one shared colorbar
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Normalized Frequency (%)")

# Save figure
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "rendezvous.pdf")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()