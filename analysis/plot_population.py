# Plot torch-env LS, TopSim, and PosDis across population sizes.
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


POPULATION_SIZES = [2, 3, 15, 30, 60, 100]
SEEDS = [1, 2, 3]
COMBINATION_NAME = "grid5_img3_ni2_nw4_ms30_comm_field100"
SCORE_ROOT = "../../logs/vary_n_pop/layout2/sr_lang_sim"

CONDITIONS = {
    "XP": "pop_ppo_{population_size}net_invisible",
    "XP+SP": "sp_pop_ppo_{population_size}net_invisible",
}

METRICS = {
    "ls": ("avg_sim", "Language Similarity", "xpsp_langsim_vs_pop.png"),
    "topsim": ("avg_topsim", "Topographic Similarity", "xpsp_topsim_vs_pop.png"),
    "posdis": ("avg_posdis", "Positional Disentanglement", "xpsp_posdis_vs_pop.png"),
}


def model_name_for(condition, population_size):
    return CONDITIONS[condition].format(population_size=population_size)


def score_path(condition, population_size, seed):
    model_name = model_name_for(condition, population_size)
    return os.path.join(
        SCORE_ROOT,
        model_name,
        f"{COMBINATION_NAME}_seed{seed}",
        "sim_scores.npz",
    )


def load_scalar(scores, key):
    return float(scores[key]) if key in scores else np.nan


def gather_stats(condition):
    stats = {"population_size": []}
    for metric in METRICS:
        stats[f"{metric}_mean"] = []
        stats[f"{metric}_std"] = []

    corr_values = []

    for population_size in POPULATION_SIZES:
        seed_values = {metric: [] for metric in METRICS}

        for seed in SEEDS:
            path = score_path(condition, population_size, seed)
            if not os.path.exists(path):
                warnings.warn(
                    f"Skipping missing torch-env sim_scores.npz for {condition}, "
                    f"n={population_size}, seed={seed}: {path}",
                    RuntimeWarning,
                )
                for metric in METRICS:
                    seed_values[metric].append(np.nan)
                continue

            scores = np.load(path)
            for metric, (score_key, _, _) in METRICS.items():
                seed_values[metric].append(load_scalar(scores, score_key))

            corr_values.append([population_size, seed_values["topsim"][-1]])
            print(f"{condition} n={population_size} seed={seed}: {path}")

        stats["population_size"].append(population_size)
        for metric in METRICS:
            values = np.array(seed_values[metric], dtype=float)
            finite_values = values[np.isfinite(values)]
            if len(finite_values) == 0:
                stats[f"{metric}_mean"].append(np.nan)
                stats[f"{metric}_std"].append(np.nan)
            else:
                stats[f"{metric}_mean"].append(np.mean(finite_values))
                stats[f"{metric}_std"].append(np.std(finite_values))

    return stats, np.array(corr_values, dtype=float)


def finite_corr(values):
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 2:
        return np.nan
    return np.corrcoef(values.T)[0, 1]


def plot_metric(all_stats, metric, ylabel, filename, saved_fig_dir, loc="best"):
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

    fig, ax = plt.subplots(figsize=(6, 5))
    markers = {"XP": "o", "XP+SP": "s"}

    for condition, stats in all_stats.items():
        x = np.array(stats["population_size"], dtype=float)
        y = np.array(stats[f"{metric}_mean"], dtype=float)
        std = np.array(stats[f"{metric}_std"], dtype=float)
        sns.lineplot(x=x, y=y, marker=markers[condition], label=condition, ax=ax)
        ax.fill_between(x, y - std, y + std, alpha=0.2)

    ax.set_xlabel("Population Size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(POPULATION_SIZES)
    ax.legend(loc=loc)
    fig.tight_layout()
    fig.savefig(os.path.join(saved_fig_dir, filename))
    plt.close(fig)


if __name__ == "__main__":
    saved_fig_dir = "plots/population/fc/sr_lang_sim/population_metrics"
    os.makedirs(saved_fig_dir, exist_ok=True)

    all_stats = {}
    for condition in CONDITIONS:
        stats, corr_values = gather_stats(condition)
        all_stats[condition] = stats
        print(f"{condition}'s corr(topsim, size)", finite_corr(corr_values))

    for metric, (_, ylabel, filename) in METRICS.items():
        plot_metric(
            all_stats,
            metric,
            ylabel,
            filename,
            saved_fig_dir,
            loc="upper right" if metric == "posdis" else "best",
        )
