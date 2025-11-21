import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- CONFIG ----

ROOT_DIR = "../../logs/repcom_dataset_len4_all"  # adjust if needed

checkpoints_dict = {
    "dec_ppo_invisible": {
        "seed1": 204800000,
        "seed2": 204800000,
        "seed3": 204800000,
    },
    "pop_ppo_3net_invisible": {
        "seed1": 204800000,
        "seed2": 204800000,
        "seed3": 204800000,
    },
    "pop_ppo_6net_invisible": {
        "seed1": 460800000,
        "seed2": 460800000,
        "seed3": 460800000,
    },
    "pop_ppo_9net_invisible": {
        "seed1": 512000000,
        "seed2": 512000000,
        "seed3": 512000000,
    },
    "pop_ppo_12net_invisible": {
        "seed1": 768000000,
        "seed2": 768000000,
        "seed3": 768000000,
    },
    "pop_ppo_15net_invisible": {
        "seed1": 819200000,
        "seed2": 819200000,
        "seed3": 819200000,
    },
    "dec_sp_ppo_invisible": {
        "seed1": 204800000,
        "seed2": 204800000,
        "seed3": 204800000,
    },
    "pop_sp_ppo_3net_invisible": {
        "seed1": 204800000,
        "seed2": 204800000,
        "seed3": 204800000,
    },
    "pop_sp_ppo_6net_invisible": {
        "seed1": 460800000,
        "seed2": 460800000,
        "seed3": 460800000,
    },
    "pop_sp_ppo_9net_invisible": {
        "seed1": 512000000,
        "seed2": 512000000,
        "seed3": 512000000,
    },
    "pop_sp_ppo_12net_invisible": {
        "seed1": 768000000,
        "seed2": 768000000,
        "seed3": 768000000,
    },
    "pop_sp_ppo_15net_invisible": {
        "seed1": 819200000,
        "seed2": 819200000,
        "seed3": 819200000,
    },
}

model2numnet = {
    "dec_ppo_invisible": 2,
    "pop_ppo_3net_invisible": 3,
    "pop_ppo_6net_invisible": 6,
    "pop_ppo_9net_invisible": 9,
    "pop_ppo_12net_invisible": 12,
    "pop_ppo_15net_invisible": 15,
    "dec_sp_ppo_invisible": 2,
    "pop_sp_ppo_3net_invisible": 3,
    "pop_sp_ppo_6net_invisible": 6,
    "pop_sp_ppo_9net_invisible": 9,
    "pop_sp_ppo_12net_invisible": 12,
    "pop_sp_ppo_15net_invisible": 15,
}

# Map population size -> (xp_model_name, xp+sp_model_name)
popsize_to_models = {
    2: ("dec_ppo_invisible", "dec_sp_ppo_invisible"),
    3: ("pop_ppo_3net_invisible", "pop_sp_ppo_3net_invisible"),
    6: ("pop_ppo_6net_invisible", "pop_sp_ppo_6net_invisible"),
    9: ("pop_ppo_9net_invisible", "pop_sp_ppo_9net_invisible"),
    12: ("pop_ppo_12net_invisible", "pop_sp_ppo_12net_invisible"),
    15: ("pop_ppo_15net_invisible", "pop_sp_ppo_15net_invisible"),
}


def load_CZ_from_file(fname):
    """Read a prequential_scores.txt and return the C_Z value."""
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("C_Z"):
                # line format: C_Z: 1.4147969831082823
                return float(line.split(":")[1].strip())
    raise ValueError(f"C_Z line not found in {fname}")


def collect_CZ_for_model(model_name):
    """
    For a given model_name (e.g. 'pop_ppo_3net_invisible'),
    aggregate C_Z across seeds and agents.

    Returns
    -------
    mean_CZ : float
    std_CZ  : float
    """
    num_networks = model2numnet[model_name]
    all_CZ_values = []

    for seed_idx in range(1, 4):
        seed_key = f"seed{seed_idx}"
        ckpt_name = checkpoints_dict[model_name][seed_key]
        combination_name = f"grid5_img3_ni2_nw4_ms10_{ckpt_name}"

        for agent_id in range(num_networks):
            score_path = os.path.join(
                ROOT_DIR,
                model_name,
                combination_name,
                f"seed{seed_idx}",
                f"agent{agent_id}",
                "prequential_scores.txt",
            )
            if not os.path.isfile(score_path):
                print(f"[WARN] Missing {score_path}")
                continue

            try:
                cz = load_CZ_from_file(score_path)
                all_CZ_values.append(cz)
            except Exception as e:
                print(f"[WARN] Could not parse {score_path}: {e}")

    if not all_CZ_values:
        print(f"[WARN] No C_Z values found for {model_name}")
        return np.nan, np.nan

    all_CZ_values = np.array(all_CZ_values, dtype=float)
    return float(all_CZ_values.mean()), float(all_CZ_values.std(ddof=0))


def main():
    population_sizes = sorted(popsize_to_models.keys())


    xp_means, xp_stds = [], []
    xp_sp_means, xp_sp_stds = [], []
    saved_fig_dir = f"plots/population/fc/"
    for pop_size in population_sizes:
        xp_model, xp_sp_model = popsize_to_models[pop_size]
        print(f"Processing pop_size={pop_size}, xp={xp_model}, xp+sp={xp_sp_model}")

        xp_mean, xp_std = collect_CZ_for_model(xp_model)
        xp_sp_mean, xp_sp_std = collect_CZ_for_model(xp_sp_model)

        xp_means.append(xp_mean)
        xp_stds.append(xp_std)
        xp_sp_means.append(xp_sp_mean)
        xp_sp_stds.append(xp_sp_std)

    population_sizes = np.array(population_sizes, dtype=float)
    xp_means = np.array(xp_means, dtype=float)
    xp_stds = np.array(xp_stds, dtype=float)
    xp_sp_means = np.array(xp_sp_means, dtype=float)
    xp_sp_stds = np.array(xp_sp_stds, dtype=float)
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 20, 
                        'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20})
    plt.figure(figsize=(6, 5))
    sns.lineplot(x=population_sizes, y=xp_sp_means, marker="s", label='XP+SP')
    plt.fill_between(population_sizes, 
            xp_sp_means - xp_sp_stds,
            xp_sp_means + xp_sp_stds,
            alpha=0.2)

    sns.lineplot(x=population_sizes, y=xp_means, marker="o", label='XP')
    plt.fill_between(population_sizes, 
            xp_means - xp_stds,
            xp_means + xp_stds,
            alpha=0.2)


    plt.xlabel("Population size")
    plt.xticks(population_sizes)
    plt.ylabel("repcom")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(saved_fig_dir, "repcom_vs_distance.pdf"))
    plt.close()


if __name__ == "__main__":
    main()
