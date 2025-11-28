import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- CONFIG ----






def load_CZ_from_file(fname):
    """Read a prequential_scores.txt and return the C_Z value."""
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("C_Z"):
                # line format: C_Z: 1.4147969831082823
                return float(line.split(":")[1].strip())
    raise ValueError(f"C_Z line not found in {fname}")


def collect_CZ_for_model(root_dir, model_name, ckpt_name, combination_name, num_networks=3):
    """
    For a given model_name (e.g. 'pop_ppo_3net_invisible'),
    aggregate C_Z across seeds and agents.

    Returns
    -------
    mean_CZ : float
    std_CZ  : float
    """
    all_CZ_values = []

    for seed_idx in range(1, 4):
        seed_key = f"seed{seed_idx}"

        for agent_id in range(num_networks):
            score_path = os.path.join(
                root_dir,
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

    xp_means, xp_stds = [], []
    root_dir = "../../logs/repcom_dataset_len4_n_words"  # adjust if needed
    saved_fig_dir = f"plots/population/fc/"
    model_name = "pop_ppo_3net_invisible"
    ckpt_name = "307200000"
    num_networks = 3
    n_words_list = [4,8,16,32]
    for n_words in n_words_list:
        combination_name = f"grid5_img3_ni2_nw{n_words}_ms10_{ckpt_name}"
        xp_mean, xp_std = collect_CZ_for_model(root_dir, model_name, ckpt_name, combination_name, num_networks)
        print(f"Vocab Size = {n_words}, repcom = {xp_mean} +- {xp_std}")
        xp_means.append(xp_mean)
        xp_stds.append(xp_std)

    xp_means = np.array(xp_means, dtype=float)
    xp_stds = np.array(xp_stds, dtype=float)

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 20, 
                        'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20})
    plt.figure(figsize=(6, 5))

    sns.lineplot(x=n_words_list, y=xp_means, marker="o", label='XP')
    plt.fill_between(n_words_list, 
            xp_means - xp_stds,
            xp_means + xp_stds,
            alpha=0.2)


    plt.xlabel("Vocabulary Size")
    plt.xticks(n_words_list)
    plt.ylabel("repcom")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(saved_fig_dir, "repcom_vs_nwords.png"))
    plt.close()


if __name__ == "__main__":
    main()
