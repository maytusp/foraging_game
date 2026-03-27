import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ---- CONFIG ----

ROOT_DIR = "../../logs/repcom_dataset_len4_all"  # adjust if needed

checkpoints_dict = {
    "pop_ppo_3net_invisible": {
        "seed1": 204800000,
        "seed2": 204800000,
        "seed3": 204800000,
    },
    "pop_ppo_3net": {
        "seed1": 256000000,
        "seed2": 256000000,
        "seed3": 256000000,
    },
}

model2numnet = {
    "pop_ppo_3net_invisible": 3,
    "pop_ppo_3net": 3,
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
    n = len(all_CZ_values)
    return float(all_CZ_values.mean()), float(all_CZ_values.std(ddof=0) / math.sqrt(n))


def main():
    for model_name in checkpoints_dict.keys():
        xp_mean, xp_se = collect_CZ_for_model(model_name)
        print(f"Model = {model_name}, repcom = {xp_mean} ± {xp_se}")


if __name__ == "__main__":
    main()
