from __future__ import annotations

import pickle
from pathlib import Path

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
LANGSIM_ROOT = REPO_ROOT / "logs" / "vary_n_pop" / "layout2" / "langsim"
SR_ROOT = REPO_ROOT / "logs" / "vary_n_pop" / "layout2" / "sr"
SCORE_ROOT = REPO_ROOT / "logs" / "vary_n_pop" / "layout2" / "ring_sr_lang_sim"
FIG_DIR = REPO_ROOT / "analysis" / "pickup_high_v1" / "plots" / "population" / "ring" / "sr_lang_sim"

NUM_NETWORKS = 100
SEEDS = (1, 2, 3)
MODE = "test"
COMBINATION_NAME = "grid5_img3_ni2_nw4_ms30_comm_field100"
MODEL_NAMES = (
    "ring_ppo_100net_invisible",
    "sp_ring_ppo_100net_invisible",
)


def ring_pairs(num_networks: int) -> list[tuple[int, int]]:
    return [(sender, (sender + 1) % num_networks) for sender in range(num_networks)]


def load_trajectory(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_episode_length(log_s_messages: np.ndarray) -> int:
    end_idxs = np.where(log_s_messages[:, 0] == -1)[0]
    if end_idxs.size == 0:
        return log_s_messages.shape[0]
    return int(end_idxs[0])


def extract_sender_messages(log_data) -> list[np.ndarray]:
    messages = []
    for episode_data in log_data.values():
        log_s_messages = episode_data["log_s_messages"]
        ep_len = get_episode_length(log_s_messages)
        messages.append(log_s_messages[:ep_len, 0].flatten())
    return messages


def load_ring_messages(model_name: str, seed: int) -> dict[int, list[np.ndarray]]:
    message_data = {}
    for sender_id, receiver_id in ring_pairs(NUM_NETWORKS):
        traj_path = (
            LANGSIM_ROOT
            / model_name
            / f"{sender_id}-{receiver_id}"
            / COMBINATION_NAME
            / f"seed{seed}"
            / f"mode_{MODE}"
            / "normal"
            / "trajectory.pkl"
        )
        if not traj_path.exists():
            raise FileNotFoundError(f"Missing ring trajectory: {traj_path}")

        print(f"loading {model_name} seed{seed} pair {sender_id}-{receiver_id}")
        message_data[sender_id] = extract_sender_messages(load_trajectory(traj_path))
    return message_data


def compute_similarity(message_data: dict[int, list[np.ndarray]]) -> tuple[np.ndarray, float]:
    n_samples = min(len(messages) for messages in message_data.values())
    if n_samples == 0:
        raise ValueError("No language-similarity samples found.")

    similarity_mat = np.zeros((NUM_NETWORKS, NUM_NETWORKS), dtype=np.float32)

    for first_sender in range(NUM_NETWORKS):
        for second_sender in range(NUM_NETWORKS):
            sim_sum = 0.0
            for episode_idx in range(n_samples):
                m1 = [x for x in message_data[first_sender][episode_idx] if x != -1]
                m2 = [x for x in message_data[second_sender][episode_idx] if x != -1]
                denom = max(len(m1), len(m2))
                sim_sum += 1.0 if denom == 0 else 1.0 - editdistance.eval(m1, m2) / denom
            similarity_mat[first_sender, second_sender] = sim_sum / n_samples

    lower_mask = np.ones_like(similarity_mat)
    lower_mask[np.triu_indices_from(lower_mask, k=0)] = 0
    avg_sim = float(np.sum(similarity_mat * lower_mask) / np.sum(lower_mask))
    return similarity_mat, avg_sim


def load_sr_mat(model_name: str, seed: int) -> np.ndarray:
    sr_path = (
        SR_ROOT
        / model_name
        / COMBINATION_NAME
        / f"seed{seed}"
        / f"mode_{MODE}"
        / "normal"
        / "sr_eval.npz"
    )
    if not sr_path.exists():
        print(f"Warning: missing SR file, filling with NaN: {sr_path}")
        return np.full((NUM_NETWORKS, NUM_NETWORKS), np.nan, dtype=np.float32)
    return np.load(sr_path)["sr_mat"]


def plot_heatmap(matrix: np.ndarray, saved_fig_path: Path, vmin: float, vmax: float, cbar: bool = False):
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    saved_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    with sns.axes_style("white"):
        sns.heatmap(
            matrix,
            mask=mask,
            square=True,
            cmap="YlGnBu",
            vmin=vmin,
            vmax=vmax,
            cbar=cbar,
            xticklabels=False,
            yticklabels=False,
        )
    plt.tight_layout()
    plt.savefig(saved_fig_path)
    plt.close()


def plot_model_comparison(avg_similarity_by_model: dict[str, np.ndarray], saved_fig_path: Path):
    saved_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(avg_similarity_by_model), figsize=(8 * len(avg_similarity_by_model), 7))
    if len(avg_similarity_by_model) == 1:
        axes = [axes]

    for ax, (model_name, matrix) in zip(axes, avg_similarity_by_model.items()):
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        sns.heatmap(
            matrix,
            mask=mask,
            square=True,
            cmap="YlGnBu",
            vmin=0.2,
            vmax=0.6,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.set_title(model_name)

    plt.tight_layout()
    plt.savefig(saved_fig_path)
    plt.close()


def compute_ls_by_graph_distance(ls_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_distance = ls_mat.shape[0] // 2
    distances = np.arange(1, max_distance + 1)
    avg_ls = np.zeros(max_distance, dtype=np.float32)

    for distance in distances:
        values = []
        for row in range(1, ls_mat.shape[0]):
            for col in range(row):
                graph_distance = min(abs(row - col), ls_mat.shape[0] - abs(row - col))
                if graph_distance == distance:
                    values.append(ls_mat[row, col])
        avg_ls[distance - 1] = float(np.mean(values)) if values else np.nan

    return distances, avg_ls


def plot_ls_by_graph_distance(ls_distance_by_model: dict[str, list[np.ndarray]], saved_fig_path: Path):
    saved_fig_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    plt.figure(figsize=(7, 5))
    for model_name, per_seed_ls in ls_distance_by_model.items():
        per_seed_ls = np.array(per_seed_ls)
        distances = np.arange(1, per_seed_ls.shape[1] + 1)
        mean_ls = np.nanmean(per_seed_ls, axis=0)
        std_ls = np.nanstd(per_seed_ls, axis=0)

        sns.lineplot(x=distances, y=mean_ls, marker="o", label=model_name)
        plt.fill_between(distances, mean_ls - std_ls, mean_ls + std_ls, alpha=0.2)

    plt.xlabel("Graph Distance")
    plt.ylabel("Language Similarity")
    plt.xlim(1, NUM_NETWORKS // 2)
    plt.tight_layout()
    plt.savefig(saved_fig_path)
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_ROOT.mkdir(parents=True, exist_ok=True)
    avg_similarity_by_model = {}
    ls_distance_by_model = {}

    for model_name in MODEL_NAMES:
        avg_similarity_mat = np.zeros((NUM_NETWORKS, NUM_NETWORKS), dtype=np.float32)
        avg_sr_mat = np.zeros((NUM_NETWORKS, NUM_NETWORKS), dtype=np.float32)
        ls_distance_by_model[model_name] = []

        for seed in SEEDS:
            message_data = load_ring_messages(model_name, seed)
            similarity_mat, avg_sim = compute_similarity(message_data)
            # sr_mat = load_sr_mat(model_name, seed)
            graph_distances, ls_by_distance = compute_ls_by_graph_distance(similarity_mat)
            ls_distance_by_model[model_name].append(ls_by_distance)

            print(f"{model_name} seed{seed} avg language similarity: {avg_sim}")

            saved_score_dir = SCORE_ROOT / model_name / f"{COMBINATION_NAME}_seed{seed}"
            saved_score_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                saved_score_dir / "sim_scores.npz",
                similarity_mat=similarity_mat,
                ls_mat=similarity_mat,
                avg_sim=avg_sim,
                # sr_mat=sr_mat,
                ring_pairs=np.array(ring_pairs(NUM_NETWORKS), dtype=np.int64),
                graph_distances=graph_distances,
                ls_by_graph_distance=ls_by_distance,
            )

            avg_similarity_mat += similarity_mat
            # avg_sr_mat += sr_mat

        avg_similarity_mat /= len(SEEDS)
        avg_sr_mat /= len(SEEDS)
        avg_similarity_by_model[model_name] = avg_similarity_mat

        np.savez(
            SCORE_ROOT / model_name / f"{COMBINATION_NAME}_avg_sim_sr_mat.npz",
            avg_similarity_mat=avg_similarity_mat,
            # avg_sr_mat=avg_sr_mat,
            graph_distances=np.arange(1, NUM_NETWORKS // 2 + 1),
            avg_ls_by_graph_distance=np.nanmean(
                np.array(ls_distance_by_model[model_name]), axis=0
            ),
            std_ls_by_graph_distance=np.nanstd(
                np.array(ls_distance_by_model[model_name]), axis=0
            ),
        )
        plot_heatmap(
            avg_similarity_mat,
            FIG_DIR / f"{model_name}_{COMBINATION_NAME}_avg_similarity.png",
            vmin=0.2,
            vmax=0.6,
        )
        # plot_heatmap(
        #     avg_sr_mat,
        #     FIG_DIR / f"{model_name}_{COMBINATION_NAME}_avg_sr.png",
        #     vmin=0.3,
        #     vmax=1.0,
        # )

    plot_model_comparison(
        avg_similarity_by_model,
        FIG_DIR / f"ring_vs_sp_ring_{COMBINATION_NAME}_avg_similarity.png",
    )
    plot_ls_by_graph_distance(
        ls_distance_by_model,
        FIG_DIR / f"ring_vs_sp_ring_{COMBINATION_NAME}_ls_by_graph_distance.png",
    )


if __name__ == "__main__":
    main()
