from __future__ import annotations

import argparse
import contextlib
import csv
import pickle
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import editdistance as _editdistance
except ImportError:
    _editdistance = None

graph_structure = "ring"
# This script lives in foraging_game/analysis, and the scoreg logs are under
# foraging_game/logs rather than the parent emergent_language/logs.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

with open("/dev/null", "w") as devnull, contextlib.redirect_stdout(devnull):
    from utils.graph_gen import clq_pairs_100, compute_all_pairs_shortest_paths

LANGSIM_ROOT = REPO_ROOT / "logs"  / "scoreg" / "langsim"
SR_ROOT = REPO_ROOT / "logs" / "scoreg" / "sr"
SCORE_ROOT = REPO_ROOT / "logs" / "scoreg" / f"{graph_structure}_metrics"
FIG_DIR = REPO_ROOT / "analysis" / "plots" / graph_structure

NUM_NETWORKS = 100
SEEDS = (1, 2, 3)
MODE = "test"
COMBINATION_NAME = "grid5_img3_ni2_nw4_ms30_comm_field100"
MODEL_NAMES = (
    f"{graph_structure}_ppo_100net_invisible",
    f"sp_{graph_structure}_ppo_100net_invisible",
)
SCORE_FILE = "sim_scores.npz"
SEED_CACHE_FILE = "seed_scores.npz"
SUMMARY_CSV = "summary_by_training_mode.csv"

MODEL_LABELS = {
    f"{graph_structure}_ppo_100net_invisible": "XP",
    f"sp_{graph_structure}_ppo_100net_invisible": "XP+SP",
}
PLOT_COLORS = {
    "XP": "#3B5BA9",
    "XP+SP": "#C44E52",
}


def ring_pairs(num_networks: int) -> list[tuple[int, int]]:
    return [(sender, (sender + 1) % num_networks) for sender in range(num_networks)]


def graph_pairs(num_networks: int) -> list[tuple[int, int]]:
    if graph_structure == "ring":
        return ring_pairs(num_networks)
    if graph_structure == "clq":
        if num_networks != 100:
            raise ValueError("graph_structure='clq' currently expects NUM_NETWORKS=100")
        pairs = [tuple(pair) for pair in clq_pairs_100]
        if (num_networks - 1, 0) not in pairs:
            pairs.append((num_networks - 1, 0))
        return pairs
    raise ValueError(f"Unknown graph_structure: {graph_structure}")


def graph_distance_matrix(num_networks: int) -> np.ndarray:
    distances = compute_all_pairs_shortest_paths(num_networks, graph_pairs(num_networks))
    return np.asarray(distances, dtype=np.float32)


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


def edit_distance(seq_a: Iterable[int], seq_b: Iterable[int]) -> int:
    a = list(seq_a)
    b = list(seq_b)
    if _editdistance is not None:
        return int(_editdistance.eval(a, b))

    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        curr = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def load_graph_messages(model_name: str, seed: int) -> dict[int, list[np.ndarray]]:
    message_data = {}
    for sender_id, receiver_id in graph_pairs(NUM_NETWORKS):
        if sender_id in message_data:
            continue
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
            raise FileNotFoundError(f"Missing graph trajectory: {traj_path}")

        print(f"loading {model_name} seed{seed} pair {sender_id}-{receiver_id}")
        message_data[sender_id] = extract_sender_messages(load_trajectory(traj_path))

    missing_senders = sorted(set(range(NUM_NETWORKS)).difference(message_data))
    if missing_senders:
        raise ValueError(f"No trajectory selected for sender ids: {missing_senders}")
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
                sim_sum += 1.0 if denom == 0 else 1.0 - edit_distance(m1, m2) / denom
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


def summarize_similarity_mat(similarity_mat: np.ndarray) -> float:
    mask = np.tril(np.ones_like(similarity_mat, dtype=bool), k=-1)
    values = np.asarray(similarity_mat[mask], dtype=np.float32)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def summarize_sr_mat(sr_mat: np.ndarray) -> tuple[float, float]:
    diag_mask = np.eye(sr_mat.shape[0], dtype=bool)
    cross_mask = ~diag_mask
    self_values = np.asarray(sr_mat[diag_mask], dtype=np.float32)
    cross_values = np.asarray(sr_mat[cross_mask], dtype=np.float32)
    self_values = self_values[np.isfinite(self_values)]
    cross_values = cross_values[np.isfinite(cross_values)]
    self_sr = float(np.mean(self_values)) if self_values.size else float("nan")
    cross_sr = float(np.mean(cross_values)) if cross_values.size else float("nan")
    return self_sr, cross_sr


def score_path(model_name: str, seed: int) -> Path:
    return SCORE_ROOT / model_name / f"{COMBINATION_NAME}_seed{seed}" / SCORE_FILE


def seed_cache_path(model_name: str) -> Path:
    return SCORE_ROOT / model_name / f"{COMBINATION_NAME}_{SEED_CACHE_FILE}"


def training_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def simple_model_name(model_name: str) -> str:
    return model_name.replace("_ppo_", "_").replace("_invisible", "")


def load_cached_scores(model_name: str, seed: int) -> dict[str, np.ndarray | float]:
    path = score_path(model_name, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached score file: {path}")

    with np.load(path) as scores:
        loaded = {key: scores[key] for key in scores.files}

    if "similarity_mat" not in loaded and "ls_mat" in loaded:
        loaded["similarity_mat"] = loaded["ls_mat"]
    if "avg_sim" in loaded:
        loaded["avg_sim"] = float(loaded["avg_sim"])
    return loaded


def load_seed_cache(model_name: str) -> dict[str, np.ndarray]:
    path = seed_cache_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing consolidated seed cache: {path}")

    with np.load(path) as scores:
        return {key: scores[key] for key in scores.files}


def save_model_seed_cache(
    model_name: str,
    seeds: list[int],
    similarity_mats: list[np.ndarray],
    sr_mats: list[np.ndarray],
    ls_by_distance: list[np.ndarray],
    sr_by_distance: list[np.ndarray],
    avg_sims: list[float],
    self_srs: list[float],
    cross_srs: list[float],
    distance_mat: np.ndarray,
    graph_distances: np.ndarray,
):
    path = seed_cache_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    scores = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "graph_pairs": np.array(graph_pairs(NUM_NETWORKS), dtype=np.int64),
        "graph_distance_mat": distance_mat,
        "graph_distances": graph_distances,
    }
    if similarity_mats:
        scores["similarity_mats"] = np.asarray(similarity_mats, dtype=np.float32)
        scores["ls_mats"] = np.asarray(similarity_mats, dtype=np.float32)
        scores["avg_sims"] = np.asarray(avg_sims, dtype=np.float32)
        scores["ls_by_graph_distance"] = np.asarray(ls_by_distance, dtype=np.float32)
    if sr_mats:
        scores["sr_mats"] = np.asarray(sr_mats, dtype=np.float32)
        scores["self_srs"] = np.asarray(self_srs, dtype=np.float32)
        scores["cross_srs"] = np.asarray(cross_srs, dtype=np.float32)
        scores["sr_by_graph_distance"] = np.asarray(sr_by_distance, dtype=np.float32)

    np.savez(path, **scores)


def save_seed_scores(
    model_name: str,
    seed: int,
    similarity_mat: np.ndarray | None,
    avg_sim: float | None,
    sr_mat: np.ndarray | None,
    graph_distances: np.ndarray | None,
    ls_by_distance: np.ndarray | None,
    sr_by_distance: np.ndarray | None,
):
    path = score_path(model_name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    scores = {}
    if path.exists():
        with np.load(path) as existing:
            scores.update({key: existing[key] for key in existing.files})

    if similarity_mat is not None:
        scores["similarity_mat"] = similarity_mat
        scores["ls_mat"] = similarity_mat
        scores["avg_sim"] = avg_sim
    if sr_mat is not None:
        scores["sr_mat"] = sr_mat
    if graph_distances is not None:
        scores["graph_distances"] = graph_distances
    if ls_by_distance is not None:
        scores["ls_by_graph_distance"] = ls_by_distance
    if sr_by_distance is not None:
        scores["sr_by_graph_distance"] = sr_by_distance

    scores["graph_pairs"] = np.array(graph_pairs(NUM_NETWORKS), dtype=np.int64)
    np.savez(path, **scores)


def plot_heatmap(matrix: np.ndarray, saved_fig_path: Path, vmin: float, vmax: float, cbar: bool = False):
    saved_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    with sns.axes_style("white"):
        sns.heatmap(
            matrix,
            square=True,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=cbar,
            xticklabels=False,
            yticklabels=False,
            linewidths=0,
            ax=ax,
        )
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout(pad=0.05)
    fig.savefig(saved_fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_ls_matrix_debug(
    model_name: str,
    seed: int,
    similarity_mat: np.ndarray,
    avg_sim: float | None,
):
    diagonal = np.diag(similarity_mat)
    max_diag_error = float(np.nanmax(np.abs(diagonal - 1.0)))
    non_one_diag = np.where(~np.isclose(diagonal, 1.0, rtol=1e-6, atol=1e-6))[0]

    print("\nSelected LS matrix debug")
    print(f"model: {model_name}")
    print(f"seed: {seed}")
    print(f"shape: {similarity_mat.shape}")
    print(f"lower-triangle avg LS, excluding diagonal: {avg_sim}")
    print(f"diagonal min/max: {np.nanmin(diagonal):.8f} / {np.nanmax(diagonal):.8f}")
    print(f"max |diagonal - 1.0|: {max_diag_error:.8g}")
    if non_one_diag.size:
        print(f"diagonal entries not close to 1.0: {non_one_diag.tolist()}")
    else:
        print("all diagonal entries are close to 1.0")

    with np.printoptions(precision=4, suppress=True, linewidth=200, threshold=np.inf):
        print("diagonal:")
        print(diagonal)
        print("LS matrix:")
        print(similarity_mat)


def plot_model_comparison(
    avg_matrix_by_model: dict[str, np.ndarray],
    saved_fig_path: Path,
    vmin: float,
    vmax: float,
):
    saved_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(avg_matrix_by_model), figsize=(8 * len(avg_matrix_by_model), 7))
    if len(avg_matrix_by_model) == 1:
        axes = [axes]

    for ax, (model_name, matrix) in zip(axes, avg_matrix_by_model.items()):
        sns.heatmap(
            matrix,
            square=True,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            linewidths=0,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout(pad=0.05)
    plt.savefig(saved_fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sr_ls_heatmap_grid(
    avg_sr_by_model: dict[str, np.ndarray],
    avg_ls_by_model: dict[str, np.ndarray],
    saved_fig_path: Path,
):
    saved_fig_path.parent.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    xp_model = f"{graph_structure}_ppo_100net_invisible"
    xpsp_model = f"sp_{graph_structure}_ppo_100net_invisible"
    panel_specs = [
        ("SR (XP)", avg_sr_by_model.get(xp_model), 0.3, 1.0),
        ("SR (XP+SP)", avg_sr_by_model.get(xpsp_model), 0.3, 1.0),
        ("LS (XP)", avg_ls_by_model.get(xp_model), 0.2, 0.6),
        ("LS (XP+SP)", avg_ls_by_model.get(xpsp_model), 0.2, 0.6),
    ]

    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0))
    for ax, panel_label, (title, matrix, vmin, vmax) in zip(axes.flat, panel_labels, panel_specs):
        if matrix is None:
            ax.axis("off")
            continue
        sns.heatmap(
            matrix,
            square=True,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            cbar_kws={"fraction": 0.046, "pad": 0.02},
            xticklabels=False,
            yticklabels=False,
            linewidths=0,
            ax=ax,
        )
        ax.set_title(f"{panel_label} {title}", fontsize=28, pad=14)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    fig.savefig(saved_fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(saved_fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def load_avg_matrices_for_graph(graph_name: str) -> dict[tuple[str, str], np.ndarray]:
    matrices = {}
    for condition, model_name in (
        ("XP", f"{graph_name}_ppo_100net_invisible"),
        ("XP+SP", f"sp_{graph_name}_ppo_100net_invisible"),
    ):
        avg_path = (
            REPO_ROOT
            / "logs"
            / "scoreg"
            / f"{graph_name}_metrics"
            / model_name
            / f"{COMBINATION_NAME}_avg_sim_sr_mat.npz"
        )
        if not avg_path.exists():
            raise FileNotFoundError(f"Missing averaged matrix file: {avg_path}")
        with np.load(avg_path) as scores:
            matrices[(condition, "SR")] = np.asarray(scores["avg_sr_mat"], dtype=np.float32)
            matrices[(condition, "LS")] = np.asarray(scores["avg_similarity_mat"], dtype=np.float32)
    return matrices


def plot_ring_clq_heatmap_grid(saved_fig_path: Path):
    saved_fig_path.parent.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    try:
        graph_matrices = {
            "Ring": load_avg_matrices_for_graph("ring"),
            "CLQ": load_avg_matrices_for_graph("clq"),
        }
    except FileNotFoundError as exc:
        print(f"Skipping combined Ring/CLQ heatmap grid: {exc}")
        return

    panel_specs = [
        ("Ring-XP", graph_matrices["Ring"][("XP", "SR")], 0.3, 1.0),
        ("Ring-XP+SP", graph_matrices["Ring"][("XP+SP", "SR")], 0.3, 1.0),
        ("Clq-XP", graph_matrices["CLQ"][("XP", "SR")], 0.3, 1.0),
        ("Clq-XP+SP", graph_matrices["CLQ"][("XP+SP", "SR")], 0.3, 1.0),
        ("Ring-XP", graph_matrices["Ring"][("XP", "LS")], 0.2, 0.6),
        ("Ring-XP+SP", graph_matrices["Ring"][("XP+SP", "LS")], 0.2, 0.6),
        ("Clq-XP", graph_matrices["CLQ"][("XP", "LS")], 0.2, 0.6),
        ("Clq-XP+SP", graph_matrices["CLQ"][("XP+SP", "LS")], 0.2, 0.6),
    ]
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)")

    fig, axes = plt.subplots(2, 4, figsize=(23, 10.2))
    for ax, panel_label, (title, matrix, vmin, vmax) in zip(axes.flat, panel_labels, panel_specs):
        sns.heatmap(
            matrix,
            square=True,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            cbar_kws={"fraction": 0.046, "pad": 0.02},
            xticklabels=False,
            yticklabels=False,
            linewidths=0,
            ax=ax,
        )
        ax.set_title(f"{panel_label} {title}", fontsize=24, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout(w_pad=1.0, h_pad=1.3)
    fig.savefig(saved_fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(saved_fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def average_score_by_graph_distance(
    score_mat: np.ndarray,
    distance_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_distances = distance_mat[np.isfinite(distance_mat)]
    finite_distances = finite_distances[finite_distances > 0]
    max_distance = int(np.max(finite_distances))
    distances = np.arange(1, max_distance + 1)
    avg_score = np.full(max_distance, np.nan, dtype=np.float32)

    for distance in distances:
        values = []
        for row in range(1, score_mat.shape[0]):
            for col in range(row):
                graph_distance = distance_mat[row, col]
                if graph_distance == distance:
                    values.append(score_mat[row, col])
        finite_values = np.asarray(values, dtype=np.float32)
        finite_values = finite_values[np.isfinite(finite_values)]
        avg_score[distance - 1] = float(np.mean(finite_values)) if finite_values.size else np.nan

    return distances, avg_score


def set_paper_style():
    sns.set_theme(
        context="paper",
        style="ticks",
        font="DejaVu Sans",
        rc={
            "axes.labelsize": 16,
            "axes.linewidth": 1.1,
            "font.size": 16,
            "legend.fontsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "lines.linewidth": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def plot_score_by_graph_distance(
    score_distance_by_model: dict[str, list[np.ndarray]],
    saved_fig_path: Path,
    ylabel: str,
    legend_loc: str = "best",
):
    saved_fig_path.parent.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    max_distance = 0
    for model_name, per_seed_scores in score_distance_by_model.items():
        if not per_seed_scores:
            continue
        per_seed_scores = np.asarray(per_seed_scores, dtype=np.float32)
        distances = np.arange(1, per_seed_scores.shape[1] + 1)
        mean_scores = np.nanmean(per_seed_scores, axis=0)
        std_scores = np.nanstd(per_seed_scores, axis=0)
        label = training_label(model_name)
        color = PLOT_COLORS.get(label)

        ax.plot(
            distances,
            mean_scores,
            color=color,
            label=label,
        )
        ax.fill_between(
            distances,
            mean_scores - std_scores,
            mean_scores + std_scores,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        max_distance = max(max_distance, len(distances))

    ax.set_xlabel("Distance")
    ax.set_ylabel(ylabel)
    if max_distance:
        ax.set_xlim(1, max_distance)
        tick_step = max(1, int(np.ceil(max_distance / 8)))
        ticks = np.arange(1, max_distance + 1, tick_step)
        if ticks[-1] != max_distance:
            ticks = np.append(ticks, max_distance)
        ax.set_xticks(ticks)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    ax.legend(frameon=False, loc=legend_loc)
    fig.tight_layout(pad=0.3)
    fig.savefig(saved_fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute or load graph-structure language similarity and SR metrics."
    )
    parser.add_argument(
        "--graph-structure",
        choices=("ring", "clq"),
        default=graph_structure,
        help="Graph structure to analyze.",
    )
    parser.add_argument(
        "--ls-mode",
        choices=("compute", "load", "skip"),
        default="load",
        help="Whether to recompute LS from trajectories, load LS from sim_scores.npz, or skip LS.",
    )
    parser.add_argument(
        "--sr-mode",
        choices=("compute", "load", "skip"),
        default="load",
        help=(
            "Whether to load raw SR from logs/scoreg/sr and save it, load cached SR from "
            "sim_scores.npz, or skip SR."
        ),
    )
    parser.add_argument(
        "--print-ls-matrix",
        action="store_true",
        help="Print one selected LS matrix, its diagonal, and diagonal-vs-1.0 diagnostics.",
    )
    parser.add_argument(
        "--print-ls-model",
        default=None,
        help=(
            "Model name to print when --print-ls-matrix is set. Defaults to the first "
            "model for the selected graph structure."
        ),
    )
    parser.add_argument(
        "--print-ls-seed",
        type=int,
        default=SEEDS[0],
        help="Seed to print when --print-ls-matrix is set.",
    )
    return parser.parse_args()


def configure_graph(graph_name: str):
    global graph_structure, SCORE_ROOT, FIG_DIR, MODEL_NAMES, MODEL_LABELS

    graph_structure = graph_name
    SCORE_ROOT = REPO_ROOT / "logs" / "scoreg" / f"{graph_structure}_metrics"
    FIG_DIR = REPO_ROOT / "analysis" / "plots" / graph_structure
    MODEL_NAMES = (
        f"{graph_structure}_ppo_100net_invisible",
        f"sp_{graph_structure}_ppo_100net_invisible",
    )
    MODEL_LABELS = {
        f"{graph_structure}_ppo_100net_invisible": "XP",
        f"sp_{graph_structure}_ppo_100net_invisible": "XP+SP",
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def save_summary_csv(rows: list[dict[str, float | str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "training",
        "model_name",
        "avg_similarity_mean",
        "avg_similarity_std",
        "self_sr_mean",
        "self_sr_std",
        "cross_sr_mean",
        "cross_sr_std",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, float | str]]):
    print("\nAverage metrics across the entire population")
    print("Training | Avg. Similarity | Self-SR | Cross-SR")
    print("-" * 57)
    for row in rows:
        print(
            f"{row['training']:>8} | "
            f"{row['avg_similarity_mean']:.4f} +/- {row['avg_similarity_std']:.4f} | "
            f"{row['self_sr_mean']:.4f} +/- {row['self_sr_std']:.4f} | "
            f"{row['cross_sr_mean']:.4f} +/- {row['cross_sr_std']:.4f}"
        )


def main():
    args = parse_args()
    configure_graph(args.graph_structure)
    set_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_ROOT.mkdir(parents=True, exist_ok=True)
    distance_mat = graph_distance_matrix(NUM_NETWORKS)
    default_graph_distances = np.arange(1, int(np.nanmax(distance_mat)) + 1)
    avg_similarity_by_model = {}
    avg_sr_by_model = {}
    ls_distance_by_model = {}
    sr_distance_by_model = {}
    summary_rows = []
    printed_ls_matrix = False
    print_ls_model = args.print_ls_model or MODEL_NAMES[0]

    for model_name in MODEL_NAMES:
        seed_similarity_mats = []
        seed_sr_mats = []
        seed_avg_sims = []
        seed_self_srs = []
        seed_cross_srs = []
        processed_seeds = []
        ls_distance_by_model[model_name] = []
        sr_distance_by_model[model_name] = []
        consolidated_cache = None
        if args.ls_mode == "load" or args.sr_mode == "load":
            with contextlib.suppress(FileNotFoundError):
                consolidated_cache = load_seed_cache(model_name)

        for seed in SEEDS:
            cached_scores = None
            similarity_mat = None
            avg_sim = None
            graph_distances = None
            ls_by_distance = None
            sr_by_distance = None
            sr_mat = None

            if args.ls_mode == "compute":
                message_data = load_graph_messages(model_name, seed)
                similarity_mat, avg_sim = compute_similarity(message_data)
                graph_distances, ls_by_distance = average_score_by_graph_distance(
                    similarity_mat, distance_mat
                )
                print(f"{model_name} seed{seed} computed avg language similarity: {avg_sim}")
            elif args.ls_mode == "load":
                if consolidated_cache is not None and "similarity_mats" in consolidated_cache:
                    seed_idx = np.where(consolidated_cache["seeds"] == seed)[0]
                    if seed_idx.size == 0:
                        raise KeyError(f"Seed {seed} is missing from {seed_cache_path(model_name)}")
                    cache_idx = int(seed_idx[0])
                    similarity_mat = np.asarray(
                        consolidated_cache["similarity_mats"][cache_idx], dtype=np.float32
                    )
                    avg_sim = float(consolidated_cache["avg_sims"][cache_idx])
                    graph_distances = np.asarray(consolidated_cache["graph_distances"], dtype=np.int64)
                    ls_by_distance = np.asarray(
                        consolidated_cache["ls_by_graph_distance"][cache_idx], dtype=np.float32
                    )
                else:
                    cached_scores = load_cached_scores(model_name, seed)
                    similarity_mat = np.asarray(cached_scores["similarity_mat"], dtype=np.float32)
                    avg_sim = float(cached_scores.get("avg_sim", summarize_similarity_mat(similarity_mat)))
                    graph_distances, ls_by_distance = average_score_by_graph_distance(
                        similarity_mat, distance_mat
                    )
                print(f"{model_name} seed{seed} loaded avg language similarity: {avg_sim}")

            if (
                args.print_ls_matrix
                and not printed_ls_matrix
                and args.ls_mode != "skip"
                and model_name == print_ls_model
                and seed == args.print_ls_seed
            ):
                print_ls_matrix_debug(model_name, seed, similarity_mat, avg_sim)
                printed_ls_matrix = True

            if args.sr_mode == "compute":
                sr_mat = load_sr_mat(model_name, seed)
                graph_distances, sr_by_distance = average_score_by_graph_distance(sr_mat, distance_mat)
                print(f"{model_name} seed{seed} loaded raw SR matrix from {SR_ROOT}")
            elif args.sr_mode == "load":
                if consolidated_cache is not None and "sr_mats" in consolidated_cache:
                    seed_idx = np.where(consolidated_cache["seeds"] == seed)[0]
                    if seed_idx.size == 0:
                        raise KeyError(f"Seed {seed} is missing from {seed_cache_path(model_name)}")
                    cache_idx = int(seed_idx[0])
                    sr_mat = np.asarray(consolidated_cache["sr_mats"][cache_idx], dtype=np.float32)
                    graph_distances = np.asarray(consolidated_cache["graph_distances"], dtype=np.int64)
                    sr_by_distance = np.asarray(
                        consolidated_cache["sr_by_graph_distance"][cache_idx], dtype=np.float32
                    )
                else:
                    if cached_scores is None:
                        cached_scores = load_cached_scores(model_name, seed)
                    if "sr_mat" not in cached_scores:
                        raise KeyError(f"Cached score file is missing sr_mat: {score_path(model_name, seed)}")
                    sr_mat = np.asarray(cached_scores["sr_mat"], dtype=np.float32)
                    graph_distances, sr_by_distance = average_score_by_graph_distance(sr_mat, distance_mat)
                print(f"{model_name} seed{seed} loaded cached SR matrix")

            if args.ls_mode != "skip":
                ls_distance_by_model[model_name].append(ls_by_distance)
                seed_similarity_mats.append(similarity_mat)
                seed_avg_sims.append(float(avg_sim))
            if args.sr_mode != "skip":
                sr_distance_by_model[model_name].append(sr_by_distance)
                seed_sr_mats.append(sr_mat)
                self_sr, cross_sr = summarize_sr_mat(sr_mat)
                seed_self_srs.append(self_sr)
                seed_cross_srs.append(cross_sr)
            if args.ls_mode != "skip" or args.sr_mode != "skip":
                processed_seeds.append(seed)

            if args.ls_mode == "compute" or args.sr_mode == "compute":
                save_seed_scores(
                    model_name,
                    seed,
                    similarity_mat,
                    avg_sim,
                    sr_mat,
                    graph_distances,
                    ls_by_distance,
                    sr_by_distance,
                )

        if processed_seeds and (args.ls_mode == "compute" or args.sr_mode == "compute"):
            save_model_seed_cache(
                model_name,
                processed_seeds,
                seed_similarity_mats,
                seed_sr_mats,
                ls_distance_by_model[model_name],
                sr_distance_by_model[model_name],
                seed_avg_sims,
                seed_self_srs,
                seed_cross_srs,
                distance_mat,
                default_graph_distances,
            )

        avg_similarity_mat = None
        avg_sr_mat = None
        if seed_similarity_mats:
            avg_similarity_mat = np.nanmean(np.array(seed_similarity_mats), axis=0)
            avg_similarity_by_model[model_name] = avg_similarity_mat
        if seed_sr_mats:
            avg_sr_mat = np.nanmean(np.array(seed_sr_mats), axis=0)
            avg_sr_by_model[model_name] = avg_sr_mat

        avg_scores = {
            "graph_pairs": np.array(graph_pairs(NUM_NETWORKS), dtype=np.int64),
            "graph_distance_mat": distance_mat,
            "graph_distances": default_graph_distances,
        }
        if avg_similarity_mat is not None:
            avg_scores["avg_similarity_mat"] = avg_similarity_mat
            avg_scores["avg_ls_by_graph_distance"] = np.nanmean(
                np.array(ls_distance_by_model[model_name]), axis=0
            )
            avg_scores["std_ls_by_graph_distance"] = np.nanstd(
                np.array(ls_distance_by_model[model_name]), axis=0
            )
            plot_heatmap(
                avg_similarity_mat,
                FIG_DIR / f"{simple_model_name(model_name)}_avg_ls.png",
                vmin=0.2,
                vmax=0.6,
                cbar=True,
            )
        if avg_sr_mat is not None:
            avg_scores["avg_sr_mat"] = avg_sr_mat
            avg_scores["avg_sr_by_graph_distance"] = np.nanmean(
                np.array(sr_distance_by_model[model_name]), axis=0
            )
            avg_scores["std_sr_by_graph_distance"] = np.nanstd(
                np.array(sr_distance_by_model[model_name]), axis=0
            )
            plot_heatmap(
                avg_sr_mat,
                FIG_DIR / f"{simple_model_name(model_name)}_avg_sr.png",
                vmin=0.3,
                vmax=1.0,
                cbar=True,
            )

        ls_mean, ls_std = mean_std(seed_avg_sims)
        self_sr_mean, self_sr_std = mean_std(seed_self_srs)
        cross_sr_mean, cross_sr_std = mean_std(seed_cross_srs)
        summary_rows.append(
            {
                "training": training_label(model_name),
                "model_name": model_name,
                "avg_similarity_mean": ls_mean,
                "avg_similarity_std": ls_std,
                "self_sr_mean": self_sr_mean,
                "self_sr_std": self_sr_std,
                "cross_sr_mean": cross_sr_mean,
                "cross_sr_std": cross_sr_std,
            }
        )

        (SCORE_ROOT / model_name).mkdir(parents=True, exist_ok=True)
        np.savez(
            SCORE_ROOT / model_name / f"{COMBINATION_NAME}_avg_sim_sr_mat.npz",
            **avg_scores,
        )

    if args.print_ls_matrix and not printed_ls_matrix:
        print(
            "\nWarning: no LS matrix was printed. Check --ls-mode, --print-ls-model, "
            f"and --print-ls-seed. Requested model={print_ls_model}, seed={args.print_ls_seed}."
        )

    if avg_similarity_by_model:
        plot_model_comparison(
            avg_similarity_by_model,
            FIG_DIR / f"{graph_structure}_vs_sp_{graph_structure}_{COMBINATION_NAME}_avg_similarity.png",
            vmin=0.2,
            vmax=0.6,
        )
        plot_score_by_graph_distance(
            ls_distance_by_model,
            FIG_DIR / f"{graph_structure}_vs_sp_{graph_structure}_{COMBINATION_NAME}_ls_by_graph_distance.png",
            ylabel="Language Similarity",
        )
        plot_score_by_graph_distance(
            ls_distance_by_model,
            FIG_DIR / f"{graph_structure}_vs_sp_{graph_structure}_{COMBINATION_NAME}_ls_by_graph_distance.pdf",
            ylabel="Language Similarity",
        )
    if avg_sr_by_model:
        plot_model_comparison(
            avg_sr_by_model,
            FIG_DIR / f"{graph_structure}_vs_sp_{graph_structure}_{COMBINATION_NAME}_avg_sr.png",
            vmin=0.3,
            vmax=1.0,
        )
        plot_score_by_graph_distance(
            sr_distance_by_model,
            FIG_DIR / f"{graph_structure}_vs_sp_{graph_structure}_{COMBINATION_NAME}_sr_by_graph_distance.png",
            ylabel="Success Rate",
            legend_loc="lower right",
        )
        plot_score_by_graph_distance(
            sr_distance_by_model,
            FIG_DIR / f"{graph_structure}_vs_sp_{graph_structure}_{COMBINATION_NAME}_sr_by_graph_distance.pdf",
            ylabel="Success Rate",
            legend_loc="lower right",
        )

    if avg_similarity_by_model and avg_sr_by_model:
        plot_sr_ls_heatmap_grid(
            avg_sr_by_model,
            avg_similarity_by_model,
            FIG_DIR / f"{graph_structure}_100net_avg_sr_ls_heatmaps.png",
        )
        plot_ring_clq_heatmap_grid(
            REPO_ROOT / "analysis" / "plots" / "ring_clq_100net_avg_sr_ls_heatmaps.png"
        )

    save_summary_csv(summary_rows, SCORE_ROOT / SUMMARY_CSV)
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
