from __future__ import annotations

import argparse
import contextlib
import csv
import pickle
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import editdistance as _editdistance
except ImportError:
    _editdistance = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

with open("/dev/null", "w") as devnull, contextlib.redirect_stdout(devnull):
    from utils.graph_gen import (
        clq_pairs_100,
        compute_all_pairs_shortest_paths,
        opt_pairs_100,
        ws_pairs_100,
    )

LANGSIM_ROOT = REPO_ROOT / "logs" / "scoreg" / "langsim"
SR_ROOT = REPO_ROOT / "logs" / "scoreg" / "sr"
SCORE_ROOT = REPO_ROOT / "logs" / "scoreg" / "xp_struct_metrics"
FIG_DIR = REPO_ROOT / "analysis" / "plots" / "xp_struct_compare"

NUM_NETWORKS = 100
DEFAULT_SEEDS = (1, 2, 3)
MODE = "test"
COMBINATION_NAME = "grid5_img3_ni2_nw4_ms30_comm_field100"
GRAPH_NAMES = ("ring", "clq", "ws", "opt")
SCORE_FILE = "sim_scores.npz"
SEED_CACHE_FILE = "seed_scores.npz"
SUMMARY_CSV = "summary_xp_struct_compare.csv"
DEFAULT_MAX_PLOT_DISTANCE = 8

GRAPH_LABELS = {
    "ring": "Ring",
    "clq": "CLQ",
    "ws": "WS",
    "opt": "OPT",
}


def model_candidates(graph_name: str) -> tuple[str, ...]:
    return (
        f"{graph_name}_ppo_100net_invisible",
        f"{graph_name}_pop100_net_invisible",
    )


def model_name_for_graph(graph_name: str) -> str:
    for model_name in model_candidates(graph_name):
        if (LANGSIM_ROOT / model_name).exists() or (SR_ROOT / model_name).exists():
            return model_name
    return model_candidates(graph_name)[0]


def static_graph_pairs(graph_name: str) -> list[tuple[int, int]]:
    if graph_name == "ring":
        return [(sender, (sender + 1) % NUM_NETWORKS) for sender in range(NUM_NETWORKS)]
    if graph_name == "clq":
        pairs = [tuple(pair) for pair in clq_pairs_100]
        if (NUM_NETWORKS - 1, 0) not in pairs:
            pairs.append((NUM_NETWORKS - 1, 0))
        return pairs
    if graph_name == "ws":
        return [tuple(pair) for pair in ws_pairs_100]
    if graph_name == "opt":
        return [tuple(pair) for pair in opt_pairs_100]
    raise ValueError(f"Unknown graph name: {graph_name}")


def trajectory_path(model_name: str, pair: tuple[int, int], seed: int) -> Path:
    sender_id, receiver_id = pair
    return (
        LANGSIM_ROOT
        / model_name
        / f"{sender_id}-{receiver_id}"
        / COMBINATION_NAME
        / f"seed{seed}"
        / f"mode_{MODE}"
        / "normal"
        / "trajectory.pkl"
    )


def pair_dirs_for_model(model_name: str) -> list[tuple[int, int]]:
    model_root = LANGSIM_ROOT / model_name
    if not model_root.exists():
        return []

    pairs = []
    for pair_dir in model_root.iterdir():
        if not pair_dir.is_dir() or "-" not in pair_dir.name:
            continue
        sender_id, receiver_id = pair_dir.name.split("-", maxsplit=1)
        if sender_id.isdigit() and receiver_id.isdigit():
            pairs.append((int(sender_id), int(receiver_id)))
    return sorted(set(pairs))


def graph_pairs(graph_name: str, model_name: str) -> list[tuple[int, int]]:
    static_pairs = static_graph_pairs(graph_name)
    if all((LANGSIM_ROOT / model_name / f"{a}-{b}").exists() for a, b in static_pairs):
        return static_pairs

    filesystem_pairs = pair_dirs_for_model(model_name)
    if filesystem_pairs:
        return filesystem_pairs
    return static_pairs


def selected_seed_pairs(
    model_name: str,
    seed: int,
    preferred_pairs: list[tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    selected = {}
    candidate_pairs = preferred_pairs + [
        pair for pair in pair_dirs_for_model(model_name) if pair not in set(preferred_pairs)
    ]
    for pair in candidate_pairs:
        sender_id, _receiver_id = pair
        if sender_id in selected:
            continue
        if trajectory_path(model_name, pair, seed).exists():
            selected[sender_id] = pair
    return selected


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


def load_graph_messages(graph_name: str, model_name: str, seed: int) -> dict[int, list[np.ndarray]]:
    seed_pairs = selected_seed_pairs(model_name, seed, graph_pairs(graph_name, model_name))
    missing_senders = sorted(set(range(NUM_NETWORKS)).difference(seed_pairs))
    if not seed_pairs:
        raise FileNotFoundError(
            f"{model_name} seed{seed} is missing trajectories for senders: {missing_senders}"
        )
    if missing_senders:
        print(
            f"Warning: {model_name} seed{seed} missing {len(missing_senders)} sender "
            f"trajectories; LS rows/columns for those senders will be NaN."
        )

    message_data = {}
    for sender_id in sorted(seed_pairs):
        pair = seed_pairs[sender_id]
        traj_path = trajectory_path(model_name, pair, seed)
        print(f"loading {model_name} seed{seed} pair {pair[0]}-{pair[1]}")
        message_data[sender_id] = extract_sender_messages(load_trajectory(traj_path))
    return message_data


def compute_similarity(message_data: dict[int, list[np.ndarray]]) -> tuple[np.ndarray, float]:
    n_samples = min(len(messages) for messages in message_data.values())
    if n_samples == 0:
        raise ValueError("No language-similarity samples found.")

    similarity_mat = np.full((NUM_NETWORKS, NUM_NETWORKS), np.nan, dtype=np.float32)
    present_senders = sorted(message_data)
    for first_sender in present_senders:
        for second_sender in present_senders:
            sim_sum = 0.0
            for episode_idx in range(n_samples):
                m1 = [x for x in message_data[first_sender][episode_idx] if x != -1]
                m2 = [x for x in message_data[second_sender][episode_idx] if x != -1]
                denom = max(len(m1), len(m2))
                sim_sum += 1.0 if denom == 0 else 1.0 - edit_distance(m1, m2) / denom
            similarity_mat[first_sender, second_sender] = sim_sum / n_samples

    return similarity_mat, summarize_similarity_mat(similarity_mat)


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
        raise FileNotFoundError(f"Missing SR file: {sr_path}")
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


def graph_distance_matrix(graph_name: str, model_name: str) -> np.ndarray:
    distances = compute_all_pairs_shortest_paths(NUM_NETWORKS, graph_pairs(graph_name, model_name))
    return np.asarray(distances, dtype=np.float32)


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
                if distance_mat[row, col] == distance:
                    values.append(score_mat[row, col])
        finite_values = np.asarray(values, dtype=np.float32)
        finite_values = finite_values[np.isfinite(finite_values)]
        avg_score[distance - 1] = float(np.mean(finite_values)) if finite_values.size else np.nan
    return distances, avg_score


def score_path(graph_name: str, model_name: str, seed: int) -> Path:
    return SCORE_ROOT / graph_name / model_name / f"{COMBINATION_NAME}_seed{seed}" / SCORE_FILE


def seed_cache_path(graph_name: str, model_name: str) -> Path:
    return SCORE_ROOT / graph_name / model_name / f"{COMBINATION_NAME}_{SEED_CACHE_FILE}"


def avg_score_path(graph_name: str, model_name: str) -> Path:
    return SCORE_ROOT / graph_name / model_name / f"{COMBINATION_NAME}_avg_sim_sr_mat.npz"


def load_cached_scores(graph_name: str, model_name: str, seed: int) -> dict[str, np.ndarray | float]:
    path = score_path(graph_name, model_name, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached score file: {path}")
    with np.load(path) as scores:
        loaded = {key: scores[key] for key in scores.files}
    if "similarity_mat" not in loaded and "ls_mat" in loaded:
        loaded["similarity_mat"] = loaded["ls_mat"]
    if "avg_sim" in loaded:
        loaded["avg_sim"] = float(loaded["avg_sim"])
    return loaded


def load_seed_cache(graph_name: str, model_name: str) -> dict[str, np.ndarray]:
    path = seed_cache_path(graph_name, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing consolidated seed cache: {path}")
    with np.load(path) as scores:
        return {key: scores[key] for key in scores.files}


def save_seed_scores(
    graph_name: str,
    model_name: str,
    seed: int,
    similarity_mat: np.ndarray | None,
    avg_sim: float | None,
    sr_mat: np.ndarray | None,
    graph_distances: np.ndarray | None,
    ls_by_distance: np.ndarray | None,
    sr_by_distance: np.ndarray | None,
):
    path = score_path(graph_name, model_name, seed)
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
    np.savez(path, **scores)


def save_model_seed_cache(
    graph_name: str,
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
    path = seed_cache_path(graph_name, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "graph_pairs": np.asarray(graph_pairs(graph_name, model_name), dtype=np.int64),
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


def set_paper_style():
    sns.set_theme(
        context="paper",
        style="ticks",
        font="DejaVu Sans",
        rc={
            "axes.labelsize": 16,
            "axes.linewidth": 1.1,
            "font.size": 16,
            "legend.fontsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "lines.linewidth": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def plot_heatmap_grid(
    avg_sr_by_graph: dict[str, np.ndarray],
    avg_ls_by_graph: dict[str, np.ndarray],
    saved_fig_path: Path,
):
    saved_fig_path.parent.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    panel_specs = []
    for graph_name in avg_sr_by_graph:
        panel_specs.append((f"{GRAPH_LABELS[graph_name]} SR", avg_sr_by_graph.get(graph_name), 0.3, 1.0))
    for graph_name in avg_ls_by_graph:
        panel_specs.append((f"{GRAPH_LABELS[graph_name]} LS", avg_ls_by_graph.get(graph_name), 0.2, 0.6))

    panel_labels = tuple(f"({chr(ord('a') + idx)})" for idx in range(len(panel_specs)))
    n_cols = max(len(avg_sr_by_graph), len(avg_ls_by_graph))
    fig, axes = plt.subplots(2, n_cols, figsize=(5.6 * n_cols, 10.0))
    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)
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
        ax.set_title(f"{panel_label} {title}", fontsize=24, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout(w_pad=1.0, h_pad=1.3)
    fig.savefig(saved_fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(saved_fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_score_by_graph_distance(
    score_distance_by_graph: dict[str, list[np.ndarray]],
    saved_fig_path: Path,
    ylabel: str,
    legend_loc: str = "best",
    max_plot_distance: int | None = DEFAULT_MAX_PLOT_DISTANCE,
):
    saved_fig_path.parent.mkdir(parents=True, exist_ok=True)
    set_paper_style()
    colors = {"ring": "#8172B3", "clq": "#4C72B0", "ws": "#55A868", "opt": "#C44E52"}

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    max_distance = 0
    for graph_name in score_distance_by_graph:
        per_seed_scores = score_distance_by_graph.get(graph_name, [])
        if not per_seed_scores:
            continue
        max_len = max(len(scores) for scores in per_seed_scores)
        if max_plot_distance is not None:
            max_len = min(max_len, max_plot_distance)
        padded = np.full((len(per_seed_scores), max_len), np.nan, dtype=np.float32)
        for idx, scores in enumerate(per_seed_scores):
            padded[idx, : min(len(scores), max_len)] = scores[:max_len]
        distances = np.arange(1, padded.shape[1] + 1)
        mean_scores = np.nanmean(padded, axis=0)
        std_scores = np.nanstd(padded, axis=0)

        ax.plot(distances, mean_scores, color=colors[graph_name], label=GRAPH_LABELS[graph_name])
        ax.fill_between(
            distances,
            mean_scores - std_scores,
            mean_scores + std_scores,
            color=colors[graph_name],
            alpha=0.18,
            linewidth=0,
        )
        max_distance = max(max_distance, len(distances))

    ax.set_xlabel("Distance")
    ax.set_ylabel(ylabel)
    if max_distance:
        x_max = max_plot_distance if max_plot_distance is not None else max_distance
        ax.set_xlim(1, x_max)
        ticks = np.arange(1, x_max + 1)
        if x_max > 8:
            ticks = np.unique(np.r_[ticks[:: int(np.ceil(x_max / 8))], x_max])
        ax.set_xticks(ticks)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    ax.legend(frameon=False, loc=legend_loc)
    fig.tight_layout(pad=0.3)
    fig.savefig(saved_fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def nanmean_quiet(values, axis=0):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def save_summary_csv(rows: list[dict[str, float | str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "graph",
        "model_name",
        "seeds",
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
    print("\nXP metrics across the entire population")
    print("Graph | Seeds | Avg. Similarity | Self-SR | Cross-SR")
    print("-" * 67)
    for row in rows:
        print(
            f"{row['graph']:>5} | "
            f"{row['seeds']:<7} | "
            f"{row['avg_similarity_mean']:.4f} +/- {row['avg_similarity_std']:.4f} | "
            f"{row['self_sr_mean']:.4f} +/- {row['self_sr_std']:.4f} | "
            f"{row['cross_sr_mean']:.4f} +/- {row['cross_sr_std']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Ring, CLQ, WS, and OPT XP structure metrics.")
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=GRAPH_NAMES,
        default=list(GRAPH_NAMES),
        help="Graph structures to include.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to process. Missing seeds are skipped with a warning.",
    )
    parser.add_argument(
        "--ls-mode",
        choices=("compute", "load", "skip"),
        default="load",
        help="Whether to recompute LS, load cached LS, or skip LS.",
    )
    parser.add_argument(
        "--sr-mode",
        choices=("compute", "load", "skip"),
        default="load",
        help="Whether to load raw SR and cache it, load cached SR, or skip SR.",
    )
    parser.add_argument(
        "--max-plot-distance",
        type=int,
        default=DEFAULT_MAX_PLOT_DISTANCE,
        help="Maximum graph distance shown in the distance-line plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_paper_style()
    SCORE_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    avg_similarity_by_graph = {}
    avg_sr_by_graph = {}
    ls_distance_by_graph = {graph_name: [] for graph_name in args.graphs}
    sr_distance_by_graph = {graph_name: [] for graph_name in args.graphs}
    summary_rows = []

    for graph_name in args.graphs:
        model_name = model_name_for_graph(graph_name)
        distance_mat = graph_distance_matrix(graph_name, model_name)
        default_graph_distances = np.arange(1, int(np.nanmax(distance_mat)) + 1)
        seed_similarity_mats = []
        seed_sr_mats = []
        seed_avg_sims = []
        seed_self_srs = []
        seed_cross_srs = []
        processed_seeds = []
        consolidated_cache = None

        if args.ls_mode == "load" or args.sr_mode == "load":
            with contextlib.suppress(FileNotFoundError):
                consolidated_cache = load_seed_cache(graph_name, model_name)

        for seed in args.seeds:
            cached_scores = None
            similarity_mat = None
            avg_sim = None
            graph_distances = None
            ls_by_distance = None
            sr_by_distance = None
            sr_mat = None

            try:
                if args.ls_mode == "compute":
                    message_data = load_graph_messages(graph_name, model_name, seed)
                    similarity_mat, avg_sim = compute_similarity(message_data)
                    graph_distances, ls_by_distance = average_score_by_graph_distance(
                        similarity_mat, distance_mat
                    )
                    print(f"{model_name} seed{seed} computed avg language similarity: {avg_sim}")
                elif args.ls_mode == "load":
                    if consolidated_cache is not None and "similarity_mats" in consolidated_cache:
                        seed_idx = np.where(consolidated_cache["seeds"] == seed)[0]
                        if seed_idx.size == 0:
                            raise KeyError(f"Seed {seed} is missing from {seed_cache_path(graph_name, model_name)}")
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
                        cached_scores = load_cached_scores(graph_name, model_name, seed)
                        similarity_mat = np.asarray(cached_scores["similarity_mat"], dtype=np.float32)
                        avg_sim = float(cached_scores.get("avg_sim", summarize_similarity_mat(similarity_mat)))
                        graph_distances, ls_by_distance = average_score_by_graph_distance(
                            similarity_mat, distance_mat
                        )
                    print(f"{model_name} seed{seed} loaded avg language similarity: {avg_sim}")

                if args.sr_mode == "compute":
                    sr_mat = load_sr_mat(model_name, seed)
                    graph_distances, sr_by_distance = average_score_by_graph_distance(sr_mat, distance_mat)
                    print(f"{model_name} seed{seed} loaded raw SR matrix")
                elif args.sr_mode == "load":
                    if consolidated_cache is not None and "sr_mats" in consolidated_cache:
                        seed_idx = np.where(consolidated_cache["seeds"] == seed)[0]
                        if seed_idx.size == 0:
                            raise KeyError(f"Seed {seed} is missing from {seed_cache_path(graph_name, model_name)}")
                        cache_idx = int(seed_idx[0])
                        sr_mat = np.asarray(consolidated_cache["sr_mats"][cache_idx], dtype=np.float32)
                        graph_distances = np.asarray(consolidated_cache["graph_distances"], dtype=np.int64)
                        sr_by_distance = np.asarray(
                            consolidated_cache["sr_by_graph_distance"][cache_idx], dtype=np.float32
                        )
                    else:
                        if cached_scores is None:
                            cached_scores = load_cached_scores(graph_name, model_name, seed)
                        if "sr_mat" not in cached_scores:
                            raise KeyError(
                                f"Cached score file is missing sr_mat: {score_path(graph_name, model_name, seed)}"
                            )
                        sr_mat = np.asarray(cached_scores["sr_mat"], dtype=np.float32)
                        graph_distances, sr_by_distance = average_score_by_graph_distance(sr_mat, distance_mat)
                    print(f"{model_name} seed{seed} loaded cached SR matrix")
            except (FileNotFoundError, KeyError) as exc:
                print(f"Warning: skipping {model_name} seed{seed}: {exc}")
                continue

            if args.ls_mode != "skip":
                ls_distance_by_graph[graph_name].append(ls_by_distance)
                seed_similarity_mats.append(similarity_mat)
                seed_avg_sims.append(float(avg_sim))
            if args.sr_mode != "skip":
                sr_distance_by_graph[graph_name].append(sr_by_distance)
                seed_sr_mats.append(sr_mat)
                self_sr, cross_sr = summarize_sr_mat(sr_mat)
                seed_self_srs.append(self_sr)
                seed_cross_srs.append(cross_sr)
            if args.ls_mode != "skip" or args.sr_mode != "skip":
                processed_seeds.append(seed)

            if args.ls_mode == "compute" or args.sr_mode == "compute":
                save_seed_scores(
                    graph_name,
                    model_name,
                    seed,
                    similarity_mat,
                    avg_sim,
                    sr_mat,
                    graph_distances,
                    ls_by_distance,
                    sr_by_distance,
                )

        if not processed_seeds:
            print(f"Warning: no usable seeds found for {graph_name} ({model_name})")
            continue

        if args.ls_mode == "compute" or args.sr_mode == "compute":
            save_model_seed_cache(
                graph_name,
                model_name,
                processed_seeds,
                seed_similarity_mats,
                seed_sr_mats,
                ls_distance_by_graph[graph_name],
                sr_distance_by_graph[graph_name],
                seed_avg_sims,
                seed_self_srs,
                seed_cross_srs,
                distance_mat,
                default_graph_distances,
            )

        avg_scores = {
            "graph_pairs": np.asarray(graph_pairs(graph_name, model_name), dtype=np.int64),
            "graph_distance_mat": distance_mat,
            "graph_distances": default_graph_distances,
        }
        if seed_similarity_mats:
            avg_similarity_mat = nanmean_quiet(np.asarray(seed_similarity_mats), axis=0)
            avg_similarity_by_graph[graph_name] = avg_similarity_mat
            avg_scores["avg_similarity_mat"] = avg_similarity_mat
            avg_scores["avg_ls_by_graph_distance"] = nanmean_quiet(
                np.asarray(ls_distance_by_graph[graph_name]), axis=0
            )
            avg_scores["std_ls_by_graph_distance"] = np.nanstd(
                np.asarray(ls_distance_by_graph[graph_name]), axis=0
            )
        if seed_sr_mats:
            avg_sr_mat = nanmean_quiet(np.asarray(seed_sr_mats), axis=0)
            avg_sr_by_graph[graph_name] = avg_sr_mat
            avg_scores["avg_sr_mat"] = avg_sr_mat
            avg_scores["avg_sr_by_graph_distance"] = nanmean_quiet(
                np.asarray(sr_distance_by_graph[graph_name]), axis=0
            )
            avg_scores["std_sr_by_graph_distance"] = np.nanstd(
                np.asarray(sr_distance_by_graph[graph_name]), axis=0
            )

        avg_score_path(graph_name, model_name).parent.mkdir(parents=True, exist_ok=True)
        np.savez(avg_score_path(graph_name, model_name), **avg_scores)

        ls_mean, ls_std = mean_std(seed_avg_sims)
        self_sr_mean, self_sr_std = mean_std(seed_self_srs)
        cross_sr_mean, cross_sr_std = mean_std(seed_cross_srs)
        summary_rows.append(
            {
                "graph": GRAPH_LABELS[graph_name],
                "model_name": model_name,
                "seeds": " ".join(str(seed) for seed in processed_seeds),
                "avg_similarity_mean": ls_mean,
                "avg_similarity_std": ls_std,
                "self_sr_mean": self_sr_mean,
                "self_sr_std": self_sr_std,
                "cross_sr_mean": cross_sr_mean,
                "cross_sr_std": cross_sr_std,
            }
        )

    if avg_similarity_by_graph and avg_sr_by_graph:
        figure_prefix = f"{'_'.join(args.graphs)}_xp"
        plot_heatmap_grid(
            avg_sr_by_graph,
            avg_similarity_by_graph,
            FIG_DIR / f"{figure_prefix}_avg_sr_ls_heatmaps.png",
        )
    if avg_similarity_by_graph:
        figure_prefix = f"{'_'.join(args.graphs)}_xp"
        plot_score_by_graph_distance(
            ls_distance_by_graph,
            FIG_DIR / f"{figure_prefix}_ls_by_graph_distance.png",
            ylabel="Language Similarity",
            max_plot_distance=args.max_plot_distance,
        )
        plot_score_by_graph_distance(
            ls_distance_by_graph,
            FIG_DIR / f"{figure_prefix}_ls_by_graph_distance.pdf",
            ylabel="Language Similarity",
            max_plot_distance=args.max_plot_distance,
        )
    if avg_sr_by_graph:
        figure_prefix = f"{'_'.join(args.graphs)}_xp"
        plot_score_by_graph_distance(
            sr_distance_by_graph,
            FIG_DIR / f"{figure_prefix}_sr_by_graph_distance.png",
            ylabel="Success Rate",
            legend_loc="lower right",
            max_plot_distance=args.max_plot_distance,
        )
        plot_score_by_graph_distance(
            sr_distance_by_graph,
            FIG_DIR / f"{figure_prefix}_sr_by_graph_distance.pdf",
            ylabel="Success Rate",
            legend_loc="lower right",
            max_plot_distance=args.max_plot_distance,
        )

    save_summary_csv(summary_rows, SCORE_ROOT / SUMMARY_CSV)
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
