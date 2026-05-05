"""Create Table 1 style summaries for layout2 population experiments.

The script aggregates:
  * SR from eval_batched_sr.py outputs:
    logs/vary_n_pop/layout2/sr/<model>/<combination>/seed*/mode_test/normal/sr_eval.npz
  * LangSim from prepare_ls_traj.py outputs:
    logs/vary_n_pop/layout2/langsim/<model>/<pair>/<combination>/seed*/mode_test/normal/trajectory.pkl

Cross-SR is the mean over evaluated off-diagonal entries. Self-SR is the
mean over evaluated diagonal entries. LS follows the existing
get_ls_topsim_torchenv.py definition: sender messages are compared with
normalized edit-distance similarity across the same recorded episodes, and
the final LS is the lower-triangle mean excluding the diagonal.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

try:
    import editdistance as _editdistance
except ImportError:
    _editdistance = None


DEFAULT_MODELS = (
    "pop_ppo_2net_invisible",
    "sp_pop_ppo_2net_invisible",
    "pop_ppo_3net_invisible",
    "sp_pop_ppo_3net_invisible",
    "pop_ppo_100net_invisible",
    "sp_pop_ppo_100net_invisible",
)
DEFAULT_COMBINATION = "grid5_img3_ni2_nw4_ms30_comm_field100"


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def infer_num_networks(model_name: str) -> int:
    match = re.search(r"_(\d+)net_", model_name)
    if not match:
        raise ValueError(f"Cannot infer population size from model name: {model_name}")
    return int(match.group(1))


def training_label(model_name: str) -> str:
    if model_name.startswith("sp_"):
        return "XP+SP"
    return "XP"


def get_episode_length(log_s_messages: np.ndarray) -> int:
    end_idxs = np.where(log_s_messages[:, 0] == -1)[0]
    if end_idxs.size == 0:
        return int(log_s_messages.shape[0])
    return int(end_idxs[0])


def extract_sender_messages(log_data: dict[str, Any]) -> list[np.ndarray]:
    messages = []
    for episode_data in log_data.values():
        log_s_messages = episode_data["log_s_messages"]
        ep_len = get_episode_length(log_s_messages)
        messages.append(np.asarray(log_s_messages[:ep_len, 0], dtype=np.int64).flatten())
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
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def normalized_message_similarity(message_a: np.ndarray, message_b: np.ndarray) -> float:
    a = [int(x) for x in message_a if int(x) != -1]
    b = [int(x) for x in message_b if int(x) != -1]
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - (edit_distance(a, b) / denom)


def find_existing_path(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def load_seed_ls(
    langsim_root: Path,
    precomputed_ls_root: Path | None,
    model_name: str,
    num_networks: int,
    combination: str,
    seed: int,
    mode: str,
    condition: str,
) -> float:
    if precomputed_ls_root is not None:
        sim_path = precomputed_ls_root / model_name / f"{combination}_seed{seed}" / "sim_scores.npz"
        if sim_path.exists():
            with np.load(sim_path) as data:
                return float(data["avg_sim"])

    sender_messages: list[list[np.ndarray]] = []
    for sender_id in range(num_networks):
        candidates = (
            langsim_root
            / model_name
            / f"{sender_id}-0"
            / combination
            / f"seed{seed}"
            / f"mode_{mode}"
            / condition
            / "trajectory.pkl",
            langsim_root
            / model_name
            / f"{sender_id}-{sender_id}"
            / combination
            / f"seed{seed}"
            / f"mode_{mode}"
            / condition
            / "trajectory.pkl",
        )
        trajectory_path = find_existing_path(candidates)
        if trajectory_path is None:
            raise FileNotFoundError(
                f"Missing LangSim trajectory for {model_name}, seed {seed}, sender {sender_id}. "
                f"Tried {candidates[0]} and {candidates[1]}"
            )
        with trajectory_path.open("rb") as f:
            sender_messages.append(extract_sender_messages(pickle.load(f)))

    n_samples = min(len(messages) for messages in sender_messages)
    if n_samples == 0:
        raise ValueError(f"No LangSim episodes found for {model_name}, seed {seed}")

    similarity_mat = np.zeros((num_networks, num_networks), dtype=np.float64)
    for first_id in range(num_networks):
        for second_id in range(num_networks):
            sims = [
                normalized_message_similarity(
                    sender_messages[first_id][episode_idx],
                    sender_messages[second_id][episode_idx],
                )
                for episode_idx in range(n_samples)
            ]
            similarity_mat[first_id, second_id] = float(np.mean(sims))

    mask = np.tril(np.ones_like(similarity_mat, dtype=bool), k=-1)
    if not mask.any():
        return float("nan")
    return float(np.mean(similarity_mat[mask]))


def load_seed_sr(
    sr_root: Path,
    model_name: str,
    combination: str,
    seed: int,
    mode: str,
    condition: str,
) -> tuple[float, float]:
    sr_path = (
        sr_root
        / model_name
        / combination
        / f"seed{seed}"
        / f"mode_{mode}"
        / condition
        / "sr_eval.npz"
    )
    if not sr_path.exists():
        raise FileNotFoundError(f"Missing SR file: {sr_path}")

    with np.load(sr_path) as data:
        sr_mat = np.asarray(data["sr_mat"], dtype=np.float64)
        if "evaluated_mask" in data.files:
            evaluated_mask = np.asarray(data["evaluated_mask"]).astype(bool)
        else:
            evaluated_mask = ~np.isnan(sr_mat)

    valid = evaluated_mask & np.isfinite(sr_mat)
    off_diag = ~np.eye(sr_mat.shape[0], dtype=bool)
    diag = np.eye(sr_mat.shape[0], dtype=bool)
    cross_values = sr_mat[valid & off_diag]
    self_values = sr_mat[valid & diag]

    cross_sr = float(np.mean(cross_values)) if cross_values.size else float("nan")
    self_sr = float(np.mean(self_values)) if self_values.size else float("nan")
    return cross_sr, self_sr


def summarize(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def format_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ("Training", "N_pop", "LS", "Cross-SR", "Self-SR")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["training"]),
                    str(row["n_pop"]),
                    str(row["ls"]),
                    str(row["cross_sr"]),
                    str(row["self_sr"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Training & $N_{\mathrm{pop}}$ & LS & Cross-SR & Self-SR \\",
        r"\midrule",
    ]
    for row in rows:
        ls_cell = row["ls"].replace("+/-", r"$\pm$")
        cross_sr_cell = row["cross_sr"].replace("+/-", r"$\pm$")
        self_sr_cell = row["self_sr"].replace("+/-", r"$\pm$")
        lines.append(
            f"{row['training']} & {row['n_pop']} & "
            f"{ls_cell} & "
            f"{cross_sr_cell} & "
            f"{self_sr_cell} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_this_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--combination", default=DEFAULT_COMBINATION)
    parser.add_argument("--mode", default="test")
    parser.add_argument("--condition", default="normal")
    parser.add_argument("--sr-root", type=Path, default=repo_root / "logs/vary_n_pop/layout2/sr")
    parser.add_argument("--langsim-root", type=Path, default=repo_root / "logs/vary_n_pop/layout2/langsim")
    parser.add_argument(
        "--precomputed-ls-root",
        type=Path,
        default=None,
        help="Optional root containing <model>/<combination>_seed<seed>/sim_scores.npz files.",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "analysis/table1_layout2")
    parser.add_argument("--strict", action="store_true", help="Fail if any model/seed is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        num_networks = infer_num_networks(model_name)
        seed_ls: list[float] = []
        seed_cross_sr: list[float] = []
        seed_self_sr: list[float] = []

        for seed in args.seeds:
            try:
                seed_ls.append(
                    load_seed_ls(
                        args.langsim_root,
                        args.precomputed_ls_root,
                        model_name,
                        num_networks,
                        args.combination,
                        seed,
                        args.mode,
                        args.condition,
                    )
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                if args.strict:
                    raise
                print(f"[warn] LS skipped for {model_name} seed {seed}: {exc}")

            try:
                cross_sr, self_sr = load_seed_sr(
                    args.sr_root,
                    model_name,
                    args.combination,
                    seed,
                    args.mode,
                    args.condition,
                )
                seed_cross_sr.append(cross_sr)
                seed_self_sr.append(self_sr)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                if args.strict:
                    raise
                print(f"[warn] SR skipped for {model_name} seed {seed}: {exc}")

        ls_mean, ls_std = summarize(seed_ls)
        cross_mean, cross_std = summarize(seed_cross_sr)
        self_mean, self_std = summarize(seed_self_sr)
        rows.append(
            {
                "model": model_name,
                "training": training_label(model_name),
                "n_pop": num_networks,
                "ls": format_mean_std(ls_mean, ls_std),
                "cross_sr": format_mean_std(cross_mean, cross_std),
                "self_sr": format_mean_std(self_mean, self_std),
            }
        )
        raw_rows.append(
            {
                "model": model_name,
                "training": training_label(model_name),
                "n_pop": num_networks,
                "ls_mean": ls_mean,
                "ls_std": ls_std,
                "cross_sr_mean": cross_mean,
                "cross_sr_std": cross_std,
                "self_sr_mean": self_mean,
                "self_sr_std": self_std,
                "n_ls_seeds": len(seed_ls),
                "n_sr_seeds": len(seed_cross_sr),
            }
        )

    md = markdown_table(rows)
    (args.output_dir / "table1_layout2.md").write_text(md)
    (args.output_dir / "table1_layout2.tex").write_text(latex_table(rows))

    with (args.output_dir / "table1_layout2.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    print(md)
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
