from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ENTITY = "maytusp"
PROJECT = "scoreg_layout2"

RUN_GROUPS = {
    3: ["d896m9fn", "wvqlpkx0", "kg3wn7ga"],
    15: ["q73a8tt5", "h2gfojqi", "u14mjn8f"],
    100: ["a9vqfs85", "wv0eodyi", "ueri8yzx"],
}

RETURN_KEYS = ("charts/episodic_return/", "charts/episodic_return")
GLOBAL_STEP_KEY = "global_step"
METRICS = {
    "return": {
        "title": "Return",
        "ylabel": "Episodic return",
        "filename": "return",
    },
    "action_entropy": {
        "title": "Action Entropy",
        "ylabel": "Action entropy",
        "filename": "action_entropy",
    },
    "message_entropy": {
        "title": "Message Entropy",
        "ylabel": "Message entropy",
        "filename": "message_entropy",
    },
}

PLOT_COLORS = {
    3: "#3B6EA8",
    15: "#D9902F",
    100: "#4D9A57",
}

PLOT_MARKERS = {
    3: "o",
    15: "s",
    100: "^",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "analysis" / ".wandb_cache" / PROJECT
DEFAULT_FIG_DIR = REPO_ROOT / "analysis" / "plots" / "wandb_struct_training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scoreg_layout2 return/action/message entropy from W&B."
    )
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only read local cache files; do not connect to W&B.",
    )
    parser.add_argument(
        "--cache-global-step-only",
        action="store_true",
        help="Fetch/cache only the W&B global_step series for each run, then exit.",
    )
    parser.add_argument("--page-size", type=int, default=10000)
    parser.add_argument("--grid-points", type=int, default=600)
    parser.add_argument(
        "--max-step",
        type=float,
        default=2.5e9,
        help="Maximum global step to show. Default is 1e9, i.e. 10 x 100M.",
    )
    parser.add_argument(
        "--no-seed-lines",
        action="store_true",
        help="Hide the faint individual-seed traces.",
    )
    return parser.parse_args()


def label_for(num_agents: int) -> str:
    return f"{num_agents} agents"


def metric_cache_path(cache_dir: Path, run_id: str, metric: str) -> Path:
    return cache_dir / f"{run_id}_{metric}.csv"


def safe_key_name(key: str) -> str:
    return key.replace("/", "__")


def scalar_cache_path(cache_dir: Path, run_id: str, key: str) -> Path:
    return cache_dir / "scalars" / run_id / f"{safe_key_name(key)}.csv"


def manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def read_cached_metric(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps = []
    values = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row["global_step"]))
            values.append(float(row["value"]))
    return np.asarray(steps, dtype=float), np.asarray(values, dtype=float)


def write_cached_metric(path: Path, steps: np.ndarray, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["global_step", "value"])
        writer.writeheader()
        for step, value in zip(steps, values):
            writer.writerow({"global_step": step, "value": value})


def read_cached_rows(path: Path) -> list[tuple[float, float]]:
    steps, values = read_cached_metric(path)
    return list(zip(steps.tolist(), values.tolist()))


def write_cache_manifest(args: argparse.Namespace) -> None:
    payload = {
        "entity": args.entity,
        "project": args.project,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_groups": RUN_GROUPS,
        "metrics": list(METRICS),
        "note": "Metric CSVs are averaged plotted series. scalars/* contains raw W&B scalar streams used to build them.",
    }
    path = manifest_path(args.cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def fetch_scalar_history(run, key: str, page_size: int) -> list[tuple[float, float]]:
    rows = []
    try:
        history = run.scan_history(keys=["_step", key], page_size=page_size)
        for row in history:
            if "_step" not in row or key not in row or row[key] is None:
                continue
            rows.append((float(row["_step"]), float(row[key])))
    except Exception as exc:
        warnings.warn(f"Could not fetch {run.id}:{key}: {exc}", RuntimeWarning)
    return rows


def load_scalar_history(
    run,
    args: argparse.Namespace,
    run_id: str,
    key: str,
) -> list[tuple[float, float]]:
    path = scalar_cache_path(args.cache_dir, run_id, key)
    if path.exists() and not args.refresh_cache:
        rows = read_cached_rows(path)
        print(f"  cache hit: {run_id}:{key} ({len(rows)} points)")
        return rows

    if args.plot_only:
        raise FileNotFoundError(
            f"Missing scalar cache for {run_id}:{key}. "
            f"Run once without --plot-only to fetch it from W&B."
        )

    print(f"  fetching: {run_id}:{key}")
    rows = fetch_scalar_history(run, key, args.page_size)
    steps, values = aggregate_by_step(rows)
    write_cached_metric(path, steps, values)
    print(f"  cached: {path} ({steps.size} points)")
    return list(zip(steps.tolist(), values.tolist()))


def aggregate_by_step(rows: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    by_step = defaultdict(list)
    for step, value in rows:
        if np.isfinite(step) and np.isfinite(value):
            by_step[step].append(value)

    steps = np.asarray(sorted(by_step), dtype=float)
    values = np.asarray([np.mean(by_step[step]) for step in steps], dtype=float)
    return steps, values


def fetch_metric(
    run,
    args: argparse.Namespace,
    run_id: str,
    metric: str,
    num_agents: int,
) -> tuple[np.ndarray, np.ndarray]:
    if metric == "return":
        for key in RETURN_KEYS:
            rows = load_scalar_history(run, args, run_id, key)
            if rows:
                return aggregate_by_step(rows)
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    if metric == "action_entropy":
        suffix = "action_entropy"
    elif metric == "message_entropy":
        suffix = "message_entropy"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    rows = []
    for agent_id in range(num_agents):
        key = f"agent{agent_id}/losses/{suffix}"
        rows.extend(load_scalar_history(run, args, run_id, key))
    return aggregate_by_step(rows)


def load_metric(api, args: argparse.Namespace, run_id: str, metric: str, num_agents: int):
    path = metric_cache_path(args.cache_dir, run_id, metric)
    if path.exists() and not args.refresh_cache:
        steps, values = read_cached_metric(path)
        print(f"metric cache hit: {path} ({steps.size} points)")
        return steps, values

    if args.plot_only:
        raise FileNotFoundError(
            f"Missing metric cache: {path}. Run once without --plot-only to fetch it from W&B."
        )

    run_path = f"{args.entity}/{args.project}/{run_id}"
    print(f"metric cache miss: fetching {metric} for {run_path}")
    run = api.run(run_path)
    steps, values = fetch_metric(run, args, run_id, metric, num_agents)
    write_cached_metric(path, steps, values)
    write_cache_manifest(args)
    print(f"metric cached: {path} ({steps.size} points)")
    return steps, values


def load_global_step_series(api, args: argparse.Namespace, run_id: str):
    path = metric_cache_path(args.cache_dir, run_id, GLOBAL_STEP_KEY)
    if path.exists() and not args.refresh_cache:
        steps, values = read_cached_metric(path)
        print(f"global_step cache hit: {path} ({steps.size} points)")
        return steps, values

    if args.plot_only:
        raise FileNotFoundError(
            f"Missing global_step cache: {path}. Run once without --plot-only to fetch it from W&B."
        )

    run_path = f"{args.entity}/{args.project}/{run_id}"
    print(f"global_step cache miss: fetching {run_path}")
    run = api.run(run_path)
    rows = load_scalar_history(run, args, run_id, GLOBAL_STEP_KEY)
    steps, values = aggregate_by_step(rows)
    write_cached_metric(path, steps, values)
    write_cache_manifest(args)
    print(f"global_step cached: {path} ({steps.size} points)")
    return steps, values


def cache_all_global_steps(api, args: argparse.Namespace) -> None:
    for run_ids in RUN_GROUPS.values():
        for run_id in run_ids:
            load_global_step_series(api, args, run_id)


def align_metric_to_global_steps(
    metric_join_steps: np.ndarray,
    global_step_join_steps: np.ndarray,
    global_step_values: np.ndarray,
) -> np.ndarray:
    """Use the first CSV column as the join key and global_step.value as x."""
    mask = np.isfinite(global_step_join_steps) & np.isfinite(global_step_values)
    global_step_join_steps = global_step_join_steps[mask]
    global_step_values = global_step_values[mask]
    if global_step_join_steps.size < 2:
        return metric_join_steps

    order = np.argsort(global_step_join_steps)
    global_step_join_steps = global_step_join_steps[order]
    global_step_values = global_step_values[order]
    unique_join_steps, unique_idx = np.unique(global_step_join_steps, return_index=True)
    unique_global_steps = global_step_values[unique_idx]

    in_range = (metric_join_steps >= unique_join_steps[0]) & (metric_join_steps <= unique_join_steps[-1])
    mapped_steps = np.full_like(metric_join_steps, np.nan, dtype=float)
    mapped_steps[in_range] = np.interp(
        metric_join_steps[in_range],
        unique_join_steps,
        unique_global_steps,
    )
    return mapped_steps


def clip_series(
    steps: np.ndarray,
    values: np.ndarray,
    max_step: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(steps) & np.isfinite(values)
    if max_step is not None:
        mask &= steps <= max_step
    steps = steps[mask]
    values = values[mask]
    if steps.size == 0:
        return steps, values

    order = np.argsort(steps)
    steps = steps[order]
    values = values[order]
    unique_steps, inverse = np.unique(steps, return_inverse=True)
    if unique_steps.size == steps.size:
        return steps, values

    unique_values = np.zeros_like(unique_steps, dtype=float)
    for idx in range(unique_steps.size):
        unique_values[idx] = np.mean(values[inverse == idx])
    return unique_steps, unique_values


def interpolate_group(
    series: list[tuple[np.ndarray, np.ndarray]],
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_series = [(x, y) for x, y in series if x.size >= 2]
    if not finite_series:
        return np.asarray([]), np.asarray([]), np.asarray([])

    min_step = min(float(x[0]) for x, _ in finite_series)
    max_step = max(float(x[-1]) for x, _ in finite_series)
    grid = np.linspace(min_step, max_step, grid_points)

    interp = np.full((len(finite_series), grid_points), np.nan, dtype=float)
    for idx, (steps, values) in enumerate(finite_series):
        in_range = (grid >= steps[0]) & (grid <= steps[-1])
        interp[idx, in_range] = np.interp(grid[in_range], steps, values)

    mean = np.nanmean(interp, axis=0)
    valid_counts = np.sum(np.isfinite(interp), axis=0)
    std = np.nanstd(interp, axis=0)
    stderr = np.divide(
        std,
        np.sqrt(valid_counts),
        out=np.full_like(std, np.nan),
        where=valid_counts > 0,
    )
    return grid, mean, stderr


def set_plot_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 16,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "round",
        }
    )


def style_axis(ax) -> None:
    ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.8)
    ax.grid(True, axis="x", color="#EEF1F4", linewidth=0.6)
    ax.tick_params(axis="both", length=0, pad=6)
    sns.despine(ax=ax)


def plot_metric(
    metric: str,
    all_data: dict,
    args: argparse.Namespace,
    ax=None,
    show_xlabel: bool = True,
    show_legend: bool = True,
) -> None:
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(6.8, 4.2))

    for num_agents, run_series in all_data[metric].items():
        color = PLOT_COLORS[num_agents]
        marker = PLOT_MARKERS[num_agents]
        label = label_for(num_agents)

        if not args.no_seed_lines:
            for steps, values in run_series:
                ax.plot(steps / 1e8, values, color=color, alpha=0.16, linewidth=0.9)

        grid, mean, stderr = interpolate_group(run_series, args.grid_points)
        if grid.size == 0:
            warnings.warn(f"No plottable data for {metric} / {label}", RuntimeWarning)
            continue

        marker_every = max(1, len(grid) // 10)
        ax.plot(
            grid / 1e8,
            mean,
            color=color,
            linewidth=2.0,
            marker=marker,
            markevery=marker_every,
            markersize=4.0,
            markeredgewidth=0,
            label=label,
        )
        ax.fill_between(
            grid / 1e8,
            mean - stderr,
            mean + stderr,
            color=color,
            alpha=0.08,
            linewidth=0,
        )

    # ax.set_title(METRICS[metric]["title"])
    ax.set_xlabel("Environment steps (100M)" if show_xlabel else "")
    ax.set_ylabel(METRICS[metric]["ylabel"])
    ax.set_xlim(left=0, right=args.max_step / 1e8 if args.max_step else None)
    style_axis(ax)
    if show_legend:
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#D7DCE2",
            framealpha=0.95,
            loc="best",
            handlelength=2.3,
            borderpad=0.6,
            labelspacing=0.5,
        )

    if own_fig:
        plt.tight_layout()
        out_path = args.fig_dir / f"{METRICS[metric]['filename']}_vs_steps.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    print(f"Using cache dir: {args.cache_dir}")
    if args.plot_only:
        print("Plot-only mode: W&B will not be queried.")
        api = None
    else:
        import wandb

        api = wandb.Api()

    if args.cache_global_step_only:
        if args.plot_only:
            raise ValueError("--cache-global-step-only cannot be combined with --plot-only")
        cache_all_global_steps(api, args)
        return

    all_data = {metric: {} for metric in METRICS}

    for num_agents, run_ids in RUN_GROUPS.items():
        for metric in METRICS:
            all_data[metric][num_agents] = []

        for run_id in run_ids:
            global_step_join_steps, global_step_values = load_global_step_series(api, args, run_id)
            for metric in METRICS:
                metric_join_steps, metric_values = load_metric(api, args, run_id, metric, num_agents)
                x_steps = align_metric_to_global_steps(
                    metric_join_steps,
                    global_step_join_steps,
                    global_step_values,
                )
                steps, values = clip_series(x_steps, metric_values, args.max_step)
                if steps.size == 0:
                    warnings.warn(
                        f"No {metric} points found for {label_for(num_agents)} run {run_id}",
                        RuntimeWarning,
                    )
                all_data[metric][num_agents].append((steps, values))

    for metric in METRICS:
        plot_metric(metric, all_data, args)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.6), sharex=True)
    for idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        plot_metric(
            metric,
            all_data,
            args,
            ax=ax,
            show_xlabel=idx == len(METRICS) - 1,
            show_legend=False,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(labels),
        frameon=False,
        handlelength=2.3,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0)
    combined_path = args.fig_dir / "wandb_struct_training_metrics.pdf"
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)
    print(f"Saved {combined_path}")


if __name__ == "__main__":
    main()
