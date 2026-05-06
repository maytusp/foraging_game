# new code updated 28 march 2026

import pickle
import argparse
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import editdistance
from language_analysis import Disent, TopographicSimilarity
import os
from collections import Counter

N_WORDS = 4
TOPSIM_PAD_TOKEN = N_WORDS
PLOT_COLORS = {
    "XP": "#3B5BA9",
    "XP+SP": "#C44E52",
}


def set_paper_style():
    sns.set_theme(
        context="paper",
        style="ticks",
        font="DejaVu Sans",
        rc={
            "axes.labelsize": 20,
            "axes.linewidth": 1.1,
            "font.size": 18,
            "legend.fontsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "lines.linewidth": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )

# Load the .pkl file
def load_trajectory(file_path):
    with open(file_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data

def get_episode_length(log_s_messages: np.ndarray) -> int:
    """
    log_s_messages: (T, num_agents) NumPy array with -1 padding after the episode.
    Returns the number of valid time steps in the episode.
    """
    # indices where sentinel -1 appears in agent 0's messages
    end_idxs = np.where(log_s_messages[:, 0] == -1)[0]
    if end_idxs.size == 0:
        # no -1 at all: whole sequence is valid
        return log_s_messages.shape[0]
    else:
        # first -1 marks the end, so its index is the length
        return int(end_idxs[0])


def get_mode_episode_length(log_data) -> int:
    lengths = []
    for episode_id, data in log_data.items():
        log_s_messages = data["log_s_messages"]
        ep_len = get_episode_length(log_s_messages)
        lengths.append(ep_len)

    counts = Counter(lengths)
    mode_length = counts.most_common(1)[0][0]
    return mode_length


def get_mode_target_episode_length(log_data) -> int:
    lengths = []
    for episode_id, data in log_data.items():
        if data["who_see_target"] == 0:
            lengths.append(get_episode_length(data["log_s_messages"]))
    if not lengths:
        raise ValueError("No target-visible episodes found for TopSim/PosDis.")
    counts = Counter(lengths)
    return counts.most_common(1)[0][0]


def pad_message(message, target_length, pad_token):
    if len(message) >= target_length:
        return message[:target_length]
    return np.pad(
        message,
        (0, target_length - len(message)),
        mode="constant",
        constant_values=pad_token,
    )

def extract_data_for_ls(log_data):
    message_data = {"agent0": [], "agent1": []}

    for episode_id, data in log_data.items():
        log_s_messages = data["log_s_messages"]
        who_see_target = data["who_see_target"]
        ep_len = get_episode_length(log_s_messages)
        messages = log_s_messages[:ep_len, 0].flatten()
        message_data["agent0"].append(messages)
        
    return message_data

def extract_data_for_topsim(log_data, pad_token=TOPSIM_PAD_TOKEN):
    mode_length = get_mode_target_episode_length(log_data)
    message_data = {"agent0": [], "agent1": []}
    attribute_data = []

    for episode_id, data in log_data.items():
        log_s_messages = data["log_s_messages"]
        who_see_target = data["who_see_target"]
        target_score = data["log_target_food_dict"]["score"]
        target_loc = data["log_target_food_dict"]["location"]       # (2,)
        ep_len = get_episode_length(log_s_messages)

        if ep_len <= mode_length+2 and who_see_target == 0:
            messages = log_s_messages[:ep_len, 0].flatten()
            messages = pad_message(messages, mode_length+2, pad_token)
            message_data["agent0"].append(messages)
            extract_attribute = [target_score, target_loc[0], target_loc[1]]
            attribute_data.append(extract_attribute)
            # else:
            #     extract_attribute = [distractor_score, distractor_loc[0], distractor_loc[1]]
            #     attribute_data.append(extract_attribute)
            
    return message_data, attribute_data


def get_comp_scores(message_data, attribute_data, num_networks):
    '''
    Input: (messages, attributes)
    Output: topsim, posdis
    
    '''
    sender_list = [i for i in range(num_networks)]
    data = []
    n_samples = 1000000
    extracted_message = []
    extracted_attribute = []
    receiver = 0
    avg_topsim = 0
    avg_posdis = 0
    topsim_list = []
    posdis_list = []
    max_eval_episodes=1000

    def get_sender_pair(sender):
        candidate_pairs = [f"{sender}-{receiver}", f"{sender}-0"]
        for pair in candidate_pairs:
            if pair in message_data:
                return pair
        raise KeyError(f"No loaded message data for sender {sender}. Tried: {candidate_pairs}")

    if num_networks > 2:
        for sender in sender_list:
            pair = get_sender_pair(sender)
            extracted_message.append(np.array(message_data[pair]["agent0"]))
            extracted_attribute.append(attribute_data[pair])
            n_samples = min(extracted_message[sender].shape[0], n_samples)
    else:
        for sender in sender_list:
            receiver_map = {0:1, 1:0}
            receiver = receiver_map[sender]
            # in case of XP n_pop=2, agent cannot successfully play with itself, we need to gather info when it plays with its partner
            pair = get_sender_pair(sender)
            extracted_message.append(np.array(message_data[pair]["agent0"]))
            extracted_attribute.append(attribute_data[pair])
            n_samples = min(extracted_message[sender].shape[0], n_samples)


    for agent_id in range(len(sender_list)):
        messages = np.array(extracted_message[sender_list[agent_id]])
        # print(f"message shape {messages.shape}")
        attributes = np.array(extracted_attribute[sender_list[agent_id]])
        topsim = TopographicSimilarity.compute_topsim(attributes[:max_eval_episodes], messages[:max_eval_episodes, :])
        torch_attributes = torch.tensor(attributes[:max_eval_episodes])
        torch_messages = torch.tensor(messages[:max_eval_episodes, :])
        posdis = Disent.posdis(torch_attributes, torch_messages)
        avg_topsim += topsim
        avg_posdis += posdis
        topsim_list.append(topsim)
        posdis_list.append(posdis)
        # print(f"agent_id {agent_id} has topsim {topsim}, posdis {posdis}")
    avg_topsim /= num_networks
    avg_posdis /= num_networks

    return avg_topsim, avg_posdis, topsim_list, posdis_list


def get_similarity(message_data, num_networks):
    sender_list = [i for i in range(num_networks)]
    similarity_mat = np.zeros((num_networks, num_networks))
    n_samples = 1000000
    extracted_message = []
    receiver = 0

    for sender in sender_list:
        msgs = message_data[f"{sender}-{receiver}"]["agent0"]   # keep as list
        extracted_message.append(msgs)
        n_samples = min(len(msgs), n_samples)

    # print("langsim n_samples =", n_samples)

    if n_samples == 0:
        raise ValueError("No langsim samples found.")

    for first_agent_id in range(len(sender_list)):
        for second_agent_id in range(len(sender_list)):
            for i in range(n_samples):
                m1 = extracted_message[sender_list[first_agent_id]][i]
                m2 = extracted_message[sender_list[second_agent_id]][i]

                m1 = [x for x in m1 if x != -1]
                m2 = [x for x in m2 if x != -1]

                denom = max(len(m1), len(m2))
                if denom == 0:
                    sim = 1.0
                else:
                    dist = editdistance.eval(m1, m2) / denom
                    sim = 1 - dist

                similarity_mat[first_agent_id, second_agent_id] += sim

    similarity_mat = similarity_mat / n_samples
    mask = np.ones_like(similarity_mat)
    mask[np.triu_indices_from(mask, k=0)] = 0
    avg_sim = np.sum(similarity_mat * mask) / np.sum(mask)

    return similarity_mat, avg_sim



def plot_heatmap(similarity_mat, saved_fig_path, cbar, vmin, vmax):
    set_paper_style()
    os.makedirs(os.path.dirname(saved_fig_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    with sns.axes_style("white"):
        sns.heatmap(
            similarity_mat,
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


def load_score(filename):
    scores = {}
    with open(filename, "r") as f:
        for line in f:
            x = line.strip().split(": ")
            if "Reward" in x[0]:
                continue
            scores[x[0].strip()] = float(x[1].strip())
    return scores


def load_sr_metrics(sr_root, model_name, combination_name, seed, mode, condition, num_networks):
    sr_path = os.path.join(
        sr_root,
        model_name,
        combination_name,
        f"seed{seed}",
        f"mode_{mode}",
        condition,
        "sr_eval.npz",
    )
    if not os.path.exists(sr_path):
        raise FileNotFoundError(f"Missing SR file: {sr_path}")

    with np.load(sr_path) as data:
        sr_mat = np.asarray(data["sr_mat"], dtype=float)
        if "evaluated_mask" in data.files:
            evaluated_mask = np.asarray(data["evaluated_mask"]).astype(bool)
        else:
            evaluated_mask = ~np.isnan(sr_mat)

    if sr_mat.shape != (num_networks, num_networks):
        raise ValueError(
            f"Expected SR matrix shape {(num_networks, num_networks)}, got {sr_mat.shape}: {sr_path}"
        )

    valid_mask = evaluated_mask & np.isfinite(sr_mat)
    self_mask = np.eye(num_networks, dtype=bool)
    cross_mask = ~self_mask

    self_values = sr_mat[valid_mask & self_mask]
    cross_values = sr_mat[valid_mask & cross_mask]

    self_sr = float(np.mean(self_values)) if self_values.size else np.nan
    cross_sr = float(np.mean(cross_values)) if cross_values.size else np.nan
    ic = float(self_sr / cross_sr) if np.isfinite(self_sr) and np.isfinite(cross_sr) and cross_sr != 0 else np.nan

    return sr_mat, self_sr, cross_sr, ic


def save_metric_scores(
    saved_score_path,
    similarity_mat,
    avg_sim,
    avg_topsim,
    avg_posdis,
    topsim_list,
    posdis_list,
    sr_mat,
    self_sr,
    cross_sr,
    ic,
):
    np.savez(
        saved_score_path,
        similarity_mat=similarity_mat,
        avg_sim=avg_sim,
        avg_topsim=avg_topsim,
        avg_posdis=avg_posdis,
        topsim_list=topsim_list,
        posdis_list=posdis_list,
        sr_mat=sr_mat,
        self_sr=self_sr,
        cross_sr=cross_sr,
        ic=ic,
    )


def load_metric_scores(saved_score_path):
    required_keys = {
        "similarity_mat",
        "avg_sim",
        "avg_topsim",
        "avg_posdis",
        "topsim_list",
        "posdis_list",
        "sr_mat",
        "self_sr",
        "cross_sr",
        "ic",
    }
    with np.load(saved_score_path, allow_pickle=True) as scores:
        missing_keys = required_keys.difference(scores.files)
        if missing_keys:
            raise KeyError(f"Saved metrics missing keys {sorted(missing_keys)}: {saved_score_path}")
        return {
            "similarity_mat": scores["similarity_mat"],
            "avg_sim": float(scores["avg_sim"]),
            "avg_topsim": float(scores["avg_topsim"]),
            "avg_posdis": float(scores["avg_posdis"]),
            "topsim_list": scores["topsim_list"].tolist(),
            "posdis_list": scores["posdis_list"].tolist(),
            "sr_mat": scores["sr_mat"],
            "self_sr": float(scores["self_sr"]),
            "cross_sr": float(scores["cross_sr"]),
            "ic": float(scores["ic"]),
        }


def condition_from_model_name(model_name):
    return "XP+SP" if model_name.startswith("sp_") else "XP"


def init_population_stats():
    return {
        "population_size": [],
        "ls_mean": [],
        "ls_std": [],
        "topsim_mean": [],
        "topsim_std": [],
        "posdis_mean": [],
        "posdis_std": [],
        "ic_mean": [],
        "ic_std": [],
    }


def append_population_stats(population_stats, num_networks, seed_ls, seed_topsim, seed_posdis, seed_ic):
    population_stats["population_size"].append(num_networks)
    population_stats["ls_mean"].append(np.mean(seed_ls))
    population_stats["ls_std"].append(np.std(seed_ls))
    population_stats["topsim_mean"].append(np.mean(seed_topsim))
    population_stats["topsim_std"].append(np.std(seed_topsim))
    population_stats["posdis_mean"].append(np.mean(seed_posdis))
    population_stats["posdis_std"].append(np.std(seed_posdis))
    population_stats["ic_mean"].append(np.mean(seed_ic))
    population_stats["ic_std"].append(np.std(seed_ic))


def population_plot_positions(population_sizes):
    positions = np.asarray(population_sizes, dtype=float).copy()
    positions[positions == 2] = 1.0
    positions[positions == 3] = 5.0
    return positions


def plot_population_metric(all_population_stats, metric, ylabel, filename, saved_fig_dir, loc="best"):
    set_paper_style()
    os.makedirs(saved_fig_dir, exist_ok=True)
    markers = {"XP": "o", "XP+SP": "s"}
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    plotted_population_sizes = set()
    for condition in ("XP", "XP+SP"):
        population_stats = all_population_stats[condition]
        population_sizes = np.array(population_stats["population_size"], dtype=float)
        means = np.array(population_stats[f"{metric}_mean"], dtype=float)
        stds = np.array(population_stats[f"{metric}_std"], dtype=float)
        if population_sizes.size == 0:
            continue

        order = np.argsort(population_sizes)
        population_sizes = population_sizes[order]
        means = means[order]
        stds = stds[order]
        plotted_population_sizes.update(population_sizes.tolist())
        plot_x = population_plot_positions(population_sizes)

        color = PLOT_COLORS[condition]
        ax.plot(
            plot_x,
            means,
            color=color,
            marker=markers[condition],
            markersize=5.0,
            markeredgewidth=0.8,
            label=condition,
        )
        ax.fill_between(
            plot_x,
            means - stds,
            means + stds,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ticks = sorted(plotted_population_sizes)
    ax.set_xlabel("Population Size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(population_plot_positions(ticks))
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    ax.legend(loc=loc, frameon=False)

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(saved_fig_dir, filename), dpi=300, bbox_inches="tight")
    pdf_filename = os.path.splitext(filename)[0] + ".pdf"
    fig.savefig(os.path.join(saved_fig_dir, pdf_filename), bbox_inches="tight")
    plt.close()


def plot_population_metrics(all_population_stats, saved_fig_dir):
    os.makedirs(saved_fig_dir, exist_ok=True)

    plot_population_metric(
        all_population_stats,
        "ls",
        "Language Similarity",
        "ls_vs_pop.png",
        saved_fig_dir,
        loc="lower right",
    )
    plot_population_metric(
        all_population_stats,
        "topsim",
        "Topographic Similarity",
        "topsim_vs_pop.png",
        saved_fig_dir,
        loc="upper right",
    )
    plot_population_metric(
        all_population_stats,
        "posdis",
        "Pos. Disentanglement",
        "posdis_vs_pop.png",
        saved_fig_dir,
        loc="lower right",
    )
    plot_population_metric(
        all_population_stats,
        "ic",
        "Interchangeability",
        "ic_vs_pop.png",
        saved_fig_dir,
        loc="lower right",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute or load LS, TopSim, PosDis, SR, and IC metrics.")
    parser.add_argument(
        "--load-metrics",
        action="store_true",
        help="Load per-seed metrics from saved sim_scores.npz instead of recalculating them.",
    )
    parser.add_argument(
        "--score-file",
        default="metric_scores.npz",
        help="Per-seed metrics filename under ../logs/scoreg/metrics/<model>/<combination>_seed<seed>/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_dir = "../logs/scoreg/"
    sr_root = "../logs/scoreg/sr/"
    saved_dir = "../logs/scoreg/metrics/"
    saved_fig_dir = f"plots/fc/"
    combination_name = "grid5_img3_ni2_nw4_ms30_comm_field100"
    mode = "test"
    model2numnet = {    
        "pop_ppo_2net_invisible": 2,
        "sp_pop_ppo_2net_invisible": 2,

        "pop_ppo_3net_invisible": 3,
        "sp_pop_ppo_3net_invisible": 3,

        "pop_ppo_15net_invisible": 15,
        "sp_pop_ppo_15net_invisible": 15,

        "pop_ppo_30net_invisible": 30,
        "sp_pop_ppo_30net_invisible": 30,

        "pop_ppo_60net_invisible": 60,
        "sp_pop_ppo_60net_invisible": 60,

        "pop_ppo_100net_invisible": 100,
        "sp_pop_ppo_100net_invisible": 100,

    }
    compute_topsim = True
    cbar = True
    all_population_stats = {
        "XP": init_population_stats(),
        "XP+SP": init_population_stats(),
    }
    for model_name in model2numnet.keys():
        num_networks = model2numnet[model_name]
        avg_similarity_mat = np.zeros((num_networks,num_networks))
        avg_sr_mat = np.zeros((num_networks,num_networks))
        per_agent_topsim = []
        per_agent_posdis = []
        seed_ls = []
        seed_topsim = []
        seed_posdis = []
        seed_ic = []

        saved_fig_path_langsim = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_similarity.png")
        saved_fig_path_sr = os.path.join(saved_fig_dir, f"{model_name}_{combination_name}_sr.png")
        for seed in range(1,4):
            
            print(f"{model_name}/{combination_name}")
            saved_score_dir = os.path.join(saved_dir, f"{model_name}/{combination_name}_seed{seed}")
            saved_score_path = os.path.join(saved_score_dir, args.score_file)
            os.makedirs(saved_fig_dir, exist_ok=True)
            os.makedirs(saved_score_dir, exist_ok=True)

            if args.load_metrics:
                try:
                    metric_scores = load_metric_scores(saved_score_path)
                    print(f"Loaded saved metrics: {saved_score_path}")
                except (FileNotFoundError, KeyError) as exc:
                    raise FileNotFoundError(
                        f"Could not load saved metrics for {model_name} seed{seed}. "
                        "Run without --load-metrics to recalculate."
                    ) from exc
            else:
                network_pairs = [f"{i}-0" for i in range(num_networks)]

                log_file_path = {}
                message_data = {}

                # For topsim and posdis calculation, we only keep episodes with length == mode_length, so we need separate dict to store the filtered data
                # This is because topsim requires all episodes to have the same length
                message_data_topsim = {} 
                attribute_data_topsim = {}

                for pair in network_pairs:
                    log_file_path[pair] = os.path.join(
                        load_dir,
                        f"langsim/{model_name}/{pair}/{combination_name}/seed{seed}/mode_{mode}/normal/trajectory.pkl",
                    )
                    log_data = load_trajectory(log_file_path[pair])

                    message_data[pair] = extract_data_for_ls(log_data)
                    message_data_topsim[pair], attribute_data_topsim[pair] = extract_data_for_topsim(log_data)

                similarity_mat, avg_sim = get_similarity(message_data, num_networks)
                if compute_topsim:
                    avg_topsim, avg_posdis, topsim_list, posdis_list = get_comp_scores(
                        message_data_topsim,
                        attribute_data_topsim,
                        num_networks,
                    )
                sr_mat, self_sr, cross_sr, ic = load_sr_metrics(
                    sr_root,
                    model_name,
                    combination_name,
                    seed,
                    mode,
                    "normal",
                    num_networks,
                )
                save_metric_scores(
                    saved_score_path,
                    similarity_mat,
                    avg_sim,
                    avg_topsim,
                    avg_posdis,
                    topsim_list,
                    posdis_list,
                    sr_mat,
                    self_sr,
                    cross_sr,
                    ic,
                )
                metric_scores = {
                    "similarity_mat": similarity_mat,
                    "avg_sim": avg_sim,
                    "avg_topsim": avg_topsim,
                    "avg_posdis": avg_posdis,
                    "topsim_list": topsim_list,
                    "posdis_list": posdis_list,
                    "sr_mat": sr_mat,
                    "self_sr": self_sr,
                    "cross_sr": cross_sr,
                    "ic": ic,
                }

            similarity_mat = metric_scores["similarity_mat"]
            sr_mat = metric_scores["sr_mat"]
            seed_ls.append(metric_scores["avg_sim"])
            seed_topsim.append(metric_scores["avg_topsim"])
            seed_posdis.append(metric_scores["avg_posdis"])
            seed_ic.append(metric_scores["ic"])
            per_agent_topsim += metric_scores["topsim_list"]
            per_agent_posdis += metric_scores["posdis_list"]
            avg_similarity_mat += similarity_mat
            avg_sr_mat += sr_mat
        print(f"Average topsim across all agents: {np.mean(per_agent_topsim)}, SE: {np.std(per_agent_topsim) / np.sqrt(len(per_agent_topsim))}")
        avg_similarity_mat /= 3 # 3 seeds
        avg_sr_mat /= 3 # 3 seeds
        np.savez(os.path.join(saved_fig_dir, "avg_sim_sr_mat.npz"), 
                                            avg_similarity_mat=avg_similarity_mat, 
                                            avg_sr_mat=avg_sr_mat)
        plot_heatmap(avg_similarity_mat, saved_fig_path_langsim, cbar, vmin=0.2, vmax=0.6)
        plot_heatmap(avg_sr_mat, saved_fig_path_sr, cbar, vmin=0.3, vmax=1.0)

        condition = condition_from_model_name(model_name)
        append_population_stats(
            all_population_stats[condition],
            num_networks,
            seed_ls,
            seed_topsim,
            seed_posdis,
            seed_ic,
        )

    plot_population_metrics(all_population_stats, os.path.join(saved_fig_dir, "population_metrics"))
