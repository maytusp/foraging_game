# Created for preparing language-similarity trajectories over graph edges.
# Based on prepare_ls_struct.py, but selects evaluation pairs from a predefined
# population graph instead of a fixed receiver / all-pairs protocol.

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import tyro

from environments.torch_scoreg_layout import EnvConfig, TorchForagingEnv
from scripts.torch_scoreg_layout.prepare_ls_traj import (
    Args as StructArgs,
    build_combination_name,
    load_init_bank,
    load_population,
    resolve_layout,
    run_pair_episodes_batched,
    save_pair_outputs,
    set_seed,
)
from utils.graph_gen import clq_pairs_100


@dataclass
class Args(StructArgs):
    # graph-probing protocol
    graph_structure: str = "ring"  # ring, fixed_receiver, all_pairs, clq_pairs_100
    include_reverse_edges: bool = False
    include_self_pairs: bool = False


def ring_pairs(num_networks: int) -> list[tuple[int, int]]:
    if num_networks < 2:
        raise ValueError("ring graph requires num_networks >= 2")
    return [(i, (i + 1) % num_networks) for i in range(num_networks)]


def fixed_receiver_pairs(args: Args) -> list[tuple[int, int]]:
    senders = list(range(args.sender_start, min(args.sender_end, args.num_networks)))
    pairs = []
    for sender_id in senders:
        if sender_id == args.fixed_receiver and not args.evaluate_self_pair:
            continue
        pairs.append((sender_id, args.fixed_receiver))
    return pairs


def add_pair_if_missing(
    pairs: list[tuple[int, int]], pair: tuple[int, int]
) -> list[tuple[int, int]]:
    if pair not in pairs:
        pairs.append(pair)
    return pairs


def get_pair_list(args: Args) -> list[tuple[int, int]]:
    if args.graph_structure == "ring":
        pairs = ring_pairs(args.num_networks)
    elif args.graph_structure == "fixed_receiver":
        pairs = fixed_receiver_pairs(args)
    elif args.graph_structure == "all_pairs":
        pairs = [
            (i, j)
            for i in range(args.num_networks)
            for j in range(args.num_networks)
        ]
    elif args.graph_structure == "clq_pairs_100":
        if args.num_networks != 100:
            raise ValueError(
                "graph_structure='clq_pairs_100' requires num_networks=100"
            )
        pairs = [tuple(pair) for pair in clq_pairs_100]
    else:
        raise ValueError(
            "graph_structure must be one of "
            "{'ring', 'fixed_receiver', 'all_pairs', 'clq_pairs_100'}"
        )

    if args.graph_structure in {"ring", "clq_pairs_100"}:
        pairs = add_pair_if_missing(pairs, (args.num_networks - 1, 0))

    if args.include_reverse_edges:
        existing = set(pairs)
        for sender_id, receiver_id in list(pairs):
            reverse_pair = (receiver_id, sender_id)
            if reverse_pair not in existing:
                pairs.append(reverse_pair)
                existing.add(reverse_pair)

    if args.include_self_pairs:
        existing = set(pairs)
        for agent_id in range(args.num_networks):
            self_pair = (agent_id, agent_id)
            if self_pair not in existing:
                pairs.append(self_pair)
                existing.add(self_pair)

    return pairs


def main():
    args = tyro.cli(Args)
    set_seed(args.seed, args.torch_deterministic)

    model_name = args.model_name
    ascii_layout = resolve_layout(args)
    bank = load_init_bank(args.init_bank_path)

    num_bank_episodes = len(bank["episode_id"])
    if args.total_episodes != num_bank_episodes:
        print(
            f"Warning: args.total_episodes={args.total_episodes} but init bank has "
            f"{num_bank_episodes} episodes. Using init bank size."
        )
        args.total_episodes = num_bank_episodes

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    print(f"Model name: {model_name}")
    print(f"Graph structure: {args.graph_structure}")
    print(f"Combination name: {build_combination_name(args)}")
    print(f"Init bank: {args.init_bank_path}")

    cfg = EnvConfig(
        grid_size=args.grid_size,
        image_size=args.image_size,
        comm_field=args.comm_field,
        num_agents=2,
        num_foods=args.N_i,
        num_walls=0,
        max_steps=args.max_steps,
        agent_visible=args.agent_visible,
        food_energy_fully_visible=args.fully_visible_score,
        mode=args.mode,
        seed=args.seed,
        time_pressure=args.time_pressure,
        communication_steps=args.communication_steps,
        ascii_layout=ascii_layout,
        use_compile=False,
    )

    envs = TorchForagingEnv(cfg, device=device, num_envs=args.total_episodes)
    num_actions = envs.num_actions
    num_channels = cfg.num_channels

    agents = load_population(args, device, num_actions, num_channels, model_name)
    pairs = get_pair_list(args)

    print(f"Preparing trajectories for {len(pairs)} graph pairs in batch")
    print(f"Pairs: {pairs}")
    start_time = time.time()

    for pair_idx, (sender_id, receiver_id) in enumerate(pairs):
        print(f"[{pair_idx + 1}/{len(pairs)}] running pair {sender_id}-{receiver_id}")
        pair_log_data = run_pair_episodes_batched(
            sender_agent=agents[sender_id],
            receiver_agent=agents[receiver_id],
            envs=envs,
            bank=bank,
            args=args,
            device=device,
        )
        save_pair_outputs(pair_log_data, sender_id, receiver_id, args, model_name)

    envs.close()
    elapsed = time.time() - start_time
    print(f"Done. Total time: {elapsed:.2f} sec")


if __name__ == "__main__":
    main()
