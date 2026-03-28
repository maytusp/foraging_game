# Batched success-rate evaluation for population agents
# Updated to match train_large_pop_fc.py
# Example:
# CUDA_VISIBLE_DEVICES=0 python -m scripts.torch_scoreg_layout.eval_batched_sr --num-networks 100 --model-name fc_ppo_100net_invisible --seed 1 --all-pairs --batch-num-envs 1024

from __future__ import annotations
import os
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import tyro

from environments.torch_scoreg_layout import (
    EnvConfig,
    TorchForagingEnv,
    simple_layout_5x5,
)
from models.pickup_models import PPOLSTMCommAgent


@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = True

    # population / checkpoints
    num_networks: int = 100
    model_name: str = "fc_ppo_100net_invisible"
    model_step: int | None = None
    auto_model_name: bool = False

    # env / rollout
    total_episodes: int = 1000
    batch_num_envs: int = 256
    grid_size: int = 5
    image_size: int = 3
    comm_field: int = 5
    N_i: int = 2
    max_steps: int = 30
    n_words: int = 4
    d_model: int = 128
    communication_steps: int = 6

    agent_visible: bool = False
    fully_visible_score: bool = False
    time_pressure: bool = True
    mode: str = "test"

    # save
    save_root: str = "logs/vary_n_pop/torch_100net/sr_batched"

    # pair protocol
    fixed_receiver: int = 0
    evaluate_self_pair: bool = True
    all_pairs: bool = True
    sender_start: int = 0
    sender_end: int = 100  # exclusive

    # intervention
    ablate_message: bool = False
    ablate_type: str = "noise"  # zero, noise
    zero_memory: bool = False

    # naming
    ckpt_root: str = "checkpoints/torch_scoreg_layout"
    ascii_layout_name: str = "simple_layout_5x5"

    num_nets_to_model_step: dict[int, int] = None

    def __post_init__(self):
        if self.num_nets_to_model_step is None:
            self.num_nets_to_model_step = {
                3: 204800000,
                6: 460800000,
                9: 512000000,
                12: 768000000,
                15: 819200000,
                100: 2048000000,
            }
        if self.model_step is None:
            self.model_step = self.num_nets_to_model_step[self.num_networks]


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def resolve_model_name(args: Args) -> str:
    if not args.auto_model_name:
        return args.model_name

    model_name = f"fc_ppo_{args.num_networks}net"
    if not args.agent_visible:
        model_name += "_invisible"
    if not args.time_pressure:
        model_name += "_wospeedrw"
    if args.ablate_message:
        model_name += "_nocom"
    return model_name


def resolve_layout(args: Args):
    layout_map = {
        "simple_layout_5x5": simple_layout_5x5,
    }
    if args.ascii_layout_name not in layout_map:
        raise ValueError(f"Unknown ascii_layout_name: {args.ascii_layout_name}")
    return layout_map[args.ascii_layout_name]


def build_combination_name(args: Args) -> str:
    return (
        f"grid{args.grid_size}_img{args.image_size}_ni{args.N_i}"
        f"_nw{args.n_words}_ms{args.max_steps}_comm_field{args.comm_field}"
    )


def build_agent(args: Args, device: torch.device, num_actions: int, num_channels: int) -> PPOLSTMCommAgent:
    return PPOLSTMCommAgent(
        d_model=args.d_model,
        num_actions=num_actions,
        grid_size=args.grid_size,
        n_words=args.n_words,
        embedding_size=16,
        num_channels=num_channels,
        image_size=args.image_size,
    ).to(device)


def load_population(
    args: Args,
    device: torch.device,
    num_actions: int,
    num_channels: int,
    model_name: str,
) -> list[PPOLSTMCommAgent]:
    agents = []
    combination_name = build_combination_name(args)

    for agent_id in range(args.num_networks):
        ckpt_path = (
            f"{args.ckpt_root}/{model_name}/{combination_name}/"
            f"seed{args.seed}/agent_{agent_id}_step_{args.model_step}.pt"
        )
        agent = build_agent(args, device, num_actions, num_channels)
        state_dict = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()
        agents.append(agent)
        print(f"Loaded agent {agent_id} from {ckpt_path}")

    return agents


def get_pair_list(args: Args) -> list[tuple[int, int]]:
    if args.all_pairs:
        return [(i, j) for i in range(args.num_networks) for j in range(args.num_networks)]

    senders = list(range(args.sender_start, min(args.sender_end, args.num_networks)))
    pairs = []
    for s in senders:
        if (s == args.fixed_receiver) and (not args.evaluate_self_pair):
            continue
        pairs.append((s, args.fixed_receiver))
    return pairs


def zero_lstm_indices(lstm_state: tuple[torch.Tensor, torch.Tensor], idx: torch.Tensor):
    if idx.numel() == 0:
        return
    h, c = lstm_state
    h[:, idx, :] = 0.0
    c[:, idx, :] = 0.0


@torch.no_grad()
def evaluate_pair_sr_batched(
    sender_agent: PPOLSTMCommAgent,
    receiver_agent: PPOLSTMCommAgent,
    args: Args,
    device: torch.device,
    ascii_layout,
) -> tuple[float, float]:
    """
    Returns:
        sr: success rate over total_episodes
        avg_len: average episode length over total_episodes
    """
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
    )

    envs = TorchForagingEnv(cfg, device=device, num_envs=args.batch_num_envs)

    num_agents = 2
    swap_agent = {0: 1, 1: 0}
    B = args.batch_num_envs

    next_obs, next_locs, _ = envs._obs_core()
    next_r_messages = torch.zeros((B, num_agents), dtype=torch.int64, device=device)
    next_done0 = torch.zeros(B, device=device)
    next_done1 = torch.zeros(B, device=device)

    next_lstm_state0 = (
        torch.zeros(sender_agent.lstm.num_layers, B, sender_agent.lstm.hidden_size, device=device),
        torch.zeros(sender_agent.lstm.num_layers, B, sender_agent.lstm.hidden_size, device=device),
    )
    next_lstm_state1 = (
        torch.zeros(receiver_agent.lstm.num_layers, B, receiver_agent.lstm.hidden_size, device=device),
        torch.zeros(receiver_agent.lstm.num_layers, B, receiver_agent.lstm.hidden_size, device=device),
    )

    ep_returns = torch.zeros(B, device=device)
    ep_lengths = torch.zeros(B, device=device)

    finished_episodes = 0
    success_count = 0
    sum_ep_lengths = 0.0

    while finished_episodes < args.total_episodes:
        if args.ablate_message:
            if args.ablate_type == "zero":
                next_r_messages = torch.zeros_like(next_r_messages)
            elif args.ablate_type == "noise":
                next_r_messages = torch.randint(0, args.n_words, next_r_messages.shape, device=device)
            else:
                raise ValueError("ablate_type must be one of {'zero', 'noise'}")

        action0, _, _, s_message0, _, _, _, next_lstm_state0 = sender_agent.get_action_and_value(
            (
                next_obs[:, 0, :, :, :],
                next_locs[:, 0, :],
                next_r_messages[:, 0],
            ),
            next_lstm_state0,
            next_done0,
        )

        action1, _, _, s_message1, _, _, _, next_lstm_state1 = receiver_agent.get_action_and_value(
            (
                next_obs[:, 1, :, :, :],
                next_locs[:, 1, :],
                next_r_messages[:, 1],
            ),
            next_lstm_state1,
            next_done1,
        )

        acts_BA = torch.stack([action0.long(), action1.long()], dim=1)

        (next_obs, next_locs, msg_masks), all_rewards, all_terminations, all_truncations, infos = envs._step_core(acts_BA)

        if args.ablate_message:
            msg_masks = torch.zeros_like(msg_masks, dtype=torch.bool)

        msg_gate = msg_masks.unsqueeze(-1).sum((1, 2)).clamp(max=1)  # [B,1]

        s_messages = torch.stack([s_message0, s_message1], dim=1)
        for i in range(num_agents):
            next_r_messages[:, i] = msg_gate.squeeze() * s_messages[:, swap_agent[i]]

        done = (all_terminations | all_truncations).bool()
        next_done0 = done.float()
        next_done1 = done.float()

        ep_returns += all_rewards[:, 0]
        ep_lengths += 1

        if done.any():
            finished_idx_all = done.nonzero(as_tuple=False).squeeze(1)

            remaining = args.total_episodes - finished_episodes
            take = min(finished_idx_all.numel(), remaining)
            finished_idx = finished_idx_all[:take]

            success_count += int((ep_returns[finished_idx] >= 1.0).sum().item())
            sum_ep_lengths += float(ep_lengths[finished_idx].sum().item())
            finished_episodes += take

            # reset policy-side states for all finished envs, not just the counted subset
            next_r_messages[finished_idx_all] = 0
            zero_lstm_indices(next_lstm_state0, finished_idx_all)
            zero_lstm_indices(next_lstm_state1, finished_idx_all)
            ep_returns[finished_idx_all] = 0.0
            ep_lengths[finished_idx_all] = 0.0

            if args.zero_memory:
                zero_lstm_indices(next_lstm_state0, finished_idx_all)
                zero_lstm_indices(next_lstm_state1, finished_idx_all)

    envs.close()

    sr = success_count / args.total_episodes
    avg_len = sum_ep_lengths / args.total_episodes
    return sr, avg_len


def save_outputs(
    sr_mat: np.ndarray,
    len_mat: np.ndarray,
    evaluated_mask: np.ndarray,
    args: Args,
    model_name: str,
):
    combination_name = build_combination_name(args)
    save_dir = os.path.join(
        args.save_root,
        model_name,
        combination_name,
        f"seed{args.seed}",
        f"mode_{args.mode}",
        "normal",
    )
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "sr_mat.npy"), sr_mat)
    np.save(os.path.join(save_dir, "len_mat.npy"), len_mat)

    np.savez(
        os.path.join(save_dir, "sr_eval.npz"),
        sr_mat=sr_mat,
        len_mat=len_mat,
        evaluated_mask=evaluated_mask,
    )

    valid = evaluated_mask.astype(bool)
    avg_sr = float(sr_mat[valid].mean()) if valid.any() else float("nan")
    avg_len = float(len_mat[valid].mean()) if valid.any() else float("nan")

    with open(os.path.join(save_dir, "score.txt"), "w") as f:
        print(f"Average Success Rate: {avg_sr}", file=f)
        print(f"Average Episode Length: {avg_len}", file=f)
        print(f"Num Evaluated Pairs: {int(valid.sum())}", file=f)

    print(f"Saved SR outputs to {save_dir}")


def main():
    args = tyro.cli(Args)
    set_seed(args.seed, args.torch_deterministic)

    model_name = resolve_model_name(args)
    ascii_layout = resolve_layout(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    print(f"Model name: {model_name}")
    print(f"Combination name: {build_combination_name(args)}")

    # build a small env just to read num_actions / num_channels
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
    )
    envs = TorchForagingEnv(cfg, device=device, num_envs=1)
    num_actions = envs.num_actions
    num_channels = cfg.num_channels
    envs.close()

    agents = load_population(args, device, num_actions, num_channels, model_name)
    pairs = get_pair_list(args)

    sr_mat = np.full((args.num_networks, args.num_networks), np.nan, dtype=np.float32)
    len_mat = np.full((args.num_networks, args.num_networks), np.nan, dtype=np.float32)
    evaluated_mask = np.zeros((args.num_networks, args.num_networks), dtype=np.int32)

    print(f"Evaluating {len(pairs)} pairs with batch_num_envs={args.batch_num_envs}")
    start_time = time.time()

    for k, (sender_id, receiver_id) in enumerate(pairs):
        print(f"[{k+1}/{len(pairs)}] running pair {sender_id}-{receiver_id}")

        pair_start = time.time()
        sr, avg_len = evaluate_pair_sr_batched(
            sender_agent=agents[sender_id],
            receiver_agent=agents[receiver_id],
            args=args,
            device=device,
            ascii_layout=ascii_layout,
        )
        pair_elapsed = time.time() - pair_start

        sr_mat[sender_id, receiver_id] = sr
        len_mat[sender_id, receiver_id] = avg_len
        evaluated_mask[sender_id, receiver_id] = 1

        print(
            f"pair {sender_id}-{receiver_id} | "
            f"SR={sr:.4f} | AvgLen={avg_len:.2f} | Time={pair_elapsed:.2f}s"
        )

    save_outputs(sr_mat, len_mat, evaluated_mask, args, model_name)

    elapsed = time.time() - start_time
    print(f"Done. Total time: {elapsed:.2f} sec")


if __name__ == "__main__":
    main()