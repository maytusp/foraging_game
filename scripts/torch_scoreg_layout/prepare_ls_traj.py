# Created for preparing trajectories for language similarity / TopSim / PosDis
# Uses prerecorded initial states so all pairs see identical episode contexts
# Batched over all recorded episodes

from __future__ import annotations
import os
import time
import random
import pickle
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

# python -m scripts.torch_scoreg_layout.prepare_ls_traj --num-networks 100 --model-name pop_ppo_100net_invisible --seed 1 --no-all-pairs --comm-field 100 --model-step 1382400000

# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 2 --model-name sp_pop_ppo_2net_invisible --seed 1
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 2 --model-name sp_pop_ppo_2net_invisible --seed 2
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 2 --model-name sp_pop_ppo_2net_invisible --seed 3

# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 3 --model-name sp_pop_ppo_3net_invisible --seed 1
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 3 --model-name sp_pop_ppo_3net_invisible --seed 2
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 3 --model-name sp_pop_ppo_3net_invisible --seed 3

# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 8 --model-name sp_pop_ppo_8net_invisible --seed 1
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 8 --model-name sp_pop_ppo_8net_invisible --seed 2
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 8 --model-name sp_pop_ppo_8net_invisible --seed 3

# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 16 --model-name sp_pop_ppo_16net_invisible --seed 1
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 16 --model-name sp_pop_ppo_16net_invisible --seed 2
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 16 --model-name sp_pop_ppo_16net_invisible --seed 3

# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 32 --model-name sp_pop_ppo_32net_invisible --seed 1
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 32 --model-name sp_pop_ppo_32net_invisible --seed 2
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 32 --model-name sp_pop_ppo_32net_invisible --seed 3

# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 64 --model-name sp_pop_ppo_64net_invisible --seed 1
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 64 --model-name sp_pop_ppo_64net_invisible --seed 2
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 64 --model-name sp_pop_ppo_64net_invisible --seed 3
@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = True

    # population / checkpoints
    num_networks: int = 100
    model_name: str = "noname"
    model_step: int | None = None
    auto_model_name: bool = False

    # env / rollout
    total_episodes: int = 1000
    grid_size: int = 5
    image_size: int = 3
    comm_field: int = 5
    N_i: int = 2
    max_steps: int = 30
    n_words: int = 5
    d_model: int = 128
    communication_steps: int = 6

    agent_visible: bool = False
    fully_visible_score: bool = False
    time_pressure: bool = True
    mode: str = "test"

    # save
    save_root: str = "logs/vary_n_pop/torch_100net/langsim"

    # initial-state bank
    init_bank_path: str = "logs/init_bank/grid5x5_ni2.npz"

    # language-probing protocol
    fixed_receiver: int = 0
    evaluate_self_pair: bool = True
    all_pairs: bool = False
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
                2: 281600000,
                3: 281600000,
                8: 486400000,
                16: 793600000,
                32: 1484800000,
                64: 1792000000,
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


def load_init_bank(path: str):
    bank = np.load(path, allow_pickle=True)
    required = [
        "episode_id",
        "agent_pos",
        "food_pos",
        "food_energy",
        "target_food_id",
        "score_visible_to_agent",
    ]
    for k in required:
        if k not in bank:
            raise ValueError(f"Missing key '{k}' in init bank: {path}")
    return bank


def set_env_from_bank(envs: TorchForagingEnv, bank, device: torch.device):
    """
    Overwrite all env states from the prerecorded bank.
    Assumes envs.B == num recorded episodes.
    """
    episode_id = bank["episode_id"]
    agent_pos = torch.tensor(bank["agent_pos"], dtype=torch.long, device=device)
    food_pos = torch.tensor(bank["food_pos"], dtype=torch.long, device=device)
    food_energy = torch.tensor(bank["food_energy"], dtype=torch.float32, device=device)
    target_food_id = torch.tensor(bank["target_food_id"], dtype=torch.long, device=device)
    score_visible_to_agent = torch.tensor(bank["score_visible_to_agent"], dtype=torch.long, device=device)

    B = agent_pos.shape[0]
    A = agent_pos.shape[1]
    F = food_pos.shape[1]

    if envs.B != B:
        raise ValueError(f"env batch size {envs.B} does not match init bank size {B}")

    envs.agent_pos[:] = agent_pos
    envs.food_pos[:] = food_pos
    envs.food_energy[:] = food_energy
    envs.target_food_id[:] = target_food_id
    envs.score_visible_to_agent[:] = score_visible_to_agent

    envs.food_done[:] = False
    envs.agent_energy[:] = 20.0
    envs.curr_steps[:] = 0
    envs.dones_batch[:] = False
    envs.trunc_batch[:] = False
    envs.total_bump[:] = 0
    envs.cum_rewards[:] = 0.0
    envs.episode_len[:] = 0
    envs.comm_started[:] = False
    envs.comm_steps_used[:] = 0

    return episode_id


@torch.no_grad()
def run_pair_episodes_batched(
    sender_agent: PPOLSTMCommAgent,
    receiver_agent: PPOLSTMCommAgent,
    envs: TorchForagingEnv,
    bank,
    args: Args,
    device: torch.device,
) -> dict:
    """
    Run all prerecorded episodes in parallel.
    Record each env trajectory only until its first done.
    """
    swap_agent = {0: 1, 1: 0}
    num_agents = 2

    episode_ids = set_env_from_bank(envs, bank, device)
    B = envs.B

    next_obs, next_locs, _ = envs.observe()
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

    # per-env fixed metadata from init bank
    target_food_id = envs.target_food_id.detach().cpu().numpy()
    food_pos = envs.food_pos.detach().cpu().numpy()
    food_energy = envs.food_energy.detach().cpu().numpy()
    score_visible_to_agent = envs.score_visible_to_agent.detach().cpu().numpy()

    # logs [T, B, A]
    log_s_messages = torch.full((args.max_steps, B, num_agents), -1, dtype=torch.int64, device=device)
    log_rewards = torch.zeros((args.max_steps, B, num_agents), device=device)

    alive = torch.ones(B, dtype=torch.bool, device=device)
    first_done_step = torch.full((B,), args.max_steps, dtype=torch.long, device=device)

    for ep_step in range(args.max_steps):
        if not alive.any():
            break

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
        s_messages = torch.stack([s_message0, s_message1], dim=1)  # [B,2]

        # record only still-alive envs
        alive_idx = alive.nonzero(as_tuple=False).squeeze(1)
        log_s_messages[ep_step, alive_idx] = s_messages[alive_idx]

        (next_obs, next_locs, msg_masks), all_rewards, all_terminations, all_truncations, infos = envs.step(
            acts_BA, auto_reset=True
        )

        if args.ablate_message:
            msg_masks = torch.zeros_like(msg_masks, dtype=torch.bool)

        msg_gate = msg_masks.unsqueeze(-1).sum((1, 2)).clamp(max=1)  # [B,1]

        for i in range(num_agents):
            next_r_messages[:, i] = msg_gate.squeeze() * s_messages[:, swap_agent[i]]

        done = (all_terminations | all_truncations).bool()
        next_done0 = done.float()
        next_done1 = done.float()

        # record rewards only for envs still alive at step start
        log_rewards[ep_step, alive_idx] = all_rewards[alive_idx]

        just_finished = alive & done
        if just_finished.any():
            first_done_step[just_finished] = ep_step + 1  # length in steps
            alive = alive & (~just_finished)

        if args.zero_memory and just_finished.any():
            idx = just_finished.nonzero(as_tuple=False).squeeze(1)
            next_lstm_state0[0][:, idx, :] = 0.0
            next_lstm_state0[1][:, idx, :] = 0.0
            next_lstm_state1[0][:, idx, :] = 0.0
            next_lstm_state1[1][:, idx, :] = 0.0
            next_r_messages[idx] = 0

    # convert to per-episode dict matching old format
    log_s_messages_np = log_s_messages.detach().cpu().numpy()   # [T,B,A]
    log_rewards_np = log_rewards.detach().cpu().numpy()         # [T,B,A]
    first_done_step_np = first_done_step.detach().cpu().numpy()

    log_data = {}

    for b in range(B):
        ep_id = int(episode_ids[b])
        tgt_id = int(target_food_id[b])
        who_see_target = int(score_visible_to_agent[b][tgt_id])

        distractor_ids = [k for k in range(food_energy.shape[1]) if k != tgt_id]
        distractor_locs = food_pos[b][distractor_ids]
        distractor_scores = food_energy[b][distractor_ids]

        # keep full padded arrays exactly like before
        ep_log_s_messages = log_s_messages_np[:, b, :]
        ep_log_rewards = log_rewards_np[:, b, :]

        log_target_food_dict = {
            "id": tgt_id,
            "location": food_pos[b][tgt_id],
            "score": food_energy[b][tgt_id],
        }
        log_distractor_food_dict = {
            "ids": np.array(distractor_ids, dtype=np.int64),
            "location": distractor_locs,
            "score": distractor_scores,
        }

        log_data[f"episode_{ep_id}"] = {
            "episode_id": ep_id,
            "log_s_messages": ep_log_s_messages,
            "log_rewards": ep_log_rewards,
            "who_see_target": who_see_target,
            "log_target_food_dict": log_target_food_dict,
            "log_distractor_food_dict": log_distractor_food_dict,
        }

    return log_data


def save_pair_outputs(
    pair_log_data: dict,
    sender_id: int,
    receiver_id: int,
    args: Args,
    model_name: str,
):
    combination_name = build_combination_name(args)
    pair_name = f"{sender_id}-{receiver_id}"
    save_dir = os.path.join(
        args.save_root,
        model_name,
        pair_name,
        combination_name,
        f"seed{args.seed}",
        f"mode_{args.mode}",
        "normal",
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "trajectory.pkl"), "wb") as f:
        pickle.dump(pair_log_data, f)

    print(f"Saved trajectory to {os.path.join(save_dir, 'trajectory.pkl')}")


def main():
    args = tyro.cli(Args)
    set_seed(args.seed, args.torch_deterministic)

    model_name = resolve_model_name(args)
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

    print(f"Preparing trajectories for {len(pairs)} pairs in batch")
    start_time = time.time()

    for k, (sender_id, receiver_id) in enumerate(pairs):
        print(f"[{k+1}/{len(pairs)}] running pair {sender_id}-{receiver_id}")
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