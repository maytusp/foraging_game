# record_init_bank.py
# Record a fixed bank of initial episode conditions for langsim / topsim evaluation
#
# Example:
# CUDA_VISIBLE_DEVICES=0 python -m scripts.torch_scoreg_layout.prepare_eval_episode_states \
#   --num-episodes 1000 --seed 1 --save-path logs/init_bank/grid5_test_seed1.npz

from __future__ import annotations
import os
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


@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = True

    num_episodes: int = 1000
    save_path: str = "logs/init_bank/grid5x5_ni2.npz"

    # env config
    grid_size: int = 5
    image_size: int = 3
    comm_field: int = 100
    num_foods: int = 2
    max_steps: int = 30
    communication_steps: int = 6

    agent_visible: bool = False
    fully_visible_score: bool = False
    time_pressure: bool = True
    mode: str = "test"

    ascii_layout_name: str = "simple_layout_5x5"
    use_compile: bool = False


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def resolve_layout(args: Args):
    layout_map = {
        "simple_layout_5x5": simple_layout_5x5,
    }
    if args.ascii_layout_name not in layout_map:
        raise ValueError(f"Unknown ascii_layout_name: {args.ascii_layout_name}")
    return layout_map[args.ascii_layout_name]


def main():
    args = tyro.cli(Args)
    set_seed(args.seed, args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    ascii_layout = resolve_layout(args)

    cfg = EnvConfig(
        grid_size=args.grid_size,
        image_size=args.image_size,
        comm_field=args.comm_field,
        num_agents=2,
        num_foods=args.num_foods,
        num_walls=0,
        max_steps=args.max_steps,
        agent_visible=args.agent_visible,
        food_energy_fully_visible=args.fully_visible_score,
        mode=args.mode,
        seed=args.seed,
        time_pressure=args.time_pressure,
        communication_steps=args.communication_steps,
        ascii_layout=ascii_layout,
        use_compile=args.use_compile,
    )

    envs = TorchForagingEnv(cfg, device=device, num_envs=1)

    episode_id_list = []
    agent_pos_list = []
    food_pos_list = []
    food_energy_list = []
    target_food_id_list = []
    score_visible_to_agent_list = []

    reset_mask = torch.ones((1,), dtype=torch.bool, device=device)

    print(f"Recording {args.num_episodes} initial episodes on {device}")

    for episode_id in range(1, args.num_episodes + 1):
        envs._reset_indices(reset_mask)

        episode_id_list.append(episode_id)
        agent_pos_list.append(envs.agent_pos.squeeze(0).detach().cpu().numpy().copy())
        food_pos_list.append(envs.food_pos.squeeze(0).detach().cpu().numpy().copy())
        food_energy_list.append(envs.food_energy.squeeze(0).detach().cpu().numpy().copy())
        target_food_id_list.append(envs.target_food_id.squeeze(0).detach().cpu().numpy().copy())
        score_visible_to_agent_list.append(
            envs.score_visible_to_agent.squeeze(0).detach().cpu().numpy().copy()
        )

        if episode_id % 100 == 0 or episode_id == args.num_episodes:
            print(f"[{episode_id}/{args.num_episodes}] recorded")

    envs.close()

    episode_id = np.array(episode_id_list, dtype=np.int64)                 # [E]
    agent_pos = np.stack(agent_pos_list, axis=0)                           # [E, A, 2]
    food_pos = np.stack(food_pos_list, axis=0)                             # [E, F, 2]
    food_energy = np.stack(food_energy_list, axis=0)                       # [E, F]
    target_food_id = np.stack(target_food_id_list, axis=0)                 # [E]
    score_visible_to_agent = np.stack(score_visible_to_agent_list, axis=0) # [E, F]

    meta = {
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "grid_size": args.grid_size,
        "image_size": args.image_size,
        "comm_field": args.comm_field,
        "num_foods": args.num_foods,
        "max_steps": args.max_steps,
        "communication_steps": args.communication_steps,
        "agent_visible": args.agent_visible,
        "fully_visible_score": args.fully_visible_score,
        "time_pressure": args.time_pressure,
        "mode": args.mode,
        "ascii_layout_name": args.ascii_layout_name,
    }

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    np.savez(
        args.save_path,
        episode_id=episode_id,
        agent_pos=agent_pos,
        food_pos=food_pos,
        food_energy=food_energy,
        target_food_id=target_food_id,
        score_visible_to_agent=score_visible_to_agent,
        meta=np.array(meta, dtype=object),
    )

    print(f"Saved init bank to: {args.save_path}")
    print(f"episode_id shape: {episode_id.shape}")
    print(f"agent_pos shape: {agent_pos.shape}")
    print(f"food_pos shape: {food_pos.shape}")
    print(f"food_energy shape: {food_energy.shape}")
    print(f"target_food_id shape: {target_food_id.shape}")
    print(f"score_visible_to_agent shape: {score_visible_to_agent.shape}")


if __name__ == "__main__":
    main()