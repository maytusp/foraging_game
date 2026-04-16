# 30 Mar 2026
# MAPPO with non-parameter sharing (actors have separate sets of parameters)
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from environments.torch_scoreg_layout import (
    TorchForagingEnv,
    EnvConfig,
    simple_layout_5x5,
)
from utils.process_data import *
from models.pickup_models import PPOLSTMCommActor, CentralizedCritic

# CUDA_VISIBLE_DEVICES=0 python -m scripts.torch_scoreg_layout.train_nops_mappo --seed 1 --comm_field 100 --num_networks 2 --no-agent-visible


@dataclass
class Args:
    seed: int = 4
    """seed of the experiment"""

    env_id: str = "Foraging-Single-v1"
    total_timesteps: int = int(5e8)

    learning_rate: float = 2.5e-4
    critic_learning_rate: float = 2.5e-4

    num_envs: int = 512
    num_steps: int = 30
    anneal_lr: bool = True

    gamma: float = 0.99
    gae_lambda: float = 0.95

    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True

    clip_coef: float = 0.1
    clip_vloss: bool = True

    ent_coef: float = 0.01
    m_ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # population
    num_networks: int = 2
    reset_iteration: int = 1
    self_play_option: bool = False

    log_every: int = 32
    d_model: int = 128
    n_words: int = 5
    image_size: int = 3
    comm_field: int = 100
    num_foods: int = 2
    grid_size: int = 5
    max_steps: int = 10
    communication_steps: int = 10

    # layout curriculum
    warmup_steps: int = int(total_timesteps * 0.1)
    reset_on_phase_change: bool = False
    first_layout = simple_layout_5x5
    final_layout = simple_layout_5x5

    agent_visible: bool = True
    time_pressure: bool = True
    ablate_message: bool = False
    mode: str = "train"

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    load_pretrained = False
    if load_pretrained:
        pretrained_global_step = 1177600000
        learning_rate = 2e-4
        critic_learning_rate = 2e-4
        print(f"LOAD from {pretrained_global_step}")
        ckpt_path = {a: f"" for a in range(num_networks)}
        critic_ckpt_path = ""

    visualize_loss = True
    save_frequency = int(5e4)

    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "scoreg_layout"
    wandb_entity: str = "maytusp"
    capture_video: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.self_play_option:
        sp_prefix = "sp_"
    else:
        sp_prefix = ""

    model_name = f"{sp_prefix}pop_mappo_{args.num_networks}net_commstep{args.communication_steps}"
    if not args.agent_visible:
        model_name += "_invisible"
    if not args.time_pressure:
        model_name += "_wospeedrw"
    if args.ablate_message:
        model_name += "_nocom"

    train_combination_name = (
        f"grid{args.grid_size}_img{args.image_size}_ni{args.num_foods}"
        f"_nw{args.n_words}_ms{args.max_steps}_comm_field{args.comm_field}"
    )

    save_dir = f"checkpoints/torch_scoreg_layout/{model_name}/{train_combination_name}/seed{args.seed}/"
    os.makedirs(save_dir, exist_ok=True)

    run_name = f"{model_name}/{train_combination_name}_seed{args.seed}"

    print("parsed seed:", args.seed)
    print("run_name:", run_name)
    print("save_dir:", save_dir)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/scoreg_layout/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    cfg = EnvConfig(
        grid_size=args.grid_size,
        image_size=args.image_size,
        comm_field=args.comm_field,
        num_agents=2,
        num_foods=args.num_foods,
        num_walls=0,
        max_steps=args.max_steps,
        agent_visible=args.agent_visible,
        mode=args.mode,
        seed=args.seed,
        time_pressure=args.time_pressure,
        communication_steps=args.communication_steps,
        ascii_layout=args.first_layout,
    )
    envs = TorchForagingEnv(cfg, device=device, num_envs=args.num_envs)

    layout_phase_switched = False
    num_agents = cfg.num_agents
    num_channels = cfg.num_channels
    num_actions = envs.num_actions

    # actor networks
    agents = {}
    optimizers = {}

    # per-agent rollout buffers
    obs = {}
    locs = {}
    r_messages = {}
    actions = {}
    s_messages = {}
    action_logprobs = {}
    message_logprobs = {}
    rewards = {}
    dones = {}
    message_masks = {}

    # centralized critic rollout buffers
    joint_obs = torch.zeros(
        (args.num_steps, args.num_envs, num_agents, num_channels, args.image_size, args.image_size),
        device=device,
    )
    joint_locs = torch.zeros(
        (args.num_steps, args.num_envs, num_agents, 2),
        device=device,
    )
    joint_r_messages = torch.zeros(
        (args.num_steps, args.num_envs, num_agents),
        dtype=torch.int64,
        device=device,
    )
    team_rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    team_dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    shared_values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, next_locs, next_done, next_lstm_state = {}, {}, {}, {}
    next_obs, next_locs, _ = envs._obs_core()
    next_r_messages = torch.zeros((args.num_envs, num_agents), dtype=torch.int64, device=device)

    swap_agent = {0: 1, 1: 0}

    for network_id in range(args.num_networks):
        agents[network_id] = PPOLSTMCommActor(
            d_model=args.d_model,
            num_actions=num_actions,
            grid_size=args.grid_size,
            n_words=args.n_words,
            embedding_size=16,
            num_channels=num_channels,
            image_size=args.image_size,
        ).to(device)

        print("NUM ACTOR PARAMS:", count_parameters(agents[network_id]))

        if args.load_pretrained:
            agents[network_id].load_state_dict(torch.load(args.ckpt_path[network_id], map_location=device))

        optimizers[network_id] = optim.Adam(
            agents[network_id].parameters(),
            lr=args.learning_rate,
            eps=1e-5,
        )

    critic = CentralizedCritic(
        num_agents=num_agents,
        num_channels=num_channels,
        image_size=args.image_size,
        n_words=args.n_words,
        embedding_size=16,
        d_model=args.d_model,
        grid_size=args.grid_size,
    ).to(device)
    print("NUM CRITIC PARAMS:", count_parameters(critic))

    if args.load_pretrained and args.critic_ckpt_path:
        critic.load_state_dict(torch.load(args.critic_ckpt_path, map_location=device))

    critic_optimizer = optim.Adam(
        critic.parameters(),
        lr=args.critic_learning_rate,
        eps=1e-5,
    )

    for i in range(num_agents):
        obs[i] = torch.zeros(
            (args.num_steps, args.num_envs, num_channels, args.image_size, args.image_size),
            device=device,
        )
        locs[i] = torch.zeros((args.num_steps, args.num_envs, 2), device=device)
        r_messages[i] = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64, device=device)
        actions[i] = torch.zeros((args.num_steps, args.num_envs), device=device)
        s_messages[i] = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64, device=device)
        action_logprobs[i] = torch.zeros((args.num_steps, args.num_envs), device=device)
        message_logprobs[i] = torch.zeros((args.num_steps, args.num_envs), device=device)
        message_masks[i] = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
        rewards[i] = torch.zeros((args.num_steps, args.num_envs), device=device)
        dones[i] = torch.zeros((args.num_steps, args.num_envs), device=device)

        next_done[i] = torch.zeros(args.num_envs, device=device)
        next_lstm_state[i] = (
            torch.zeros(agents[0].lstm.num_layers, args.num_envs, agents[0].lstm.hidden_size, device=device),
            torch.zeros(agents[0].lstm.num_layers, args.num_envs, agents[0].lstm.hidden_size, device=device),
        )

    start_time = time.time()
    global_step = 0
    initial_lstm_state = {}

    possible_networks = [i for i in range(args.num_networks)]
    selected_networks = [0, 1]

    # logging
    episodes_since_log = 0
    sum_return_since_log = 0.0
    sum_length_since_log = 0.0

    ep_ret = torch.zeros(args.num_envs, num_agents, device=device)
    ep_len = torch.zeros(args.num_envs, num_agents, device=device)

    LOG_EVERY_EPISODES = getattr(args, "log_every_episodes", args.num_envs)

    for iteration in range(1, args.num_iterations + 1):
        if iteration % args.reset_iteration == 0:
            for i in range(num_agents):
                next_lstm_state[i] = (
                    torch.zeros(agents[0].lstm.num_layers, args.num_envs, agents[0].lstm.hidden_size, device=device),
                    torch.zeros(agents[0].lstm.num_layers, args.num_envs, agents[0].lstm.hidden_size, device=device),
                )
            selected_networks = np.random.choice(
                possible_networks, num_agents, replace=args.self_play_option
            )

        for i in range(num_agents):
            initial_lstm_state[i] = (
                next_lstm_state[i][0].clone(),
                next_lstm_state[i][1].clone(),
            )

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow_actor = frac * args.learning_rate
            lrnow_critic = frac * args.critic_learning_rate

            for network_id in range(args.num_networks):
                optimizers[network_id].param_groups[0]["lr"] = lrnow_actor
            critic_optimizer.param_groups[0]["lr"] = lrnow_critic

        if args.reset_on_phase_change and (not layout_phase_switched) and (global_step >= args.warmup_steps):
            print(f"[Phase Switch] global_step={global_step}: switching to wall layout")
            envs.set_layout(args.final_layout, reset_now=args.reset_on_phase_change)
            layout_phase_switched = True

            next_r_messages = torch.zeros((args.num_envs, num_agents), dtype=torch.int64, device=device)
            for i in range(num_agents):
                next_done[i] = torch.zeros(args.num_envs, device=device)
                next_lstm_state[i] = (
                    torch.zeros(agents[0].lstm.num_layers, args.num_envs, agents[0].lstm.hidden_size, device=device),
                    torch.zeros(agents[0].lstm.num_layers, args.num_envs, agents[0].lstm.hidden_size, device=device),
                )
            next_obs, next_locs, _ = envs._obs_core()

        # =========================
        # rollout
        # =========================
        for step in range(args.num_steps):
            global_step += args.num_envs

            action = {}
            s_message = {}
            action_logprob = {}
            message_logprob = {}

            # store centralized critic inputs
            joint_obs[step] = next_obs
            joint_locs[step] = next_locs
            joint_r_messages[step] = next_r_messages

            with torch.no_grad():
                shared_values[step] = critic(next_obs, next_locs, next_r_messages)

            for i in range(num_agents):
                obs[i][step] = next_obs[:, i, :, :, :]
                locs[i][step] = next_locs[:, i, :]
                r_messages[i][step] = next_r_messages[:, i]
                dones[i][step] = next_done[i]

                network_id = selected_networks[i]

                with torch.no_grad():
                    (
                        action[i],
                        action_logprob[i],
                        _,
                        s_message[i],
                        message_logprob[i],
                        _,
                        next_lstm_state[i],
                    ) = agents[network_id].get_action_and_logprob(
                        (next_obs[:, i, :, :, :], next_locs[:, i, :], next_r_messages[:, i]),
                        next_lstm_state[i],
                        next_done[i],
                    )

                actions[i][step] = action[i]
                s_messages[i][step] = s_message[i]
                action_logprobs[i][step] = action_logprob[i]
                message_logprobs[i][step] = message_logprob[i]

            acts_BA = torch.stack(
                [action[i].long().to(device) for i in range(num_agents)], dim=1
            )

            (next_obs, next_locs, msg_masks), all_rewards, all_terminations, all_truncations, infos = envs._step_core(acts_BA)

            if args.ablate_message:
                msg_masks = torch.zeros_like(msg_masks, dtype=torch.bool)

            pair_msg_mask = msg_masks.flatten(1).any(dim=1).to(torch.float32)
            done_batch = (all_terminations | all_truncations).float()

            for i in range(num_agents):
                message_masks[i][step] = pair_msg_mask
                next_r_messages[:, i] = pair_msg_mask.to(torch.int64) * s_message[swap_agent[i]]
                next_done[i] = done_batch
                rewards[i][step] = all_rewards[:, i]

            # centralized critic uses team reward
            team_rewards[step] = all_rewards.mean(dim=1)
            team_dones[step] = done_batch

            if (global_step // args.num_envs) % args.save_frequency == 0:
                for network_id in range(args.num_networks):
                    if args.load_pretrained:
                        saved_step = global_step + args.pretrained_global_step
                    else:
                        saved_step = global_step
                    save_path = os.path.join(save_dir, f"actor_{network_id}_step_{saved_step}.pt")
                    torch.save(agents[network_id].state_dict(), save_path)
                    print(f"Actor saved to {save_path}")

                critic_save_path = os.path.join(save_dir, f"critic_step_{saved_step}.pt")
                torch.save(critic.state_dict(), critic_save_path)
                print(f"Critic saved to {critic_save_path}")

            ep_ret += all_rewards
            ep_len += 1

            finished = (all_terminations | all_truncations).bool()
            if finished.any():
                b_idx = finished.nonzero(as_tuple=False).squeeze(1)
                finished_returns = ep_ret[b_idx].mean(dim=1).detach().cpu()
                finished_lengths = ep_len[b_idx].mean(dim=1).detach().cpu()

                next_r_messages[b_idx] = torch.zeros(
                    (b_idx.numel(), num_agents), dtype=torch.int64, device=device
                )

                sum_return_since_log += float(finished_returns.sum())
                sum_length_since_log += float(finished_lengths.sum())
                episodes_since_log += finished_returns.numel()

                ep_ret[b_idx] = 0
                ep_len[b_idx] = 0

            if args.visualize_loss and episodes_since_log >= LOG_EVERY_EPISODES:
                mean_ret_since_log = sum_return_since_log / episodes_since_log
                mean_len_since_log = sum_length_since_log / episodes_since_log

                writer.add_scalar("charts/episodic_return/", mean_ret_since_log, global_step)
                writer.add_scalar("charts/episodic_length/", mean_len_since_log, global_step)

                sum_return_since_log = 0.0
                sum_length_since_log = 0.0
                episodes_since_log = 0

        # =========================
        # centralized GAE
        # =========================
        with torch.no_grad():
            next_shared_value = critic(next_obs, next_locs, next_r_messages)

            advantages_shared = torch.zeros_like(team_rewards, device=device)
            lastgaelam = torch.zeros(args.num_envs, device=device)

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - team_dones[t]
                    nextvalues = next_shared_value
                else:
                    nextnonterminal = 1.0 - team_dones[t + 1]
                    nextvalues = shared_values[t + 1]

                delta = team_rewards[t] + args.gamma * nextvalues * nextnonterminal - shared_values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages_shared[t] = lastgaelam

            returns_shared = advantages_shared + shared_values

        # flatten critic batch
        b_joint_obs = joint_obs.reshape(
            args.batch_size, num_agents, num_channels, args.image_size, args.image_size
        )
        b_joint_locs = joint_locs.reshape(args.batch_size, num_agents, 2)
        b_joint_r_messages = joint_r_messages.reshape(args.batch_size, num_agents)
        b_advantages_shared = advantages_shared.reshape(-1)
        b_returns_shared = returns_shared.reshape(-1)
        b_shared_values = shared_values.reshape(-1)

        # flatten actor batches
        b_obs = {}
        b_locs = {}
        b_r_messages = {}
        b_action_logprobs = {}
        b_s_messages = {}
        b_message_logprobs = {}
        b_message_masks = {}
        b_actions = {}
        b_dones = {}
        tracks = {}

        for i in range(num_agents):
            b_obs[i] = obs[i].reshape((-1, num_channels, args.image_size, args.image_size))
            b_locs[i] = locs[i].reshape(-1, 2)
            b_r_messages[i] = r_messages[i].reshape(-1)
            b_action_logprobs[i] = action_logprobs[i].reshape(-1)
            b_s_messages[i] = s_messages[i].reshape(-1)
            b_message_logprobs[i] = message_logprobs[i].reshape(-1)
            b_message_masks[i] = message_masks[i].reshape(-1)
            b_actions[i] = actions[i].reshape(-1)
            b_dones[i] = dones[i].reshape(-1)
            tracks[i] = torch.tensor(
                np.array(
                    [int(str(env_i) + str(step_i)) for step_i in range(args.num_steps) for env_i in range(args.num_envs)]
                ),
                device=device,
            )

        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        # =========================
        # actor updates
        # =========================
        actor_stats = {}

        for i in range(num_agents):
            network_id = selected_networks[i]
            action_clipfracs = []
            message_clipfracs = []

            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()

                    (
                        new_action_logprob,
                        action_entropy,
                        new_message_logprob,
                        message_entropy,
                        _,
                    ) = agents[network_id].evaluate_actions(
                        (b_obs[i][mb_inds], b_locs[i][mb_inds], b_r_messages[i][mb_inds]),
                        (
                            initial_lstm_state[i][0][:, mbenvinds],
                            initial_lstm_state[i][1][:, mbenvinds],
                        ),
                        b_dones[i][mb_inds],
                        b_actions[i].long()[mb_inds],
                        b_s_messages[i].long()[mb_inds],
                        tracks[i][mb_inds],
                    )

                    action_logratio = new_action_logprob - b_action_logprobs[i][mb_inds]
                    action_ratio = action_logratio.exp()

                    message_logratio = new_message_logprob - b_message_logprobs[i][mb_inds]
                    message_ratio = message_logratio.exp()

                    with torch.no_grad():
                        old_action_approx_kl = (-action_logratio).mean()
                        action_approx_kl = ((action_ratio - 1) - action_logratio).mean()
                        action_clipfracs.append(
                            ((action_ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        )

                    mb_advantages = b_advantages_shared[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # action PPO loss
                    pg_loss1 = -mb_advantages * action_ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        action_ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # message PPO loss
                    mb_message_mask = b_message_masks[i][mb_inds]
                    mg_loss1 = -mb_advantages * message_ratio
                    mg_loss2 = -mb_advantages * torch.clamp(
                        message_ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    mg_loss_elem = torch.max(mg_loss1, mg_loss2)

                    valid_msg_count = mb_message_mask.sum().clamp_min(1.0)
                    mg_loss = (mg_loss_elem * mb_message_mask).sum() / valid_msg_count

                    with torch.no_grad():
                        old_message_approx_kl = ((-message_logratio) * mb_message_mask).sum() / valid_msg_count
                        message_approx_kl = (
                            (((message_ratio - 1) - message_logratio) * mb_message_mask).sum()
                            / valid_msg_count
                        )
                        msg_clip = ((message_ratio - 1.0).abs() > args.clip_coef).float()
                        message_clipfracs.append(
                            ((msg_clip * mb_message_mask).sum() / valid_msg_count).item()
                        )

                    action_entropy_loss = action_entropy.mean()
                    message_entropy_loss = (message_entropy * mb_message_mask).sum() / valid_msg_count

                    actor_loss = (
                        pg_loss
                        + mg_loss
                        - args.ent_coef * action_entropy_loss
                        - args.m_ent_coef * message_entropy_loss
                    )

                    optimizers[network_id].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(agents[network_id].parameters(), args.max_grad_norm)
                    optimizers[network_id].step()

                if args.target_kl is not None and action_approx_kl > args.target_kl:
                    break

            actor_stats[i] = {
                "network_id": network_id,
                "actor_loss": actor_loss.item(),
                "pg_loss": pg_loss.item(),
                "mg_loss": mg_loss.item(),
                "action_entropy_loss": action_entropy_loss.item(),
                "message_entropy_loss": message_entropy_loss.item(),
                "old_action_approx_kl": old_action_approx_kl.item(),
                "old_message_approx_kl": old_message_approx_kl.item(),
                "action_approx_kl": action_approx_kl.item(),
                "message_approx_kl": message_approx_kl.item(),
                "action_clipfrac": float(np.mean(action_clipfracs)) if len(action_clipfracs) > 0 else 0.0,
                "message_clipfrac": float(np.mean(message_clipfracs)) if len(message_clipfracs) > 0 else 0.0,
            }

        # =========================
        # critic updates
        # =========================
        critic_clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                new_values = critic(
                    b_joint_obs[mb_inds],
                    b_joint_locs[mb_inds],
                    b_joint_r_messages[mb_inds],
                ).view(-1)

                if args.clip_vloss:
                    v_loss_unclipped = (new_values - b_returns_shared[mb_inds]) ** 2
                    v_clipped = b_shared_values[mb_inds] + torch.clamp(
                        new_values - b_shared_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns_shared[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    with torch.no_grad():
                        critic_clipfracs.append(
                            ((new_values - b_shared_values[mb_inds]).abs() > args.clip_coef)
                            .float()
                            .mean()
                            .item()
                        )
                else:
                    v_loss = 0.5 * ((new_values - b_returns_shared[mb_inds]) ** 2).mean()

                critic_optimizer.zero_grad()
                (args.vf_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()

        # recompute value predictions for explained variance
        with torch.no_grad():
            final_value_pred = critic(
                b_joint_obs,
                b_joint_locs,
                b_joint_r_messages,
            ).view(-1)

        y_pred = final_value_pred.cpu().numpy()
        y_true = b_returns_shared.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.visualize_loss and (global_step // args.num_envs) % args.log_every == 0:
            for i in range(num_agents):
                network_id = actor_stats[i]["network_id"]
                writer.add_scalar(
                    f"agent{network_id}/charts/learning_rate",
                    optimizers[network_id].param_groups[0]["lr"],
                    global_step,
                )
                writer.add_scalar(f"agent{network_id}/losses/actor_loss", actor_stats[i]["actor_loss"], global_step)
                writer.add_scalar(f"agent{network_id}/losses/action_loss", actor_stats[i]["pg_loss"], global_step)
                writer.add_scalar(f"agent{network_id}/losses/message_loss", actor_stats[i]["mg_loss"], global_step)
                writer.add_scalar(
                    f"agent{network_id}/losses/action_entropy",
                    actor_stats[i]["action_entropy_loss"],
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/message_entropy",
                    actor_stats[i]["message_entropy_loss"],
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/old_action_approx_kl",
                    actor_stats[i]["old_action_approx_kl"],
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/old_message_approx_kl",
                    actor_stats[i]["old_message_approx_kl"],
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/action_approx_kl",
                    actor_stats[i]["action_approx_kl"],
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/message_approx_kl",
                    actor_stats[i]["message_approx_kl"],
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/action_clipfrac",
                    actor_stats[i]["action_clipfrac"],
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/message_clipfrac",
                    actor_stats[i]["message_clipfrac"],
                    global_step,
                )

            writer.add_scalar(
                "critic/charts/learning_rate",
                critic_optimizer.param_groups[0]["lr"],
                global_step,
            )
            writer.add_scalar("critic/losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar(
                "critic/losses/value_clipfrac",
                float(np.mean(critic_clipfracs)) if len(critic_clipfracs) > 0 else 0.0,
                global_step,
            )
            writer.add_scalar("critic/losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))

    if args.load_pretrained:
        saved_step = global_step + args.pretrained_global_step
    else:
        saved_step = global_step

    for network_id in range(args.num_networks):
        final_save_path = os.path.join(save_dir, f"actor_{network_id}_step_{saved_step}.pt")
        torch.save(agents[network_id].state_dict(), final_save_path)
        print(f"Final actor {network_id} saved to {final_save_path}")

    final_critic_save_path = os.path.join(save_dir, f"critic_step_{saved_step}.pt")
    torch.save(critic.state_dict(), final_critic_save_path)
    print(f"Final critic saved to {final_critic_save_path}")

    envs.close()
    writer.close()