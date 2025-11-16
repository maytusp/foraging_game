# train.py
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

from environments.torch_pickup_high_v1 import TorchForagingEnv, EnvConfig
from utils.process_data import *
from models.pickup_models import PPOTransformerCommAgent  # <-- changed import

# CUDA_VISIBLE_DEVICES=1 python -m scripts.torch_pickup.train_gpt_ppo
@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    # Algorithm specific arguments
    env_id: str = "Foraging-Single-v1"
    """the id of the environment"""
    total_timesteps: int = int(6e8)
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel game environments"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the action_entropy"""
    m_ent_coef: float = 0.001
    """coefficient of the message_entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    # Populations
    num_networks = 3
    reset_iteration: int = 1
    self_play_option: bool = False

    log_every = 32

    n_words = 4
    image_size = 3
    N_i = 2
    grid_size = 5
    max_steps = 10
    fully_visible_score = False
    agent_visible = False
    time_pressure = True
    mode = "train"
    model_name = f"ft_gpt_ppo_{num_networks}net"

    if not (agent_visible):
        model_name += "_invisible"

    if not (time_pressure):
        model_name += "_wospeedrw"

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    train_combination_name = f"grid{grid_size}_img{image_size}_ni{N_i}_nw{n_words}_ms{max_steps}"
    save_dir = f"checkpoints/torch_pickup_high_v1/{model_name}/{train_combination_name}/seed{seed}/"
    os.makedirs(save_dir, exist_ok=True)
    load_pretrained = False
    if load_pretrained:
        pretrained_global_step = 102400000
        learning_rate = 2e-4
        print(f"LOAD from {pretrained_global_step}")
        ckpt_path = {
            a: f"checkpoints/torch_pickup_high_v1/project_mem/gpt_ppo_3net_invisible/grid5_img3_ni2_nw4_ms10/seed1/agent_{a}_step_102400000.pt"
            for a in range(num_networks)
        }
    visualize_loss = True
    save_frequency = int(5e4)

    exp_name = f"{model_name}/{train_combination_name}_seed{seed}"
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "torch_pickup_high_v1"
    """the wandb's project name"""
    wandb_entity: str = "maytusp"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = args.exp_name
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    cfg = EnvConfig(
        grid_size=args.grid_size,
        image_size=args.image_size,
        num_agents=2,
        num_foods=args.N_i,
        num_walls=0,
        max_steps=args.max_steps,
        agent_visible=args.agent_visible,
        food_energy_fully_visible=args.fully_visible_score,
        mode=args.mode,
        seed=args.seed,
        time_pressure=args.time_pressure,
    )
    envs = TorchForagingEnv(cfg, device=device, num_envs=args.num_envs)
    num_agents = cfg.num_agents
    num_channels = 2

    num_actions = envs.num_actions

    # Initialize dicts for keeping agent models and experiences
    agents = {}
    optimizers = {}
    obs = {}
    locs = {}
    r_messages = {}
    actions = {}
    s_messages = {}
    action_logprobs = {}
    message_logprobs = {}
    rewards = {}
    dones = {}
    values = {}

    next_obs, next_locs, next_done, next_memory_state = {}, {}, {}, {}
    # TRY NOT TO MODIFY: start the game
    next_obs, next_locs = envs._obs_core()
    next_r_messages = torch.zeros((args.num_envs, num_agents), dtype=torch.int64).to(device)

    swap_agent = {0: 1, 1: 0}

    # --- create population of transformer agents ---
    for network_id in range(args.num_networks):
        agents[network_id] = PPOTransformerCommAgent(
            num_actions=num_actions,
            grid_size=args.grid_size,
            n_words=args.n_words,
            embedding_size=16,
            num_channels=num_channels,
            image_size=args.image_size,
            d_model=128,
            n_layers=2,
            n_heads=4,
        ).to(device)
        if args.load_pretrained:
            agents[network_id].load_state_dict(
                torch.load(args.ckpt_path[network_id], map_location=device),
                strict=True, 
            )
        optimizers[network_id] = optim.Adam(
            agents[network_id].parameters(), lr=args.learning_rate, eps=1e-5
        )

    # Storage tensors
    for i in range(num_agents):
        obs[i] = torch.zeros(
            (args.num_steps, args.num_envs, num_channels, args.image_size, args.image_size)
        ).to(device)
        locs[i] = torch.zeros((args.num_steps, args.num_envs, 2)).to(device)
        r_messages[i] = torch.zeros(
            (args.num_steps, args.num_envs), dtype=torch.int64
        ).to(device)
        actions[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        s_messages[i] = torch.zeros(
            (args.num_steps, args.num_envs), dtype=torch.int64
        ).to(device)
        action_logprobs[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        message_logprobs[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)

        next_done[i] = torch.zeros(args.num_envs).to(device)
        # memory state: [num_envs, d_model]
        next_memory_state[i] = agents[0].init_memory(args.num_envs, device)

    start_time = time.time()
    global_step = 0
    initial_memory_state = {}
    possible_networks = [i for i in range(args.num_networks)]
    selected_networks = [0, 1]

    # --- log performance ---
    episodes_since_log = 0
    sum_return_since_log = 0.0
    sum_length_since_log = 0.0
    successes_since_log = 0.0    # <--- NEW

    ep_ret = torch.zeros(args.num_envs, num_agents, device=device)
    ep_len = torch.zeros(args.num_envs, num_agents, device=device)

    LOG_EVERY_EPISODES = getattr(args, "log_every_episodes", args.num_envs)

    # Start training
    for iteration in range(1, args.num_iterations + 1):
        if iteration % args.reset_iteration == 0:
            # Reset memory state whenever we re-sample networks
            for i in range(num_agents):
                next_memory_state[i] = agents[0].init_memory(args.num_envs, device)
            selected_networks = np.random.choice(
                possible_networks, num_agents, replace=args.self_play_option
            )

        for i in range(num_agents):
            initial_memory_state[i] = next_memory_state[i].clone()

        if args.anneal_lr:
            for network_id in range(args.num_networks):
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizers[network_id].param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            action = {}
            s_message = {}
            action_logprob = {}
            message_logprob = {}
            value = {}

            for i in range(num_agents):
                obs[i][step] = next_obs[:, i, :, :, :]
                locs[i][step] = next_locs[:, i, :]
                r_messages[i][step] = next_r_messages[:, i].squeeze()
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
                        value[i],
                        next_memory_state[i],
                    ) = agents[network_id].get_action_and_value(
                        (next_obs[:, i, :, :, :], next_locs[:, i, :], next_r_messages[:, i]),
                        next_memory_state[i],
                        next_done[i],
                    )
                    values[i][step] = value[i].flatten()

                actions[i][step] = action[i]
                s_messages[i][step] = s_message[i]
                action_logprobs[i][step] = action_logprob[i]
                message_logprobs[i][step] = message_logprob[i]

            # Step env
            acts_BA = torch.stack(
                [action[i].long().to(device) for i in range(num_agents)], dim=1
            )  # [B, A]

            (next_obs, next_locs), all_rewards, all_terminations, all_truncations, infos = \
                envs._step_core(acts_BA)

            for i in range(num_agents):
                next_r_messages[:, i] = s_message[swap_agent[i]]
                next_done[i] = (all_terminations | all_truncations).float()
                rewards[i][step] = all_rewards[:, i]

            # Save checkpoints
            if (global_step // args.num_envs) % args.save_frequency == 0:
                for network_id in range(args.num_networks):
                    if args.load_pretrained:
                        saved_step = global_step + args.pretrained_global_step
                    else:
                        saved_step = global_step
                    save_path = os.path.join(
                        args.save_dir, f"agent_{network_id}_step_{saved_step}.pt"
                    )
                    torch.save(agents[network_id].state_dict(), save_path)
                    print(f"Model saved to {save_path}")

            # Logging returns
            ep_ret += all_rewards
            ep_len += 1

            finished = (all_terminations | all_truncations).bool()
            if finished.any():
                b_idx = finished.nonzero(as_tuple=False).squeeze(1)
                finished_returns = ep_ret[b_idx].mean(dim=1).detach().cpu()
                finished_lengths = ep_len[b_idx].mean(dim=1).detach().cpu()
                next_r_messages[b_idx] = torch.zeros(
                    (b_idx.numel(), num_agents), dtype=torch.int64
                ).to(device)

                sum_return_since_log += float(finished_returns.sum())
                sum_length_since_log += float(finished_lengths.sum())
                episodes_since_log += finished_returns.numel()
                # count successes: episodes with return >= 1.0
                successes_since_log += float((finished_returns >= 1.0).sum())
                ep_ret[b_idx] = 0
                ep_len[b_idx] = 0

            if args.visualize_loss and episodes_since_log >= LOG_EVERY_EPISODES:
                mean_ret_since_log = sum_return_since_log / episodes_since_log
                mean_len_since_log = sum_length_since_log / episodes_since_log
                success_rate_since_log = successes_since_log / episodes_since_log

                writer.add_scalar("charts/episodic_return/", mean_ret_since_log, global_step)
                writer.add_scalar("charts/episodic_length/", mean_len_since_log, global_step)
                writer.add_scalar("charts/success_rate/", success_rate_since_log, global_step)

                sum_return_since_log = 0.0
                sum_length_since_log = 0.0
                successes_since_log = 0.0
                episodes_since_log = 0

        # --- PPO update ---
        b_obs = {}
        b_locs = {}
        b_r_messages = {}
        b_action_logprobs = {}
        b_s_messages = {}
        b_message_logprobs = {}
        b_actions = {}
        b_dones = {}
        b_advantages = {}
        b_returns = {}
        b_values = {}
        tracks = {}
        advantages = {}
        returns = {}

        for i in range(num_agents):
            network_id = selected_networks[i]
            with torch.no_grad():
                next_value = agents[network_id].get_value(
                    (next_obs[:, i, :, :, :], next_locs[:, i, :], next_r_messages[:, i]),
                    next_memory_state[i],
                    next_done[i],
                ).reshape(1, -1)
                advantages[i] = torch.zeros_like(rewards[i]).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done[i]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[i][t + 1]
                        nextvalues = values[i][t + 1]
                    delta = (
                        rewards[i][t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[i][t]
                    )
                    advantages[i][t] = lastgaelam = (
                        delta
                        + args.gamma
                        * args.gae_lambda
                        * nextnonterminal
                        * lastgaelam
                    )
                returns[i] = advantages[i] + values[i]

            # Flatten the batch
            b_obs[i] = obs[i].reshape(
                (-1, num_channels, args.image_size, args.image_size)
            )
            b_locs[i] = locs[i].reshape(-1, 2)
            b_r_messages[i] = r_messages[i].reshape(-1)
            b_action_logprobs[i] = action_logprobs[i].reshape(-1)
            b_s_messages[i] = s_messages[i].reshape(-1)
            b_message_logprobs[i] = message_logprobs[i].reshape(-1)
            b_actions[i] = actions[i].reshape(-1)
            b_dones[i] = dones[i].reshape(-1)
            b_advantages[i] = advantages[i].reshape(-1)
            b_returns[i] = returns[i].reshape(-1)
            b_values[i] = values[i].reshape(-1)
            tracks[i] = torch.tensor(
                np.array(
                    [
                        int(str(i_) + str(j))
                        for j in range(args.num_steps)
                        for i_ in range(args.num_envs)
                    ]
                )
            )

            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

            action_clipfracs = []
            message_clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()

                    init_mem = initial_memory_state[i][mbenvinds]  # [B, d_model]

                    (
                        _,
                        new_action_logprob,
                        action_entropy,
                        _,
                        new_message_logprob,
                        message_entropy,
                        newvalue,
                        _,
                    ) = agents[network_id].get_action_and_value(
                        (b_obs[i][mb_inds], b_locs[i][mb_inds], b_r_messages[i][mb_inds]),
                        init_mem,
                        b_dones[i][mb_inds],
                        b_actions[i].long()[mb_inds],
                        b_s_messages[i].long()[mb_inds],
                        tracks[i][mb_inds],
                    )

                    action_logratio = new_action_logprob - b_action_logprobs[i][mb_inds]
                    action_ratio = action_logratio.exp()

                    message_logratio = (
                        new_message_logprob - b_message_logprobs[i][mb_inds]
                    )
                    message_ratio = message_logratio.exp()

                    with torch.no_grad():
                        old_action_approx_kl = (-action_logratio).mean()
                        action_approx_kl = (
                            (action_ratio - 1) - action_logratio
                        ).mean()
                        action_clipfracs += [
                            ((action_ratio - 1.0).abs() > args.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                        old_message_approx_kl = (-message_logratio).mean()
                        message_approx_kl = (
                            (message_ratio - 1) - message_logratio
                        ).mean()
                        message_clipfracs += [
                            ((message_ratio - 1.0).abs() > args.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[i][mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Action policy loss
                    pg_loss1 = -mb_advantages * action_ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        action_ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Message policy loss
                    mg_loss1 = -mb_advantages * message_ratio
                    mg_loss2 = -mb_advantages * torch.clamp(
                        message_ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    mg_loss = torch.max(mg_loss1, mg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[i][mb_inds]) ** 2
                        v_clipped = b_values[i][mb_inds] + torch.clamp(
                            newvalue - b_values[i][mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[i][mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = (
                            0.5 * ((newvalue - b_returns[i][mb_inds]) ** 2).mean()
                        )

                    action_entropy_loss = action_entropy.mean()
                    message_entropy_loss = message_entropy.mean()
                    loss = (
                        pg_loss
                        + mg_loss
                        - (args.ent_coef * action_entropy_loss)
                        - (args.m_ent_coef * message_entropy_loss)
                        + v_loss * args.vf_coef
                    )

                    optimizers[network_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        agents[network_id].parameters(), args.max_grad_norm
                    )
                    optimizers[network_id].step()

                if args.target_kl is not None and action_approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values[i].cpu().numpy(), b_returns[i].cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            if args.visualize_loss and (global_step // args.num_envs) % args.log_every == 0:
                writer.add_scalar(
                    f"agent{network_id}/charts/learning_rate",
                    optimizers[network_id].param_groups[0]["lr"],
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/value_loss", v_loss.item(), global_step
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/action_loss", pg_loss.item(), global_step
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/message_loss", mg_loss.item(), global_step
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/action_entropy",
                    action_entropy_loss.item(),
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/message_entropy",
                    message_entropy_loss.item(),
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/old_action_approx_kl",
                    old_action_approx_kl.item(),
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/old_message_approx_kl",
                    old_message_approx_kl.item(),
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/action_approx_kl",
                    action_approx_kl.item(),
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/message_approx_kl",
                    message_approx_kl.item(),
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/action_clipfrac",
                    np.mean(action_clipfracs),
                    global_step,
                )
                writer.add_scalar(
                    f"agent{network_id}/losses/message__clipfrac",
                    np.mean(message_clipfracs),
                    global_step,
                )
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                print("SPS:", int(global_step / (time.time() - start_time)))

    if args.load_pretrained:
        saved_step = global_step + args.pretrained_global_step
    else:
        saved_step = global_step
    for network_id in range(args.num_networks):
        final_save_path = os.path.join(
            args.save_dir, f"agent_{network_id}_step_{saved_step}.pt"
        )
        torch.save(agents[network_id].state_dict(), final_save_path)
        print(f"Final model of network {network_id} saved to {final_save_path}")

    envs.close()
    writer.close()
