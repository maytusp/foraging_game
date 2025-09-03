# Created: 24 Aug 2025
# The code is for training agents with separated networks during training and execution (no parameter sharing)
# Fully Decentralise Training and Decentralise Execution
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


from environments.torch_pickup_high_v1 import TorchForagingEnv, EnvConfig
from utils.process_data import *
from utils.process_env import *
from models.pickup_models import PPOLSTMCommAgent

from utils.graph_gen import WS_PAIRS
# CUDA_VISIBLE_DEIVCES=1 python -m scripts.torch_pickup.train_wsnet
@dataclass
class Args:
    seed: int = 3
    """seed of the experiment"""
    # Algorithm specific arguments
    env_id: str = "Foraging-Single-v1"
    """the id of the environment"""
    total_timesteps: int = int(1e9)
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4 * 64
    """the learning rate of the optimizer"""
    num_envs: int = 1024 * 8
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
    m_ent_coef: float = 0.002
    """coefficient of the message_entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    # Populations
    num_networks = 15
    reset_iteration: int = 1
    self_play_option: bool = True
    
    """
    By default, agent0 and agent1 uses network0 and network1
    However, agent0 will speak the language that itself cannot understand
    so we have to randomnly picked networks for agent0 and agent1
    For example,
    episode1: we pick [n0, n1]
    episode2: we pick [n0, n0]
    episode3: we pick [n1, n1]
    """

    log_every = 32

    n_words = 4
    image_size = 3
    N_i = 2
    grid_size = 5
    max_steps = 10
    fully_visible_score = False
    agent_visible = False
    mode = "train"
    model_name = "wsk4p02_sp_ppo_15net"
    
    if not(agent_visible):
        model_name+= "_invisible"
    

    """train or test (different attribute combinations)"""
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    train_combination_name = f"grid{grid_size}_img{image_size}_ni{N_i}_nw{n_words}_ms{max_steps}"
    save_dir = f"checkpoints/torch_pickup_high_v1/{model_name}/{train_combination_name}/seed{seed}/"
    os.makedirs(save_dir, exist_ok=True)
    load_pretrained = False
    if load_pretrained:
        pretrained_global_step = 51200000
        learning_rate = 2e-4
        print(f"LOAD from {pretrained_global_step}")
        ckpt_path = {
                    a: f"" for a in range(num_networks)
                    }
    visualize_loss = True
    save_frequency = int(2e5 / (num_envs/128))
    # exp_name: str = os.path.basename(__file__)[: -len(".py")]
    
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
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
        num_agents=2,               # keep your setting
        num_foods=args.N_i,
        num_walls=0,
        max_steps=args.max_steps,
        use_message=True,           # we’ll feed/receive messages
        agent_visible=args.agent_visible,
        food_energy_fully_visible=args.fully_visible_score,
        mode=args.mode,
        seed=args.seed,
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

    next_obs, next_locs, next_r_messages, next_done, next_lstm_state = {}, {}, {}, {}, {}
    # TRY NOT TO MODIFY: start the game
    next_obs, next_locs, next_r_messages = envs._obs_core()
    next_r_messages = next_r_messages.squeeze()
    
    swap_agent = {0:1, 1:0}

    for net_id in range(args.num_networks):
        agents[net_id] = PPOLSTMCommAgent(num_actions=num_actions, 
                                    grid_size=args.grid_size, 
                                    n_words=args.n_words, 
                                    embedding_size=16, 
                                    num_channels=num_channels, 
                                    image_size=args.image_size).to(device)
        if args.load_pretrained:
            agents[net_id].load_state_dict(torch.load(args.ckpt_path[net_id], map_location=device))
        optimizers[net_id] = optim.Adam(agents[net_id].parameters(), lr=args.learning_rate, eps=1e-5)

    for i in range(num_agents):
        # ALGO Logic: Storage setup
        obs[i] = torch.zeros((args.num_steps, args.num_envs, num_channels, args.image_size, args.image_size)).to(device) # obs: vision
        locs[i] = torch.zeros((args.num_steps, args.num_envs, 2)).to(device) # obs: location
        r_messages[i] = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64).to(device) # obs: received message
        actions[i] = torch.zeros((args.num_steps, args.num_envs)).to(device) # action: physical action
        s_messages[i] = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64).to(device) # action: sent message
        action_logprobs[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        message_logprobs[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values[i] = torch.zeros((args.num_steps, args.num_envs)).to(device)

        next_done[i] = torch.zeros(args.num_envs).to(device)
        next_lstm_state[i] = (
            torch.zeros(agents[0].lstm.num_layers, args.num_envs, agents[0].lstm.hidden_size).to(device),
            torch.zeros(agents[0].lstm.num_layers, args.num_envs, agents[0].lstm.hidden_size).to(device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    
    start_time = time.time()
    global_step = 0
    initial_lstm_state = {}
    possible_pairs = WS_PAIRS
    if args.self_play_option:
        possible_pairs += [[a,a] for a in range(args.num_networks)]
        print("Enable self-play")
        print(f"Pairs {possible_pairs}")
    pair_cursor = 0  # rotation pointer across iterations
    pair_ids = None  # [B, 2], filled per-iteration


    # --- log performance ---
    episodes_since_log = 0
    sum_return_since_log = 0.0
    sum_length_since_log = 0.0

    # per-env accumulators
    ep_ret = torch.zeros(args.num_envs, num_agents, device=device)
    ep_len = torch.zeros(args.num_envs, num_agents, device=device)

    episodes_since_log = 0
    LOG_EVERY_EPISODES = getattr(args, "log_every_episodes", args.num_envs)  # tune as you like
    # Start training
    for iteration in range(1, args.num_iterations + 1):
        # print("iteration", iteration)
        if iteration % args.reset_iteration == 0:
            # reset LSTM states because network assignments change
            for i in range(num_agents):
                next_lstm_state[i] = (
                    torch.zeros(agents[0].lstm.num_layers, args.num_envs, agents[0].lstm.hidden_size).to(device),
                    torch.zeros(agents[0].lstm.num_layers, args.num_envs, agents[0].lstm.hidden_size).to(device),
                )
            # assign a pair (netA, netB) per environment
            pair_ids, pair_cursor = assign_pairs_per_env(args.num_envs, possible_pairs, cursor=pair_cursor)
            pair_ids = pair_ids.to(device)  # [B, 2] on device


        for i in range(num_agents):
            initial_lstm_state[i] = (next_lstm_state[i][0].clone(), next_lstm_state[i][1].clone())

        if args.anneal_lr:
            for net_id in range(args.num_networks):
                # Annealing the rate if instructed to do so.
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizers[net_id].param_groups[0]["lr"] = lrnow


        for step in range(0, args.num_steps):
            global_step += args.num_envs

            action = {}
            s_message = {}
            action_logprob = {}
            message_logprob = {}
            value = {}
            reward = {}

            for i in range(num_agents):
                obs[i][step] = next_obs[:,i,:,:,:]
                locs[i][step] = next_locs[:,i,:]
                r_messages[i][step] = next_r_messages[:, i].squeeze()
                dones[i][step] = next_done[i]
                
            action = {0: torch.empty(args.num_envs, device=device, dtype=torch.long),
                    1: torch.empty(args.num_envs, device=device, dtype=torch.long)}
            s_message = {0: torch.empty(args.num_envs, device=device, dtype=torch.long),
                        1: torch.empty(args.num_envs, device=device, dtype=torch.long)}
            action_logprob, message_logprob, value = {}, {}, {}
            for i in range(num_agents):
                action_logprob[i]  = torch.empty(args.num_envs, device=device)
                message_logprob[i] = torch.empty(args.num_envs, device=device)
                value[i]           = torch.empty(args.num_envs, device=device)

            for net_id in range(args.num_networks):
                for i in (0, 1):  # agent slot
                    env_idx = (pair_ids[:, i] == net_id).nonzero(as_tuple=False).squeeze(-1)
                    if env_idx.numel() == 0:
                        continue

                    with torch.no_grad():
                        a, a_logp, _, m, m_logp, _, v, lstm_next = agents[net_id].get_action_and_value(
                            (next_obs[env_idx, i, :, :, :],
                            next_locs[env_idx, i, :],
                            next_r_messages[env_idx, i]),
                            (next_lstm_state[i][0][:, env_idx, :], next_lstm_state[i][1][:, env_idx, :]),
                            next_done[i][env_idx],
                        )

                    # scatter results back
                    action[i][env_idx]            = a
                    s_message[i][env_idx]         = m
                    action_logprob[i][env_idx]    = a_logp
                    message_logprob[i][env_idx]   = m_logp
                    value[i][env_idx]             = v.flatten()

                    # update the slot's LSTM state at those env indices
                    next_lstm_state[i][0][:, env_idx, :] = lstm_next[0]
                    next_lstm_state[i][1][:, env_idx, :] = lstm_next[1]

            # store rollout tensors
            for i in (0, 1):
                actions[i][step]          = action[i]
                s_messages[i][step]       = s_message[i]
                action_logprobs[i][step]  = action_logprob[i]
                message_logprobs[i][step] = message_logprob[i]
                values[i][step]           = value[i]

            # TRY NOT TO MODIFY: execute the game and log data.
            # --- build [B,A] tensors and step env ONCE on GPU ---
            acts_BA = torch.stack([action[i].long().to(device) for i in range(num_agents)], dim=1)  # [B,A]

            (next_obs, next_locs, next_msgs), all_rewards, all_terminations, all_truncations, infos =  envs._step_core(acts_BA)
            env_info = (all_rewards, all_terminations, all_truncations)

            for i in range(num_agents):
                next_r_messages[:,i] = next_msgs[:, swap_agent[i]] # agent exchange msgs
                next_done[i] = (all_terminations | all_truncations).float()
                rewards[i][step] = all_rewards[:, i] # (B,A)

            # Save Model Checkpoints: loop over networks not agents
            if (global_step // args.num_envs) % args.save_frequency == 0:  # Adjust `save_frequency` as needed
                for network_id in range(args.num_networks):
                    if args.load_pretrained:
                        saved_step = global_step + args.pretrained_global_step
                    else:
                        saved_step = global_step
                    save_path = os.path.join(args.save_dir, f"agent_{network_id}_step_{saved_step}.pt")
                    torch.save(agents[network_id].state_dict(), save_path)
                    print(f"Model saved to {save_path}")


            # For logging: accumulate returns/lengths
            ep_ret += all_rewards          # shape [B, A]
            ep_len += 1

            finished = (all_terminations | all_truncations).bool()   # shape [B]
            if finished.any():
                b_idx = finished.nonzero(as_tuple=False).squeeze(1)
                finished_returns = ep_ret[b_idx].mean(dim=1).detach().cpu()   # team-mean per env
                finished_lengths = ep_len[b_idx].mean(dim=1).detach().cpu()


                # exact aggregation since last log
                sum_return_since_log += float(finished_returns.sum())
                sum_length_since_log += float(finished_lengths.sum())
                episodes_since_log += finished_returns.numel()

                # reset accumulators for those finished envs
                ep_ret[b_idx] = 0
                ep_len[b_idx] = 0

            # Log periodically (episode-based cadence)
            if args.visualize_loss and episodes_since_log >= LOG_EVERY_EPISODES and len(recent_returns) > 0:
                mean_ret_since_log = sum_return_since_log / episodes_since_log
                mean_len_since_log = sum_length_since_log / episodes_since_log

                writer.add_scalar("charts/episodic_return/", mean_ret_since_log, global_step)
                writer.add_scalar("charts/episodic_length/", mean_len_since_log, global_step)

                sum_return_since_log = 0.0
                sum_length_since_log = 0.0
                episodes_since_log = 0
                        
        #TODO Implement seprate network for this part
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
        advantages, returns = {0: None, 1: None}, {0: None, 1: None}
        for i in range(num_agents):
            advantages[i] = torch.zeros_like(rewards[i], device=device)  # [T, B]
            with torch.no_grad():
                for net_id in range(args.num_networks): 
                    env_idx = (pair_ids[:, i] == net_id).nonzero(as_tuple=False).squeeze(-1)
                    if env_idx.numel() == 0:
                        continue
                    # bootstrap value if not done
                    
                    next_value = agents[net_id].get_value(
                        (next_obs[env_idx, i, :, :, :],
                        next_locs[env_idx, i, :],
                        next_r_messages[env_idx, i]),
                        (next_lstm_state[i][0][:, env_idx, :], next_lstm_state[i][1][:, env_idx, :]),
                        next_done[i][env_idx],
                    ).reshape(1, -1)  # [1, |S|]
                    
                    lastgaelam = torch.zeros_like(next_done[i][env_idx], device=device)

                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done[i][env_idx]
                            nextvalues = next_value.squeeze(0)
                        else:
                            nextnonterminal = 1.0 - dones[i][t + 1, env_idx]
                            nextvalues = values[i][t + 1, env_idx]
                        delta = rewards[i][t, env_idx] + args.gamma * nextvalues * nextnonterminal - values[i][t, env_idx]
                        advantages[i][t, env_idx] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

            returns[i] = advantages[i]+ values[i]

        # flatten (keep exactly as you had, per slot i=0/1)
        b_obs, b_locs, b_r_messages = {}, {}, {}
        b_action_logprobs, b_s_messages, b_message_logprobs = {}, {}, {}
        b_actions, b_dones, b_advantages, b_returns, b_values = {}, {}, {}, {}, {}
        tracks = {}
        for i in (0, 1):
            b_obs[i]              = obs[i].reshape((-1, num_channels, args.image_size, args.image_size))
            b_locs[i]             = locs[i].reshape(-1, 2)
            b_r_messages[i]       = r_messages[i].reshape(-1)
            b_action_logprobs[i]  = action_logprobs[i].reshape(-1)
            b_s_messages[i]       = s_messages[i].reshape(-1)
            b_message_logprobs[i] = message_logprobs[i].reshape(-1)
            b_actions[i]          = actions[i].reshape(-1)
            b_dones[i]            = dones[i].reshape(-1)
            b_advantages[i]       = advantages[i].reshape(-1)
            b_returns[i]          = returns[i].reshape(-1)
            b_values[i]           = values[i].reshape(-1)
            # Keep your "tracks" scheme
            tracks[i] = (
                torch.arange(args.num_steps, device=device).unsqueeze(1) * args.num_envs
                + torch.arange(args.num_envs, device=device).unsqueeze(0)
            ).reshape(-1)

        # indices helper
        flatinds    = torch.arange(args.batch_size).reshape(args.num_steps, args.num_envs).to(device)
        for i in (0, 1):
            net_metrics = []
            for net_id in range(args.num_networks):
            
                envinds = (pair_ids[:, i] == net_id).nonzero(as_tuple=False).squeeze(-1).to(device)
                if envinds.numel() == 0:
                    continue

                # minibatches over ONLY those envs
                envsperbatch = max(1, math.ceil(len(envinds) / args.num_minibatches))
                action_clipfracs, message_clipfracs = [], []
                for epoch in range(args.update_epochs):
                    perm = torch.randperm(envinds.numel(), device=envinds.device)
                    envinds = envinds[perm]

                    for start in range(0, len(envinds), envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                        _, new_action_logprob, action_entropy, _, new_message_logprob, message_entropy, newvalue, _ = agents[net_id].get_action_and_value(
                            (b_obs[i][mb_inds], b_locs[i][mb_inds], b_r_messages[i][mb_inds]),
                            (initial_lstm_state[i][0][:, mbenvinds], initial_lstm_state[i][1][:, mbenvinds]),
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
                            # calculate action_approx_kl http://joschu.net/blog/kl-approx.html
                            old_action_approx_kl = (-action_logratio).mean()
                            action_approx_kl = ((action_ratio - 1) - action_logratio).mean()
                            action_clipfracs += [((action_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                            old_message_approx_kl = (-message_logratio).mean()
                            message_approx_kl = ((message_ratio - 1) - message_logratio).mean()
                            message_clipfracs += [((message_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[i][mb_inds]
                        if args.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * action_ratio
                        pg_loss2 = -mb_advantages * torch.clamp(action_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Message loss
                        mg_loss1 = -mb_advantages * message_ratio
                        mg_loss2 = -mb_advantages * torch.clamp(message_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
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
                            v_loss = 0.5 * ((newvalue - b_returns[i][mb_inds]) ** 2).mean()

                        action_entropy_loss = action_entropy.mean()
                        message_entropy_loss = message_entropy.mean()
                        loss = pg_loss + mg_loss - (args.ent_coef * action_entropy_loss) - (args.m_ent_coef * message_entropy_loss) + v_loss * args.vf_coef

                        optimizers[net_id].zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agents[net_id].parameters(), args.max_grad_norm)
                        optimizers[net_id].step()

                    if args.target_kl is not None and action_approx_kl > args.target_kl:
                        break

                y_pred, y_true = b_values[i].cpu().numpy(), b_returns[i].cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                # stash for logging
                net_metrics.append({
                    "v_loss": v_loss.item(),
                    "pg_loss": pg_loss.item(),
                    "mg_loss": mg_loss.item(),
                    "a_ent": action_entropy_loss.item(),
                    "m_ent": message_entropy_loss.item(),
                    "old_a_kl": old_action_approx_kl.item(),
                    "old_m_kl": old_message_approx_kl.item(),
                    "a_kl": action_approx_kl.item(),
                    "m_kl": message_approx_kl.item(),
                    "a_clipfrac": float(np.mean(action_clipfracs)),
                    "m_clipfrac": float(np.mean(message_clipfracs)),
                })

                # per-net logging (optional)
                if args.visualize_loss and (global_step // args.num_envs) % args.log_every == 0 and net_metrics:
                    agg = {k: float(np.mean([m[k] for m in net_metrics])) for k in net_metrics[0].keys()}
                    writer.add_scalar(f"agent{net_id}/charts/learning_rate", optimizers[net_id].param_groups[0]["lr"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/value_loss", agg["v_loss"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/action_loss", agg["pg_loss"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/message_loss", agg["mg_loss"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/action_entropy", agg["a_ent"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/message_entropy", agg["m_ent"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/old_action_approx_kl", agg["old_a_kl"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/old_message_approx_kl", agg["old_m_kl"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/action_approx_kl", agg["a_kl"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/message_approx_kl", agg["m_kl"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/action_clipfrac", agg["a_clipfrac"], global_step)
                    writer.add_scalar(f"agent{net_id}/losses/message__clipfrac", agg["m_clipfrac"], global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))

    for net_id in range(args.num_networks):
        final_save_path = os.path.join(args.save_dir, f"final_model_agent_{net_id}.pt")
        torch.save(agents[net_id].state_dict(), final_save_path)
        print(f"Final model of network {net_id} saved to {final_save_path}")

    envs.close()
    writer.close()