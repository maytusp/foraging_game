# Edit: 20Dec2024
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import supersuit as ss


from environment_pickup_high import *
from utils import *
# from models import PPOLSTMAgent, PPOLSTMCommAgent
from models_v2 import PPOLSTMAgent, PPOLSTMCommAgent


@dataclass
class Args:
    save_dir = "checkpoints/ppo_ps_comm_v2_pickup_high_stage2"
    os.makedirs(save_dir, exist_ok=True)
    load_pretrained = True
    ckpt_path = "checkpoints/ppo_ps_comm_v2_pickup_high_stage1/final_model.pt"
    save_frequency = int(1e5)
    # exp_name: str = os.path.basename(__file__)[: -len(".py")]
    exp_name = "ppo_ps_comm_stage2"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "31_pickup_high"
    """the wandb's project name"""
    wandb_entity: str = "maytusp"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    fully_visible_score = True
    """Fully visible food highest score for pretraining"""
    # Algorithm specific arguments
    env_id: str = "Foraging-Single-v1"
    """the id of the environment"""
    total_timesteps: int = int(1e8)
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 4
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
    ent_coef: float = 0.03 # ori 0.01
    """coefficient of the action_entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """number of action"""
    num_channels = 2
    """number of channels in observation (non rgb case)"""
    num_obs_grid = 5
    """number of observation grid"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""



if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    env = Environment(use_message=True, food_ener_fully_visible=args.fully_visible_score)
    grid_size = (env.image_size, env.image_size)
    num_channels = env.num_channels
    num_agents = len(env.possible_agents)
    # print("env.action_space(env.possible_agents[0])", env.action_space(env.possible_agents[0]))
    num_actions = env.action_space(env.possible_agents[0])['action'].n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    # Vectorise env
    envs = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(envs, args.num_envs // num_agents, num_cpus=0, base_class="gymnasium")


    agent = PPOLSTMCommAgent(num_actions=num_actions, num_channels=args.num_channels).to(device)
    if args.load_pretrained:
        agent.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, args.num_channels, args.num_obs_grid, args.num_obs_grid)).to(device) # obs: vision
    locs = torch.zeros((args.num_steps, args.num_envs, 2)).to(device) # obs: location
    eners = torch.zeros((args.num_steps, args.num_envs, 1)).to(device) # obs: energy
    r_messages = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64).to(device) # obs: received message
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device) # action: physical action
    s_messages = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64).to(device) # action: sent message
    action_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    message_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_dict, _ = envs.reset(seed=args.seed)
    next_obs, next_locs, next_eners, next_r_messages = extract_dict(next_obs_dict, device, use_message=True)
    next_r_messages = torch.tensor(next_r_messages).squeeze().to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            locs[step] = next_locs
            eners[step] = next_eners
            r_messages[step] = next_r_messages.squeeze()
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, action_logprob, _, s_message, message_logprob, _, value, next_lstm_state = agent.get_action_and_value((next_obs, next_locs, next_eners, next_r_messages), 
                                                                                                    next_lstm_state, next_done)
                values[step] = value.flatten()

            # print(f"action {action.shape}")
            # print(f"s_message {s_message.shape}")
            actions[step] = action
            s_messages[step] = s_message
            action_logprobs[step] = action_logprob
            message_logprobs[step] = message_logprob
            

            # TRY NOT TO MODIFY: execute the game and log data.
            env_action, env_message = action.cpu().numpy(), s_message.cpu().numpy()
            next_obs_dict, reward, terminations, truncations, infos = envs.step({"action": env_action, "message": env_message})
            next_obs, next_locs, next_eners, next_r_messages = extract_dict(next_obs_dict, device, use_message=True)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_r_messages = torch.tensor(next_r_messages).to(device)

            if (global_step // args.num_envs) % args.save_frequency == 0:  # Adjust `save_frequency` as needed
                save_path = os.path.join(args.save_dir, f"model_step_{global_step}.pt")
                torch.save(agent.state_dict(), save_path)
                print(f"Model saved to {save_path}")

            for info in infos:
                if "terminal_observation" in info:
                    # for info in each_infos:
                    if "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                (next_obs, next_locs, next_eners, next_r_messages),
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, args.num_channels, args.num_obs_grid, args.num_obs_grid))
        b_locs = locs.reshape(-1, 2)
        b_eners = eners.reshape(-1, 1)
        b_r_messages = r_messages.reshape(-1)
        b_action_logprobs = action_logprobs.reshape(-1)
        b_s_messages = s_messages.reshape(-1)
        b_message_logprobs = message_logprobs.reshape(-1)
        b_actions = actions.reshape((-1))
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
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
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, new_action_logprob, action_entropy, _, new_message_logprob, message_entropy, newvalue, _ = agent.get_action_and_value(
                    (b_obs[mb_inds], b_locs[mb_inds], b_eners[mb_inds], b_r_messages[mb_inds]),
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                    b_s_messages.long()[mb_inds],
                )
                action_logratio = new_action_logprob - b_action_logprobs[mb_inds]
                action_ratio = action_logratio.exp()

                message_logratio = new_message_logprob - b_message_logprobs[mb_inds]
                message_ratio = message_logratio.exp()

                with torch.no_grad():
                    # calculate action_approx_kl http://joschu.net/blog/kl-approx.html
                    old_action_approx_kl = (-action_logratio).mean()
                    action_approx_kl = ((action_ratio - 1) - action_logratio).mean()
                    action_clipfracs += [((action_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    old_message_approx_kl = (-message_logratio).mean()
                    message_approx_kl = ((message_ratio - 1) - message_logratio).mean()
                    message_clipfracs += [((message_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
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
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                action_entropy_loss = action_entropy.mean()
                message_entropy_loss = message_entropy.mean()
                loss = pg_loss + mg_loss - args.ent_coef * (action_entropy_loss+message_entropy_loss) + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and action_approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/action_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/message_loss", mg_loss.item(), global_step)
        writer.add_scalar("losses/action_entropy", action_entropy_loss.item(), global_step)
        writer.add_scalar("losses/message_entropy", message_entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_action_approx_kl", old_action_approx_kl.item(), global_step)
        writer.add_scalar("losses/old_message_approx_kl", old_message_approx_kl.item(), global_step)
        writer.add_scalar("losses/action_approx_kl", action_approx_kl.item(), global_step)
        writer.add_scalar("losses/message_approx_kl", message_approx_kl.item(), global_step)
        writer.add_scalar("losses/action_clipfrac", np.mean(action_clipfracs), global_step)
        writer.add_scalar("losses/message__clipfrac", np.mean(message_clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    final_save_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(agent.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")
    envs.close()
    writer.close()