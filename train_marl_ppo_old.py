"""Basic code which shows what it's like to run PPO on the Pistonball env using the parallel API, this code is inspired by CleanRL.

This code is exceedingly basic, with no logging or weights saving.
The intention was for users to have a (relatively clean) ~200 line file to refer to when they want to design their own learning algorithm.

Author: Jet (https://github.com/jjshoots)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import supersuit as ss

from nets import *
from constants import *
from keyboard_control import *
from environment import *
from buffer import *

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOLSTMAgent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.nonlinear = nn.Sequential(nn.Flatten(), # (1,5,5) to (25)
                                        layer_init(nn.Linear(25, 256)), 
                                        nn.ReLU(),
                                        layer_init(nn.Linear(256, 256)),
                                        nn.ReLU(),
                                        layer_init(nn.Linear(256, 256)),
                                        nn.ReLU(),
                                        layer_init(nn.Linear(256, 256)),
                                        nn.ReLU(),
                                        )       
        self.lstm = nn.LSTM(256, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        # hidden = self.nonlinear(self.network(x / 255.0)) # CNN
        hidden = self.nonlinear(x / 255.0) # MLP

        # LSTM logic
        batch_size = lstm_state[0].shape[1]

        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            d_adjust =  (1.0 - d).view(1, -1, 1)
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    d_adjust * lstm_state[0],
                    d_adjust * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


def batchify_obs(obs_dict, device):
    #TODO next_obs, next_locs, next_eners = next_obs_dict["image"], next_obs_dict["location"], next_obs_dict["energy"]
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays

    obs = np.array([a for a in obs_dict['image']])
    locs = np.array([a for a in obs_dict['location']])
    eners = np.array([a for a in obs_dict['energy']])

    # convert to torch
    obs = torch.tensor(obs).to(device)
    locs = torch.tensor(locs).to(device)
    eners = torch.tensor(eners).to(device)
    return obs, locs, eners



def unbatchify(x, num_envs, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()

    # x = {i: x[i] for i in range(num_envs)}
    x = {i:{j:0 for j in range(2)} for i in range(8)}

    return x


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 16
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 64
    max_cycles = 125
    total_episodes = 2
    update_epochs=4
    seed=1

    """ ENV SETUP """
    env = Environment()
    grid_size = (env.image_size, env.image_size)
    num_channels = env.num_channels
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    # Vectorise env
    # env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs // num_agents, num_cpus=0, base_class="gymnasium")
    
    
    

    """ LEARNER SETUP """
    agent = PPOLSTMAgent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_envs, num_channels, *grid_size)).to(device)
    rb_actions = torch.zeros((max_cycles, num_envs)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_envs)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_envs)).to(device)
    rb_terms = torch.zeros((max_cycles, num_envs)).to(device)
    rb_values = torch.zeros((max_cycles, num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    terms = torch.zeros(num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, num_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs_dict, _ = env.reset()
            
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                obs, locs, eners = batchify_obs(next_obs_dict, device)
                actions, logprobs, _, values, next_lstm_state = agent.get_action_and_value(obs, next_lstm_state, terms)

                # get action from the agent
                # execute the environment and log data
                # next_obs_dict, rewards, terms, truncs, infos = env.step(
                #     unbatchify(actions, num_envs, env)
                # )
                next_obs_dict, rewards, terms, truncs, infos = env.step(
                    actions.cpu().numpy()
                )

                terms = torch.tensor(terms).to(device)
                rewards = torch.Tensor(rewards).to(device)

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = rewards
                rb_terms[step] = terms
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()


                
                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([done for done in terms]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values
        print("end_step", end_step)
        print("rb_obs", rb_obs.shape)
        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)
        b_terms = torch.flatten(rb_terms[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        print("b_obs", b_obs.shape)
        b_index = np.arange(len(b_obs))


        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    # """ RENDER THE POLICY """
    # env = pistonball_v6.parallel_env(render_mode="human", continuous=False)
    # env = color_reduction_v0(env)
    # env = resize_v1(env, 64, 64)
    # env = frame_stack_v1(env, stack_size=4)

    # agent.eval()

    # with torch.no_grad():
    #     # render 5 episodes out
    #     for episode in range(5):
    #         obs, infos = env.reset(seed=None)
    #         obs = batchify_obs(obs, device)
    #         terms = [False]
    #         truncs = [False]
    #         while not any(terms) and not any(truncs):
    #             actions, logprobs, _, values = agent.get_action_and_value(obs)
    #             obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
    #             obs = batchify_obs(obs, device)
    #             terms = [terms[a] for a in terms]
    #             truncs = [truncs[a] for a in truncs]