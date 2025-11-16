import numpy as np
import torch

def extract_dict(obs_dict, device, use_message=False):
    obs, locs, eners = obs_dict["image"], obs_dict["location"], obs_dict["energy"]
    # convert to torch
    obs = torch.tensor(obs).to(device)
    locs = torch.tensor(locs).to(device)
    eners = torch.tensor(eners).to(device)

    if use_message:
        messages = obs_dict["message"]
        return obs, locs, eners, messages

    return obs, locs, eners

def extract_dict_with_goal(obs_dict, device, use_message=False):
    obs, locs, goals = obs_dict["image"], obs_dict["location"], obs_dict["goal"]
    # convert to torch
    obs = torch.tensor(obs).to(device)
    locs = torch.tensor(locs).to(device)
    goals = torch.tensor(goals).to(device)

    if use_message:
        messages = obs_dict["message"]
        return obs, locs, goals, messages

    return obs, locs, goals


def extract_dict_separate(obs_dict, env_info, device, agent_id, num_agents, use_message=False):
    obs, locs, _ = obs_dict["image"], obs_dict["location"], obs_dict["energy"]
    selected_indices = torch.arange(agent_id, obs.shape[0], num_agents)

    # convert to torch
    obs = torch.tensor(obs)[selected_indices].to(device)
    locs = torch.tensor(locs)[selected_indices].to(device)
    out_env_info = None
    if env_info is not None:
        selected_indices_np = list(selected_indices.cpu().numpy())
        reward, terminations, truncations = env_info
        reward, terminations, truncations = reward[selected_indices_np], terminations[selected_indices_np], truncations[selected_indices_np]
        out_env_info = (reward, terminations, truncations)
    if use_message:
        messages = obs_dict["message"][selected_indices]
        return obs, locs, messages, out_env_info

    return obs, locs, env_info


def extract_dict_separate_with_goal(obs_dict, env_info, device, agent_id, num_agents, use_message=False):
    obs, locs, goals = obs_dict["image"], obs_dict["location"], obs_dict["goal"]
    selected_indices = torch.arange(agent_id, obs.shape[0], num_agents)

    # convert to torch
    obs = torch.tensor(obs)[selected_indices].to(device)
    locs = torch.tensor(locs)[selected_indices].to(device)
    goals = torch.tensor(goals)[selected_indices].to(device)
    out_env_info = None
    if env_info is not None:
        selected_indices_np = list(selected_indices.cpu().numpy())
        reward, terminations, truncations = env_info
        reward, terminations, truncations = reward[selected_indices_np], terminations[selected_indices_np], truncations[selected_indices_np]
        out_env_info = (reward, terminations, truncations)
    if use_message:
        messages = obs_dict["message"][selected_indices]
        return obs, locs, goals, messages, out_env_info

    return obs, locs, goals, env_info


def get_action_message_for_env(action, message):
    '''
    For agents with separated networks
    Input:
    action = {0: (n_envs,1) ndarray, 1: (n_envs,1) ndarray}
    message = {0: (n_envs,1) ndarray: (n_envs,1) ndarray}

    Output: out_action (2*n_envs, 1) ndarray, out_message (2*n_envs, 1) ndarray
    out_action = [action[0][0], action[1][0], action[0][1], action[1][1], ..., action[0][n_envs-1], action[1][n_envs-1]]
    '''

    out_action = np.vstack((action[0].cpu().numpy(), action[1].cpu().numpy())).flatten("F")
    out_message = np.vstack((message[0].cpu().numpy(), message[1].cpu().numpy())).flatten("F")

    return out_action, out_message


def batchify_obs(obs_dict, device):
    #TODO next_obs, next_locs, next_eners = next_obs_dict["image"], next_obs_dict["location"], next_obs_dict["energy"]
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.array([obs_dict[a]['image'] for a in obs_dict])
    locs = np.array([obs_dict[a]['location'] for a in obs_dict])
    eners = np.array([obs_dict[a]['energy'] for a in obs_dict])

    # convert to torch
    obs = torch.tensor(obs).to(device)
    locs = torch.tensor(locs).to(device)
    eners = torch.tensor(eners).to(device)
    return obs, locs, eners

def batchify(x, device):
    #TODO next_obs, next_locs, next_eners = next_obs_dict["image"], next_obs_dict["location"], next_obs_dict["energy"]
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    x = np.array([x[a] for a in x])

    # convert to torch
    x = torch.Tensor(x).to(device)
    return x

def unbatchify(x, num_envs, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()

    # x = {i: x[i] for i in range(num_envs)}
    x = {i:{j:0 for j in range(2)} for i in range(8)}

    return x
def init_logs(device, cfg, envs, agents):
    log_obs = torch.zeros(
        (
            cfg.max_steps,
            cfg.num_agents,
            cfg.num_channels,
            cfg.image_size,
            cfg.image_size,
        )
    ).to(device)
    log_locs = torch.zeros(
        (cfg.max_steps, cfg.num_agents, 2)
    ).to(device)
    log_r_messages = torch.zeros(
        (cfg.max_steps, cfg.num_agents), dtype=torch.int64
    ).to(device)
    log_actions = torch.zeros(
        (cfg.max_steps, cfg.num_agents)
    ).to(device)
    log_s_messages = (
        torch.zeros(
            (cfg.max_steps, cfg.num_agents), dtype=torch.int64
        ).to(device)
        - 1
    )
    log_rewards = torch.zeros(
        (cfg.max_steps, cfg.num_agents)
    ).to(device)
    log_s_message_embs = torch.zeros(
        (cfg.max_steps, agents[0].embedding_size, cfg.num_agents)
    ).to(device)

    log_food_dict = {}
    log_food_dict["target_food_id"] = (
        envs.target_food_id.squeeze(0).cpu().numpy()
    )
    log_food_dict["location"] = (
        envs.food_pos.squeeze(0).cpu().numpy()
    )
    log_food_dict["score"] = (
        envs.food_energy.squeeze(0).cpu().numpy()
    )

    who_see_target = (
        envs.score_visible_to_agent.squeeze(0).cpu().numpy()
    )
    return {
        "log_obs": log_obs,
        "log_locs": log_locs,
        "log_r_messages": log_r_messages,
        "log_actions": log_actions,
        "log_s_messages": log_s_messages,
        "log_rewards": log_rewards,
        "log_s_message_embs": log_s_message_embs,
        "log_food_dict": log_food_dict,
        "who_see_target": who_see_target,
    }
if __name__ == "__main__":
    action = {}
    action[0] = np.array([f"agent0_env{i}" for i in range(32)])
    action[1] = np.array([f"agent1_env{i}" for i in range(32)])
    out_action = np.vstack((action[0], action[1]))
    out_action = np.vstack((action[0], action[1])).flatten("F")
    print(out_action)