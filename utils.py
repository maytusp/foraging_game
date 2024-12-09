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