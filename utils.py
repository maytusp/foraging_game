import numpy as np
import torch

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