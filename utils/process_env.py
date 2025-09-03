import math
import random
import torch
import numpy as np

def assign_pairs_per_env(num_envs: int, pairs: list[list[int]], cursor: int = 0):
    """
    Deterministically cover all pairs over iterations.
    If num_envs >= len(pairs): cover all within a rollout (shuffled).
    If num_envs <  len(pairs): take a sliding window so coverage completes over multiple rollouts.
    Returns:
        pair_ids: LongTensor [B, 2] of network ids (agent0_net, agent1_net) for each env.
        next_cursor: int for next iteration's window start.
    """
    m = len(pairs)
    if num_envs >= m:
        # tile, shuffle, truncate
        tiled = (pairs * math.ceil(num_envs / m))[:num_envs]
        pair_ids = torch.tensor(tiled, dtype=torch.long)
        return pair_ids, 0
    else:
        # sliding window over pairs
        idxs = [(cursor + i) % m for i in range(num_envs)]
        pair_ids = torch.tensor([pairs[k] for k in idxs], dtype=torch.long)
        next_cursor = (cursor + num_envs) % m
        return pair_ids, next_cursor

if __name__ == "__main__":
    num_envs = 128
    num_steps = 16
    
    t = torch.tensor([[1, 2], [3, 4]])
    t_p = torch.gather(t, 0, torch.tensor([[0, 0], [1, 0]]))
    print(t_p)
    # pair_ids, _ = assign_pairs_per_env(40, [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5], [4, 6], [5, 7], [6, 7], [7, 8], [8, 9], [8, 10], [9, 10], [10, 12], [11, 12], [11, 13], [12, 13], [14, 0], [14, 1], [5, 13], [6, 7], [7, 10], [9, 6], [10, 8], [12, 5], [13, 6], [13, 5]])
    # envinds_all = np.arange(40)
    # num_networks = 15
    # for net_id in range(num_networks):
    #     for agent_id in [0,1]:
    #         env_idx = torch.where(pair_ids[:,agent_id]==net_id)[0]
            
    #         print(env_idx)

    #         env_idx = (pair_ids[:, agent_id] == net_id).nonzero(as_tuple=False).squeeze(-1)
    #         print(env_idx)
                