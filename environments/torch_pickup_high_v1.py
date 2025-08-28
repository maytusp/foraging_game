# 22 Aug 2025
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn.functional as F

@dataclass
class EnvConfig:
    grid_size: int = 5
    image_size: int = 3            # local obs crop (odd)
    num_agents: int = 2
    num_foods: int = 2             # N_i
    num_walls: int = 0
    max_steps: int = 10
    use_message: bool = False
    agent_visible: bool = False    # show other agents in obs
    food_energy_fully_visible: bool = False
    identical_item_obs: bool = False
    time_pressure: bool = True
    n_words: int = 10
    message_length: int = 1
    mode: Literal["train", "test"] = "train"
    test_moderate_score: bool = False
    seed: int = 42
    # constants (match your code where possible)
    N_val: int = 255
    N_att: int = 1
    agent_strength: int = 3
    food_strength_required: int = 6
    reward_scale: float = 1.0
    use_compile: bool = True

class TorchForagingEnv:
    """
    Batched GPU-native reimplementation of your env.
    State lives in torch tensors on `device`.
    Vectorized over: batch (num_envs), agents, foods.

    API:
      reset() -> (obs: Dict[int, dict], info: Dict)
      step(actions) -> (obs, rewards, dones, truncated, infos)
        - actions: Dict[int, int] or 1D LongTensor [num_agents] or 2D [B, num_agents]
      observe() -> obs dict (PettingZoo-style for single batch) or batched via get_batch_obs().
    """
    def __init__(self, cfg: EnvConfig, device: torch.device | str = "cuda", num_envs: int = 1):
        assert cfg.image_size % 2 == 1, "image_size must be odd"
        self.cfg = cfg
        self.device = torch.device(device)
        self.B = int(num_envs)
        self.num_actions = 5

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(cfg.seed)

        G = cfg.grid_size
        A = cfg.num_agents
        Fd = cfg.num_foods

        # --- RNG
        g = torch.Generator(device="cpu")
        g.manual_seed(cfg.seed)
        self._cpu_gen = g  # for random init on CPU then move to device

        # --- persistent tensors (allocated on device)
        self.agent_pos = torch.zeros(self.B, A, 2, dtype=torch.long, device=self.device)  # (y,x)
        self.agent_energy = torch.full((self.B, A), 20, dtype=torch.float32, device=self.device)
        self.food_pos = torch.zeros(self.B, Fd, 2, dtype=torch.long, device=self.device)
        self.food_done = torch.zeros(self.B, Fd, dtype=torch.bool, device=self.device)
        self.food_energy = torch.zeros(self.B, Fd, dtype=torch.float32, device=self.device)
        self.target_food_id = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.score_visible_to_agent = torch.zeros(self.B, Fd, dtype=torch.long, device=self.device)  # which agent sees which food
        self.wall_pos = torch.empty(self.B, 0, 2, dtype=torch.long, device=self.device)  # (B, W, 2)
        if cfg.num_walls > 0:
            self.wall_pos = torch.zeros(self.B, cfg.num_walls, 2, dtype=torch.long, device=self.device)

        self.curr_steps = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.dones_batch = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.trunc_batch = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.total_bump = torch.zeros(self.B, dtype=torch.long, device=self.device)

        # book-keeping per-agent (scalar env stats)
        self.cum_rewards = torch.zeros(self.B, A, dtype=torch.float32, device=self.device)
        self.episode_len = torch.zeros(self.B, A, dtype=torch.long, device=self.device)

        # spawn ranges like your code (two sides); simplified: top row vs bottom row; agents left/right halves
        self.food_top_row = True   # toggled when placing multiple foods (alternating sides)

        # precompute time-pressure denom
        self._max_steps_f = float(cfg.max_steps)

        # movement deltas
        self._delta = torch.tensor([[-1,0],[1,0],[0,-1],[0,1]], dtype=torch.long, device=self.device)  # up,down,left,right

        # occupancy values for observation
        self._occ_agent = torch.tensor(cfg.N_val//2, dtype=torch.float32, device=self.device)
        self._occ_food  = torch.tensor(cfg.N_val//3, dtype=torch.float32, device=self.device)
        self._occ_wall  = torch.tensor(cfg.N_val,    dtype=torch.float32, device=self.device)
        self._carry_add = torch.tensor(cfg.N_val//10, dtype=torch.float32, device=self.device)

        # score list (train/test) like your code
        if cfg.mode == "train":
            steps = torch.arange(0, 50, dtype=torch.long)
            score_list = (steps + 1) * 5
        else:
            if cfg.test_moderate_score:
                steps = torch.arange(100, 120, dtype=torch.long)
                score_list = (steps + 1) * 2
                score_list = score_list[(score_list % 5) != 0]
            else:
                steps = torch.arange(0, 125, dtype=torch.long)
                score_list = (steps + 1) * 2
                score_list = score_list[(score_list % 5) != 0]

        self.last_msgs = torch.zeros(self.B, cfg.num_agents, dtype=torch.long, device=self.device)
        
        self._score_list = score_list.to(torch.float32).to(self.device)

        # Precompute neighbor packs for every grid cell (P = G*G)
        # For spawning agents near foods
        P = G * G
        cells = torch.arange(P, device=self.device, dtype=torch.long)
        cy = (cells // G).view(P, 1) # (P,1)
        cx = (cells %  G).view(P, 1) # (P,1)

        neigh4 = torch.tensor([[-1,0],[1,0],[0,-1],[0,1]], device=self.device, dtype=torch.long) # (4,2)
        self.num_neigh = neigh4.shape[0]
        ny = cy + neigh4[:,0].view(1,4)   # (P,1) + (1,4) = (P,4)
        nx = cx + neigh4[:,1].view(1,4)   # (P,1) + (1,4) = (P,4)

        valid = (ny >= 0) & (ny < G) & (nx >= 0) & (nx < G)              # (P,4)
        cand_flat = (ny.clamp(0, G-1) * G + nx.clamp(0, G-1)).to(torch.long)  # (P,4)

        # Pack valid neighbors to the left so the first K entries are valid (K varies by cell)
        # Sort by validity (True before False)
        sort_key = valid.to(torch.int64)
        _, sort_idx = torch.sort(sort_key, dim=1, descending=True)       # (P,4)
        packed_neighbors = cand_flat.gather(1, sort_idx)                 # (P,4)

        valid_counts = valid.sum(dim=1).to(torch.long)                   # (P,)

        self._packed_neighbors = packed_neighbors                        # (P,4)
        self._valid_counts = valid_counts                                # (P,)

        self._reset_indices(torch.ones((self.B,), device=self.device))

        if cfg.use_compile:
            self._step_core = torch.compile(self.step, fullgraph=False)
            self._obs_core  = torch.compile(self.observe,  fullgraph=False)

    # ---------------------- Utilities ----------------------
    def _rand_choice(self, choices: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        idx = torch.randint(0, choices.numel(), shape, device=self.device, generator=self.rng)
        return choices[idx]

    def _random_positions(self, count: int) -> torch.Tensor:
        """Return (B, count, 2) random free positions (naive sampling; fine for small grids)."""
        G = self.cfg.grid_size
        pos = torch.stack([
            torch.randint(0, G, (self.B, count), device=self.device, generator=self.rng),
            torch.randint(0, G, (self.B, count), device=self.device, generator=self.rng),
        ], dim=-1).to(torch.long)
        return pos.to(self.device)

    def _make_index(self, yx: torch.Tensor, W: int) -> torch.Tensor:
        """Flattened indices from (.., 2) coords (y,x)."""
        return (yx[...,0] * W + yx[...,1]).to(torch.long)

    # ---------------------- Observation ----------------------
    def _make_occupancy_and_attr_maps(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
          occ_map: (B, G, G) float32
          attr_map: (B, G, G, N_att) float32
        """
        B, G, Fd, A, cfg = self.B, self.cfg.grid_size, self.cfg.num_foods, self.cfg.num_agents, self.cfg
        occ = torch.zeros(B, G, G, dtype=torch.float32, device=self.device)
        # walls
        if self.wall_pos.numel() > 0:
            idx_w = self._make_index(self.wall_pos, G)  # (B, W)
            occ.view(B, -1).scatter_(1, idx_w, self._occ_wall.expand_as(idx_w).to(torch.float32))
        # foods
        alive_food_mask = (~self.food_done)
        if alive_food_mask.any():
            pos_f = self.food_pos.clone()
            pos_f[~alive_food_mask] = 0  # dummy
            idx_f = self._make_index(pos_f, G)
            val_f = self._occ_food.expand_as(idx_f).to(torch.float32)
            # zero where done
            val_f = val_f * alive_food_mask.to(torch.float32)
            occ.view(B, -1).scatter_add_(1, idx_f, val_f)

        # agents (optionally add carry_add if you add carrying later)
        idx_a = self._make_index(self.agent_pos, G)  # (B, A)
        val_a = self._occ_agent.expand_as(idx_a).to(torch.float32)
        occ.view(B, -1).scatter_add_(1, idx_a, val_a)

        # attributes: only at food cells; either full or masked to the agent later
        attr = torch.zeros(B, G, G, cfg.N_att, dtype=torch.float32, device=self.device)
        if alive_food_mask.any():
            # single attribute = energy
            yx = self.food_pos.clone()
            en = self.food_energy.clone()  # (B, Fd)
            # write energy into attr[..., 0] at each food location
            idx_flat = self._make_index(yx, G)  # (B, Fd)
            attr0 = attr[..., 0].contiguous().view(B, -1)
            attr0.scatter_(1, idx_flat, en * (~self.food_done).to(en.dtype))
        return occ, attr

    def _crop_around(self, grid, centers, pad_val: float):
        # grid: (B,H,W) or (B,H,W,C); centers: (B,A,2); return (B,A,C,K,K)
        B, G, K, A = self.B, self.cfg.grid_size, self.cfg.image_size, self.cfg.num_agents
        r = K // 2
        if grid.dim() == 3:
            grid = grid.unsqueeze(-1)                   # (B,H,W,1)
        grid = grid.permute(0,3,1,2).contiguous()       # (B,C,H,W) channels-first

        # constant pad on GPU
        pad = (r, r, r, r)
        grid = F.pad(grid, pad, mode="constant", value=pad_val)

        # unfold windows at every (y,x); size now (G+2r, G+2r) with kernel K and stride 1 → (G*G) positions
        patches = F.unfold(grid, kernel_size=K, padding=0, stride=1)  # (B, C*K*K, G*G)

        # flatten centers to position indices in the unpadded HxW lattice
        idx = self._make_index(centers, G)  # (B, A)
        idx = idx.unsqueeze(1).expand(B, grid.size(1)*K*K, A)  # (B, C*K*K, A)

        # gather all agent patches in one go
        pa = patches.gather(2, idx)                         # (B, C*K*K, A)
        pa = pa.view(B, grid.size(1), K, K, A).permute(0,4,1,2,3).contiguous()  # (B,A,C,K,K)
        return pa

    @torch.no_grad()
    def observe(self):
        B, G, A, Fd, K = self.B, self.cfg.grid_size, self.cfg.num_agents, self.cfg.num_foods, self.cfg.image_size
        occ, attr = self._make_occupancy_and_attr_maps() # occ:  (B, G, G), attr:  (B, G, G, N_att)
        occ_crops  = self._crop_around(occ,  self.agent_pos, pad_val=float(self.cfg.N_val)) # (B,A,1,K,K)
        attr_crops = self._crop_around(attr, self.agent_pos, pad_val=0.0) # (B,A,1,K,K)

        #TODO Mask Attribute bt agents who can see those items
        owner_map = torch.full((B, G*G), -1, dtype=torch.long, device=self.device)   # (B, G*G)

        idx_flat = self._make_index(self.food_pos, G)                 # (B, Fd) (Fd <= G*G)
        alive    = ~self.food_done                                    # (B, Fd)
        owners   = self.score_visible_to_agent                        # (B, Fd)
        bidx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, Fd)  # (B, Fd)

        # fill the onwer_map with agent id at the food position (0 or 1)
        owner_map[bidx[alive], idx_flat[alive]] = owners[alive] # (B, G*G)
        owner_map = owner_map.view(B, G, G)  # (B, G, G)

        # One-hot over agents per cell: (B, G, G, A); zeros where no item
        one_hot = F.one_hot(owner_map.clamp_min(0), num_classes=A).to(attr_crops.dtype)  # (B,G,G,A)
        one_hot = one_hot * (owner_map >= 0).unsqueeze(-1).to(one_hot.dtype) # (B,G,G,A)

        # local observation
        one_hot_crop = self._crop_around(one_hot, self.agent_pos, pad_val=0.0) # (B,A,A,K,K)
        channel_idx = torch.arange(A, device=self.device).view(1,A,1,1,1)
        channel_idx_exp = channel_idx.expand(B,-1,1,K,K)
        # print(f"one_hot_crop {one_hot_crop}")
        # print(f"channel_idx_exp {channel_idx_exp}")
        attr_mask = one_hot_crop.gather(2, channel_idx_exp) # (B,A,K,K)
        # print(f"attr_mask {attr_mask.shape}")
        attr_crops = attr_mask * attr_crops
        #TODO check again before training

        img = torch.cat([occ_crops, attr_crops], dim=2)  # (B,A,C,K,K)
        pos = self.agent_pos.to(torch.float32)             # (B,A,2)

        if self.cfg.use_message:
            msgs = self.last_msgs                           # (B,A,L)
        else:
            msgs = None
        return img, pos, msgs


    def _reset_indices(self, mask: torch.Tensor):
        cfg, B, A, Fd, G, nb = self.cfg, self.B, self.cfg.num_agents, self.cfg.num_foods, self.cfg.grid_size, self.num_neigh
        mask = mask.bool()
        if not mask.any(): return
        idxs = torch.nonzero(mask, as_tuple=False).view(-1)
        n = idxs.numel()

        score_pool = self._score_list
        # sample Fd scores per env from pool
        perm = torch.randperm(score_pool.numel(), device=self.device, generator=self.rng)
        perm = perm[:Fd].unsqueeze(0).expand(n, -1)
        self.food_energy[idxs] = score_pool[perm]

        # round-robin-ish visibility (random perm then mod agents)
        order = torch.randperm(Fd, device=self.device, generator=self.rng).unsqueeze(0).expand(n,-1)
        self.score_visible_to_agent[idxs] = order % A

        # target food = argmax
        self.target_food_id[idxs] = torch.argmax(self.food_energy[idxs], dim=1)

        # foods alternate top/bottom rows + random x
        y = torch.zeros(n, Fd, dtype=torch.long, device=self.device) # (n, Fd)
        y[:, 1::2] = G - 1 # transform y from [0,0,0,0,...] to [0,G-1,0,G-1,...]
        x = torch.randint(0, G, (n, Fd), device=self.device, generator=self.rng)
        self.food_pos[idxs] = torch.stack([y, x], dim=-1)
        self.food_done[idxs] = False


        # spawn agents near a food they "see" (no Python loops)
        # pick visible food id per (env, agent); fallback to random if none
        # Example: suppose n = 1 (only one environment) and A = 2 (two agents)
        # vis (1,2) = [0,1] means the first item is seen by agent0 and the second is seen by agent1
        # mark_vis = [[1,0],[0,1]]. The value indicates whether item #col_id is seen by agent #row_id or not.
        vis = self.score_visible_to_agent[idxs]                 # (n, Fd) in [0..A-1]
        mask_vis = vis.unsqueeze(1).eq(torch.arange(A, device=self.device).view(1,A,1)) # (n,A,Fd)
        food_idx = mask_vis.float().argmax(-1)                 # (n,A)

        # gather base food positions for each agent
        base = self.food_pos[idxs].gather(1, food_idx.unsqueeze(-1).expand(n, A, 2))     # (n,A,2)
        p_flat = (base[...,0] * G + base[...,1]).to(torch.long) # (n,A)
        all_neighs = self._packed_neighbors.index_select(0, p_flat.view(-1)).view(n, A, nb) # (n,A,nb)
        counts = self._valid_counts.index_select(0, p_flat.view(-1)).view(n, A).to(torch.float32) # (n,A)
        
        # randomly select neighbours given food positions
        u = torch.rand(n, A, device=self.device, generator=self.rng)            # ~ U[0,1)
        c = counts.to(torch.float32)                                            # (n,A)
        r = (u * c).floor().to(torch.long)                                      # (n,A)
        chosen_flat = all_neighs.gather(2, r.unsqueeze(-1)).squeeze(-1)                     # (n, A)
        chosen_pos = torch.stack([chosen_flat // G, chosen_flat % G], dim=-1) # (n,A,2)

        self.agent_pos[idxs] = chosen_pos

        self.agent_energy[idxs] = 20.0
        self.curr_steps[idxs] = 0
        self.dones_batch[idxs] = False
        self.trunc_batch[idxs] = False
        self.total_bump[idxs] = 0
        self.cum_rewards[idxs] = 0
        self.episode_len[idxs] = 0
        self.last_msgs[idxs] = 0

    @torch.no_grad()
    def step(self, actions, auto_reset=True) -> Tuple[Dict[int, dict], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict]:
        """
        actions: Dict[int, int] or LongTensor [A] or [B,A]
          0: up, 1: down, 2: left, 3: right, 4: pick_up
        """
        cfg = self.cfg
        B, G, A, Fd = self.B, cfg.grid_size, cfg.num_agents, cfg.num_foods

        # normalize actions to LongTensor [B,A]
        if isinstance(actions, dict):
            a = torch.tensor([int(actions[i]) for i in range(A)], device=self.device, dtype=torch.long).unsqueeze(0).expand(self.B, -1)
        elif isinstance(actions, torch.Tensor):
            a = actions.to(self.device).long()
            if a.dim() == 1: a = a.unsqueeze(0).expand(self.B, -1)
        else:
            raise TypeError("actions must be dict or torch.Tensor")

        self.curr_steps += 1

        # movement phase (0..3)
        move_mask = (a < 4)
        deltas = torch.zeros(self.B, A, 2, dtype=torch.long, device=self.device)
        for i in range(4):
            sel = (a == i)
            if sel.any():
                deltas[sel] = self._delta[i]

        proposed = self.agent_pos + deltas
        proposed = torch.clamp(proposed, 0, G-1)

        # block on walls & other agents: build occupancy of current agent positions
        occ_agents_now = torch.full((self.B, G, G), -1, dtype=torch.long, device=self.device)  # -1 empty else agent id
        idx_flat = self._make_index(self.agent_pos, G)
        # mark occupied
        b_idx = torch.arange(self.B, device=self.device).unsqueeze(1).expand_as(idx_flat)
        occ_agents_now.view(self.B, -1)[b_idx, idx_flat] = torch.arange(A, device=self.device).unsqueeze(0).expand_as(idx_flat)

        # walls occupancy
        occ_walls = torch.zeros(self.B, G, G, dtype=torch.bool, device=self.device)
        if self.wall_pos.numel() > 0:
            idx_w = self._make_index(self.wall_pos, G)
            b_idx_w = torch.arange(self.B, device=self.device).unsqueeze(1).expand_as(idx_w)
            occ_walls.view(self.B, -1)[b_idx_w[:,:idx_w.size(1)], idx_w] = True

        # food occupancy
        occ_foods = torch.zeros(self.B, G, G, dtype=torch.bool, device=self.device)
        if self.food_pos.numel() > 0:
            idx_f = self._make_index(self.food_pos, G)
            b_idx_f = torch.arange(self.B, device=self.device).unsqueeze(1).expand_as(idx_f)
            occ_foods.view(self.B, -1)[b_idx_f[:,:idx_f.size(1)], idx_f] = True

        # invalid if target cell is a wall or another agent *current* pos
        tgt_flat = self._make_index(proposed, G)
        tgt_is_wall = occ_walls.view(self.B, -1)[b_idx, tgt_flat]
        tgt_has_agent = (occ_agents_now.view(self.B, -1)[b_idx, tgt_flat] != -1)
        tgt_has_food = occ_foods.view(self.B, -1)[b_idx, tgt_flat]

        tgt_flat = self._make_index(proposed, G)   # (B,A)
        dup = tgt_flat.unsqueeze(2).eq(tgt_flat.unsqueeze(1))  # (B,A,A)
        dup = dup.triu(diagonal=1).any(-1)  # (B,A) true if another agent targets same cell

        can_move = move_mask & (~tgt_is_wall) & (~tgt_has_agent) & (~tgt_has_food) & (~dup)

        # bumps: moving into another agent
        self.total_bump += (move_mask & tgt_has_agent).sum(dim=1)

        # apply movement
        self.agent_pos = torch.where(can_move.unsqueeze(-1), proposed, self.agent_pos)

        # pickup phase (action==4), within l2<=sqrt(2)
        pick_mask = (a == 4)
        picked_food = torch.full((self.B,), -1, dtype=torch.long, device=self.device)

        if pick_mask.any():
            # pairwise dist^2 between agents and foods: (B,A,Fd)
            ay = self.agent_pos[...,0].unsqueeze(-1)  # (B,A,1)
            ax = self.agent_pos[...,1].unsqueeze(-1)
            fy = self.food_pos[...,0].unsqueeze(1)    # (B,1,Fd)
            fx = self.food_pos[...,1].unsqueeze(1)
            dy2 = (ay - fy).to(torch.float32).pow(2)
            dx2 = (ax - fx).to(torch.float32).pow(2)
            near = (dy2 + dx2) <= 2.0  # l2<=sqrt(2)

            # agents attempting pickup and near each food
            attempt = pick_mask.unsqueeze(-1) & near & (~self.food_done.unsqueeze(1))
            # combined strength per food
            strength_sum = attempt.to(torch.float32).sum(dim=1) * float(cfg.agent_strength)  # (B,Fd)

            # foods that can be lifted now
            can_lift = (strength_sum >= float(cfg.food_strength_required)) & (~self.food_done)
            if can_lift.any():
                # pick the first food (lowest id) per env to match your single-collection terminal
                first_idx = torch.argmax(can_lift.to(torch.int64), dim=1)  # gives 0 if none; guard below
                mask_any = can_lift.any(dim=1)
                picked_food[mask_any] = first_idx[mask_any]
                # mark done & remove from map
                self.food_done[mask_any, first_idx[mask_any]] = True
                # move it off-grid (optional)
                self.food_pos[mask_any, first_idx[mask_any]] = torch.tensor([-2000, -2000], device=self.device)

        # rewards, termination
        rewards = torch.zeros(self.B, A, dtype=torch.float32, device=self.device)
        success = torch.zeros(self.B, dtype=torch.long, device=self.device)

        # terminal on single collection
        collected = (picked_food >= 0)
        if collected.any():
            # right target?
            right = (picked_food == self.target_food_id)
            wrong = collected & (~right)
            # +1 (and time bonus) to all agents if right
            if right.any():
                bonus = ((self.cfg.max_steps - self.curr_steps[right]) / self._max_steps_f).clamp_min(0.0) if self.cfg.time_pressure else 0.0
                rewards[right] += (1.0 + bonus).unsqueeze(-1)
                success[right] = 1
            if wrong.any():
                rewards[wrong] -= 1.0
            # everyone done in those envs
            self.dones_batch = self.dones_batch | collected
        
        # timeout
        timed_out = (self.curr_steps >= self.cfg.max_steps)
        if timed_out.any():
            rewards[timed_out] -= 1.0
            self.trunc_batch = self.trunc_batch | timed_out
            self.dones_batch = self.dones_batch | timed_out

        finished  = self.dones_batch | self.trunc_batch

        dones_out = self.dones_batch.clone()
        truncs_out = self.trunc_batch.clone()

        # normalize rewards and update stats
        rewards = rewards / float(cfg.reward_scale)
        self.cum_rewards += rewards
        self.episode_len += 1

        # infos (report per-agent like PZ)
        infos: Dict[int, dict] = {}
        # Only for first env to match PZ-style return
        b = 0
        infos = None

        if auto_reset and finished.any():
            self._reset_indices(finished)

        obs = self.observe()
    
        return obs, rewards, dones_out, truncs_out, infos

    def close(self):
        pass
