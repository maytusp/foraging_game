# TODO Debug this, it does not converge
# torch_temporalg_v1.py
# TemporalG variant of ScoreG
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn.functional as F
import torch._dynamo as dynamo
@dataclass
class EnvConfig:
    grid_size: int = 13
    image_size: int = 7            # local obs crop (odd)
    comm_field: int = 7
    num_channels: int = 1          # only occupancy channel in TemporalG
    num_agents: int = 2
    num_foods: int = 2             # N_i
    num_walls: int = 0
    max_steps: int = 50            # can be 20 or 50, etc.
    agent_visible: bool = False    # show other agents in obs
    freeze_dur: int = 6 # duration that agent observes spawning items
    # retained for compatibility
    time_pressure: bool = True
    n_words: int = 4
    message_length: int = 1
    mode: Literal["train", "test"] = "train"
    seed: int = 42
    # constants
    N_val: int = 255
    N_att: int = 0                 # no attribute (score) channel in TemporalG
    agent_strength: int = 3
    food_strength_required: int = 6
    reward_scale: float = 1.0
    use_compile: bool = True


class TorchTemporalEnv:
    """
    TemporalG environment.

    Differences from ScoreG:
      - No score channel in observation (only occupancy).
      - Two items spawn at distinct times in {1..freeze_dur}.
      - Agents frozen for t in [1,freeze_dur]; can move / pick only after t>freeze_dur.
      - Items must be picked up in the order they spawn (cooperative pickup).
      - Success only when both items picked in correct order; reward +1 + time bonus.
      - Wrong-order pickup or timeout gives -1.
      - Each item is guaranteed to be visible to exactly one agent at spawn,
        using a receptive field determined by cfg.image_size.
    """

    def __init__(self, cfg: EnvConfig, device: torch.device | str = "cuda", num_envs: int = 1):
        assert cfg.image_size % 2 == 1, "image_size must be odd"
        self.cfg = cfg
        self.device = torch.device(device)
        self.B = int(num_envs)
        self.num_actions = 5  # 0: up,1:down,2:left,3:right,4:pick_up

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(cfg.seed)

        G = cfg.grid_size
        A = cfg.num_agents
        Fd = cfg.num_foods
        self.agent_visible = cfg.agent_visible

        # RNG also on CPU (if ever needed)
        g = torch.Generator(device="cpu")
        g.manual_seed(cfg.seed)
        self._cpu_gen = g

        # --- persistent tensors (allocated on device) ---
        # positions: (y, x)
        self.agent_pos = torch.zeros(self.B, A, 2, dtype=torch.long, device=self.device)
        self.agent_energy = torch.full((self.B, A), 20, dtype=torch.float32, device=self.device)

        # Items ("foods")
        self.food_pos = torch.full((self.B, Fd, 2), -1000, dtype=torch.long, device=self.device)
        self.food_done = torch.zeros(self.B, Fd, dtype=torch.bool, device=self.device)      # collected
        self.food_spawned = torch.zeros(self.B, Fd, dtype=torch.bool, device=self.device)   # has appeared

        # Each item has a spawn step in {1..freeze_dur}, distinct per env
        self.food_spawn_step = torch.zeros(self.B, Fd, dtype=torch.long, device=self.device)

        # Which agent will see the item when it spawns (used to ensure visibility)
        self.food_visible_agent = torch.zeros(self.B, Fd, dtype=torch.long, device=self.device)

        # Spawn order: indices of foods ordered by spawn time
        self.spawn_order = torch.zeros(self.B, Fd, dtype=torch.long, device=self.device)
        # Stage: 0 = need first item, 1 = need second item, 2 = done
        self.collection_stage = torch.zeros(self.B, dtype=torch.long, device=self.device)
        # Current target food id (according to spawn order)
        self.current_target_food_id = torch.zeros(self.B, dtype=torch.long, device=self.device)

        # Walls (none by default, but kept for generality)
        self.wall_pos = torch.empty(self.B, 0, 2, dtype=torch.long, device=self.device)
        if cfg.num_walls > 0:
            self.wall_pos = torch.zeros(self.B, cfg.num_walls, 2, dtype=torch.long, device=self.device)

        # Bookkeeping
        self.curr_steps = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.dones_batch = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.trunc_batch = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.total_bump = torch.zeros(self.B, dtype=torch.long, device=self.device)

        self.cum_rewards = torch.zeros(self.B, A, dtype=torch.float32, device=self.device)
        self.episode_len = torch.zeros(self.B, A, dtype=torch.long, device=self.device)

        # Precompute time-pressure denom
        self._max_steps_f = float(cfg.max_steps)

        # movement deltas
        self._delta = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1]],
            dtype=torch.long,
            device=self.device
        )  # up,down,left,right

        # occupancy values for observation
        self._occ_agent = torch.tensor(cfg.N_val // 2, dtype=torch.float32, device=self.device)
        self._occ_food  = torch.tensor(cfg.N_val // 3, dtype=torch.float32, device=self.device)
        self._occ_wall  = torch.tensor(cfg.N_val,      dtype=torch.float32, device=self.device)

        # Precompute neighbors per cell (for potential use; not strictly needed here)
        P = G * G
        cells = torch.arange(P, device=self.device, dtype=torch.long)
        cy = (cells // G).view(P, 1)
        cx = (cells %  G).view(P, 1)

        neigh4 = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1]],
            device=self.device,
            dtype=torch.long
        )
        self.num_neigh = neigh4.shape[0]
        ny = cy + neigh4[:, 0].view(1, 4)
        nx = cx + neigh4[:, 1].view(1, 4)

        valid = (ny >= 0) & (ny < G) & (nx >= 0) & (nx < G)
        cand_flat = (ny.clamp(0, G - 1) * G + nx.clamp(0, G - 1)).to(torch.long)

        sort_key = valid.to(torch.int64)
        _, sort_idx = torch.sort(sort_key, dim=1, descending=True)
        packed_neighbors = cand_flat.gather(1, sort_idx)
        valid_counts = valid.sum(dim=1).to(torch.long)

        self._packed_neighbors = packed_neighbors  # (P,4)
        self._valid_counts = valid_counts          # (P,)

        # Precompute grid coordinates (flattened 0..G*G-1) to help item spawning
        self._grid_indices = torch.arange(P, device=self.device, dtype=torch.long)
        self._grid_y = (self._grid_indices // G)  # (P,)
        self._grid_x = (self._grid_indices %  G)  # (P,)

        self._reset_indices(torch.ones((self.B,), device=self.device, dtype=torch.bool))

        if cfg.use_compile:
            self._step_core = torch.compile(self.step, fullgraph=False)
            self._obs_core  = torch.compile(self.observe,  fullgraph=False)

    # ---------------------- Utilities ----------------------
    def _rand_choice(self, choices: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        idx = torch.randint(0, choices.numel(), shape, device=self.device, generator=self.rng)
        return choices[idx]

    def _make_index(self, yx: torch.Tensor, W: int) -> torch.Tensor:
        """Flattened indices from (.., 2) coords (y,x)."""
        return (yx[..., 0] * W + yx[..., 1]).to(torch.long)

    # ---------------------- Observation helpers ----------------------
    def _make_occupancy_map(self) -> torch.Tensor:
        """
        Return:
          occ_map: (B, G, G) float32
        """
        B, G, Fd, A, cfg = self.B, self.cfg.grid_size, self.cfg.num_foods, self.cfg.num_agents, self.cfg
        occ = torch.zeros(B, G, G, dtype=torch.float32, device=self.device)

        # walls
        if self.wall_pos.numel() > 0:
            idx_w = self._make_index(self.wall_pos, G)  # (B, W)
            occ.view(B, -1).scatter_(1, idx_w, self._occ_wall.expand_as(idx_w))

        # active foods (spawned and not done)
        active_food_mask = self.food_spawned & (~self.food_done)  # (B,Fd)
        if active_food_mask.any():
            pos_f = self.food_pos.clone()
            pos_f[~active_food_mask] = 0
            idx_f = self._make_index(pos_f, G)  # (B,Fd)
            val_f = self._occ_food.expand_as(idx_f).to(torch.float32)
            val_f = val_f * active_food_mask.to(torch.float32)
            occ.view(B, -1).scatter_add_(1, idx_f, val_f)

        # agents
        idx_a = self._make_index(self.agent_pos, G)  # (B,A)
        val_a = self._occ_agent.expand_as(idx_a)
        occ.view(B, -1).scatter_add_(1, idx_a, val_a)

        return occ

    def _crop_around(self, grid, centers, pad_val: float):
        # grid: (B,H,W) or (B,H,W,C); centers: (B,A,2); return (B,A,C,K,K)
        B, G, K, A = self.B, self.cfg.grid_size, self.cfg.image_size, self.cfg.num_agents
        r = K // 2
        if grid.dim() == 3:
            grid = grid.unsqueeze(-1)                   # (B,H,W,1)
        grid = grid.permute(0, 3, 1, 2).contiguous()    # (B,C,H,W) channels-first

        pad = (r, r, r, r)
        grid = F.pad(grid, pad, mode="constant", value=pad_val)

        patches = F.unfold(grid, kernel_size=K, padding=0, stride=1)  # (B, C*K*K, G*G)

        idx = self._make_index(centers, G)  # (B,A)
        idx = idx.unsqueeze(1).expand(B, grid.size(1) * K * K, A)  # (B, C*K*K, A)

        pa = patches.gather(2, idx)  # (B, C*K*K, A)
        pa = pa.view(B, grid.size(1), K, K, A).permute(0, 4, 1, 2, 3).contiguous()  # (B,A,C,K,K)
        return pa

    @torch.no_grad()
    def observe(self):
        """
        Observation:
          - image: (B, A, 1, K, K) occupancy only
          - pos:   (B, A, 2) agent absolute coords
        """
        B, G, A, Fd, K = self.B, self.cfg.grid_size, self.cfg.num_agents, self.cfg.num_foods, self.cfg.image_size

        occ = self._make_occupancy_map()  # (B,G,G)
        occ_crops = self._crop_around(occ, self.agent_pos, pad_val=float(self.cfg.N_val))  # (B,A,1,K,K)

        # Mask ONLY other agents if not visible
        if not self.agent_visible:
            # agent_map[b, y, x] = agent id in that cell, or -1 if no agent
            agent_map = torch.full((B, G, G), -1, dtype=torch.long, device=self.device)
            agent_idx_flat = self._make_index(self.agent_pos, G)  # (B,A)
            agent_bidx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, A)
            agent_id = torch.arange(A, device=self.device).unsqueeze(0).expand(B, A)
            agent_map.view(B, -1)[agent_bidx, agent_idx_flat] = agent_id

            # Crop this integer map around each agent
            # Use float for F.unfold/F.pad, then compare by value
            agent_map_crop = self._crop_around(
                agent_map.to(torch.float32), self.agent_pos, pad_val=-1.0
            )  # (B,A,1,K,K)
            agent_map_crop = agent_map_crop.squeeze(2)  # (B,A,K,K), entries in {-1, 0..A-1}

            # For each observing agent i, hide cells that contain some j != i
            aid = torch.arange(A, device=self.device, dtype=torch.float32).view(1, A, 1, 1)  # (1,A,1,1)
            is_other_agent = (agent_map_crop != -1.0) & (agent_map_crop != aid)             # (B,A,K,K)

            # mask = 0 where other agent present, 1 elsewhere
            mask = (~is_other_agent).unsqueeze(2).to(occ_crops.dtype)                      # (B,A,1,K,K)

            occ_crops = occ_crops * mask

        img = occ_crops  # (B,A,1,K,K)
        pos = self.agent_pos.to(torch.float32)

        # limit communication range
        ay = self.agent_pos[..., 0].unsqueeze(2)  # (B,A,1)
        ax = self.agent_pos[..., 1].unsqueeze(2)
        by = self.agent_pos[..., 0].unsqueeze(1)  # (B,1,A)
        bx = self.agent_pos[..., 1].unsqueeze(1)

        cheb = torch.maximum((ay - by).abs(), (ax - bx).abs())  # (B,A,A)

        R = (self.cfg.comm_field - 1) // 2
        comm_inrange = (cheb <= R)
        eye = torch.eye(A, dtype=torch.bool, device=self.device).unsqueeze(0)
        comm_inrange = comm_inrange & (~eye)


        return img, pos, comm_inrange

    # ---------------------- Reset ----------------------
    @dynamo.disable
    def _reset_indices(self, mask: torch.Tensor):
        """
        Reset all envs where mask==True.
        TemporalG-specific reset:
          - spawn agents on opposite vertical sides of the grid (top vs bottom rows)
          - sample item spawn times (distinct in {1..6})
          - assign each item a visible agent (ideally spread over agents)
          - place random walls in the middle band of the grid
          - ensure items will spawn inside the visible agent's RF (handled in _maybe_spawn_items).
        """
        cfg, B, A, Fd, G = self.cfg, self.B, self.cfg.num_agents, self.cfg.num_foods, self.cfg.grid_size
        mask = mask.bool()
        if not mask.any():
            return
        idxs = torch.nonzero(mask, as_tuple=False).view(-1)
        n = idxs.numel()

        # --- agents on opposite vertical sides: TOP (y=0) and BOTTOM (y=G-1) ---
        top_count = (A + 1) // 2
        bottom_count = A // 2

        x_top = torch.randint(0, G, (n, top_count), device=self.device, generator=self.rng)
        x_bottom = torch.randint(0, G, (n, bottom_count), device=self.device, generator=self.rng)

        pos = torch.zeros(n, A, 2, dtype=torch.long, device=self.device)

        # top agents
        if top_count > 0:
            pos[:, :top_count, 0] = 0          # y = 0 (top row)
            pos[:, :top_count, 1] = x_top      # x random

        # bottom agents
        if bottom_count > 0:
            pos[:, top_count:, 0] = G - 1      # y = G-1 (bottom row)
            pos[:, top_count:, 1] = x_bottom   # x random

        self.agent_pos[idxs] = pos
        self.agent_energy[idxs] = 20.0

        # --- WALLS: SOME IN AGENT RF, SOME IN MIDDLE BAND ---
        if cfg.num_walls > 0:
            rf_radius = (cfg.image_size - 1) // 2
            num_mid_walls = cfg.num_walls

            y_min_pref = 1
            y_max_pref = G - 2

            max_attempts = 200

            for b in idxs.tolist():
                walls_b = []

                def is_blocked(y, x):
                    cand = torch.tensor([y, x], device=self.device, dtype=torch.long)
                    # avoid agents
                    if (self.agent_pos[b] == cand).all(dim=-1).any():
                        return True
                    # avoid existing walls
                    if len(walls_b) > 0:
                        wb = torch.stack(walls_b, dim=0)
                        if (wb == cand).all(dim=-1).any():
                            return True
                    return False


                # 2) MIDDLE-BAND WALLS (num_mid_walls)
                placed_mid = 0
                while placed_mid < num_mid_walls and len(walls_b) < cfg.num_walls:
                    attempts = 0
                    placed = False
                    while (not placed) and (attempts < max_attempts):
                        attempts += 1
                        # middle vertical band, away from top/bottom bands
                        y = torch.randint(y_min_pref, y_max_pref + 1,
                                          (1,), device=self.device,
                                          generator=self.rng).item()
                        x = torch.randint(1, G - 1,
                                          (1,), device=self.device,
                                          generator=self.rng).item()

                        if is_blocked(y, x):
                            continue

                        walls_b.append(torch.tensor([y, x],
                                                    device=self.device,
                                                    dtype=torch.long))
                        placed = True
                        placed_mid += 1

                    if not placed:
                        # geometry tight; relax and place anywhere free
                        for _ in range(max_attempts):
                            y = torch.randint(0, G, (1,), device=self.device,
                                              generator=self.rng).item()
                            x = torch.randint(0, G, (1,), device=self.device,
                                              generator=self.rng).item()
                            if not is_blocked(y, x):
                                walls_b.append(torch.tensor([y, x],
                                                            device=self.device,
                                                            dtype=torch.long))
                                placed_mid += 1
                                break
                        else:
                            # still failed; break to avoid infinite loop
                            break

                # If we somehow have fewer than cfg.num_walls, we can pad by duplicating
                if len(walls_b) == 0:
                    # degenerate fallback: put a single wall at (0,0)
                    walls_b.append(torch.tensor([0, 0], device=self.device, dtype=torch.long))

                if len(walls_b) < cfg.num_walls:
                    # duplicate last wall to fill remaining slots (harmless)
                    last = walls_b[-1]
                    while len(walls_b) < cfg.num_walls:
                        walls_b.append(last.clone())

                self.wall_pos[b] = torch.stack(walls_b, dim=0)


        # --- item spawn schedule: distinct steps in {1..freeze_dur} ---
        noise = torch.rand(n, cfg.freeze_dur, device=self.device, generator=self.rng)
        perm = noise.argsort(dim=1)[:, :Fd]          # (n,Fd), values in [0..freeze_dur]
        spawn_steps = perm + 1                       # convert to [1..freeze_dur]
        self.food_spawn_step[idxs] = spawn_steps

        # spawn order: sorted by spawn step (earliest first)
        spawn_order = spawn_steps.argsort(dim=1)     # (n,Fd), indices 0..Fd-1
        self.spawn_order[idxs] = spawn_order

        # nothing spawned / done yet; items off-grid
        self.food_spawned[idxs] = False
        self.food_done[idxs] = False
        self.food_pos[idxs] = torch.tensor([-1000, -1000], device=self.device)

        # choose visible agent for each item
        if Fd == A:
            noise_va = torch.rand(n, A, device=self.device, generator=self.rng)
            perm_va = noise_va.argsort(dim=1)        # (n,A)
            visible_agent = perm_va                  # (n,Fd) since Fd==A
        elif Fd < A:
            noise_va = torch.rand(n, A, device=self.device, generator=self.rng)
            visible_agent = noise_va.argsort(dim=1)[:, :Fd]  # (n,Fd)
        else:
            noise_va = torch.rand(n, A, device=self.device, generator=self.rng)
            perm_va = noise_va.argsort(dim=1)        # (n,A)
            visible_agent = torch.zeros(n, Fd, dtype=torch.long, device=self.device)
            visible_agent[:, :A] = perm_va
            if Fd > A:
                extra = torch.randint(0, A, (n, Fd - A),
                                      device=self.device,
                                      generator=self.rng)
                visible_agent[:, A:] = extra

        self.food_visible_agent[idxs] = visible_agent

        # collection stage / target food
        self.collection_stage[idxs] = 0
        self.current_target_food_id[idxs] = spawn_order[:, 0]

        # bookkeeping
        self.curr_steps[idxs] = 0
        self.dones_batch[idxs] = False
        self.trunc_batch[idxs] = False
        self.total_bump[idxs] = 0
        self.cum_rewards[idxs] = 0
        self.episode_len[idxs] = 0


    # ---------------------- Item spawning (TOP/BOTTOM BANDS) ---------------------
    def _maybe_spawn_items(self):
        """
        At each step, spawn any items whose spawn_step == curr_steps
        and which have not yet spawned.

        Spawn rules:
          - Prefer cells in a vertical band near the TOP or BOTTOM:
                y in [0 .. band] or [G-1-band .. G-1]
            where band = rf_radius = (image_size - 1) // 2.
          - Item f in env b appears within the receptive field of its
            designated agent food_visible_agent[b,f], and (ideally)
            outside the RF of all other agents, so only that agent sees
            it when it spawns.
          - RF radius derived from cfg.image_size (Chebyshev).
        """
        B, G, Fd, A = self.B, self.cfg.grid_size, self.cfg.num_foods, self.cfg.num_agents
        P = G * G
        rf_radius = (self.cfg.image_size - 1) // 2

        # which items should spawn now
        curr = self.curr_steps.view(B, 1).expand(B, Fd)
        spawn_now = (~self.food_spawned) & (self.food_spawn_step == curr)  # (B,Fd)
        if not spawn_now.any():
            return

        # occupancy
        blocked = torch.zeros(B, P, dtype=torch.bool, device=self.device)

        if self.wall_pos.numel() > 0:
            idx_w = self._make_index(self.wall_pos, G)
            b_idx_w = torch.arange(B, device=self.device).unsqueeze(1).expand_as(idx_w)
            blocked.view(B, -1)[b_idx_w[:, :idx_w.size(1)], idx_w] = True

        idx_a = self._make_index(self.agent_pos, G)
        b_idx_a = torch.arange(B, device=self.device).unsqueeze(1).expand_as(idx_a)
        blocked.view(B, -1)[b_idx_a, idx_a] = True

        active_food_mask = self.food_spawned & (~self.food_done)
        if active_food_mask.any():
            pos_f = self.food_pos.clone()
            pos_f[~active_food_mask] = 0
            idx_f = self._make_index(pos_f, G)
            b_idx_f = torch.arange(B, device=self.device).unsqueeze(1).expand_as(idx_f)
            blocked.view(B, -1)[b_idx_f[:, :idx_f.size(1)], idx_f] = True

        # precompute in-RF masks
        gy = self._grid_y.view(1, 1, P)  # (1,1,P)
        gx = self._grid_x.view(1, 1, P)  # (1,1,P)

        ay = self.agent_pos[..., 0].unsqueeze(-1)  # (B,A,1)
        ax = self.agent_pos[..., 1].unsqueeze(-1)  # (B,A,1)

        dy_all = (gy - ay).abs()  # (B,A,P)
        dx_all = (gx - ax).abs()
        in_rf_all = (torch.max(dy_all, dx_all) <= rf_radius)  # (B,A,P)

        va = self.food_visible_agent  # (B,Fd)
        in_rf_all_exp = in_rf_all.unsqueeze(1).expand(B, Fd, A, P)

        vis_oh = F.one_hot(va, num_classes=A).to(torch.bool).unsqueeze(-1)  # (B,Fd,A,1)

        in_rf_vis = (in_rf_all_exp & vis_oh).any(dim=2)       # (B,Fd,P)
        others_mask = (~vis_oh)                               # (B,Fd,A,1)
        in_rf_others_any = (in_rf_all_exp & others_mask).any(dim=2)  # (B,Fd,P)

        free = (~blocked).unsqueeze(1).expand(B, Fd, P)       # (B,Fd,P)

        # --- NEW: restrict to TOP/BOTTOM BANDS instead of single rows ---
        band = rf_radius
        gy_flat = self._grid_y  # (P,)
        top_band = (gy_flat <= band)
        bottom_band = (gy_flat >= (G - 1 - band))
        band_mask = (top_band | bottom_band).view(1, 1, P)    # (1,1,P)
        band_mask = band_mask.expand(B, Fd, P)                # (B,Fd,P)

        free_band = free & band_mask  # free cells only in top/bottom bands

        # 1) best: in RF of visible agent, not in RF of others, in band, free
        candidate1 = in_rf_vis & (~in_rf_others_any) & free_band
        has_valid1 = candidate1.any(dim=-1)  # (B,Fd)

        # 2) fallback: in RF of visible agent, in band, free
        candidate2 = in_rf_vis & free_band
        has_valid2 = candidate2.any(dim=-1)  # (B,Fd)  (kept for symmetry; not used directly but harmless)

        # choose between candidate1 and candidate2 for each (B,Fd)
        cond1 = has_valid1.unsqueeze(-1)
        valid_temp = torch.where(cond1, candidate1, candidate2)  # (B,Fd,P)
        has_valid_temp = valid_temp.any(dim=-1)                  # (B,Fd)

        # 3) if still no valid cell in band but we must spawn, use any free cell in band
        fallback_band = free_band
        cond_band = (~has_valid_temp & spawn_now).unsqueeze(-1)
        valid_band = torch.where(cond_band, fallback_band, valid_temp)
        has_valid_band = valid_band.any(dim=-1)

        # 4) final fallback: if even band is impossible (e.g. blocked), use any free cell
        cond_free = (~has_valid_band & spawn_now).unsqueeze(-1)
        valid = torch.where(cond_free, free, valid_band)  # (B,Fd,P)

        # random choice among valid cells
        rand = torch.rand(B, Fd, P, device=self.device, generator=self.rng)
        weights = rand * valid.to(rand.dtype)
        chosen_flat = weights.argmax(dim=-1)  # (B,Fd)

        y = chosen_flat // G
        x = chosen_flat % G
        new_pos = torch.stack([y, x], dim=-1)  # (B,Fd,2)

        mask3 = spawn_now.unsqueeze(-1)
        self.food_pos = torch.where(mask3, new_pos, self.food_pos)
        self.food_spawned = self.food_spawned | spawn_now

    # ---------------------- Step ----------------------
    @torch.no_grad()
    def step(self, actions, auto_reset=True):
        """
        actions: Dict[int, int] or LongTensor [A] or [B,A]
          0: up, 1: down, 2: left, 3: right, 4: pick_up

        Returns (same as ScoreG TorchForagingEnv):
          obs, rewards, dones, truncs, infos
        where obs = self.observe() = (img, pos).
        """
        cfg = self.cfg
        B, G, A, Fd = self.B, cfg.grid_size, cfg.num_agents, cfg.num_foods

        # normalize actions to LongTensor [B,A]
        if isinstance(actions, dict):
            a = torch.tensor(
                [int(actions[i]) for i in range(A)],
                device=self.device,
                dtype=torch.long
            ).unsqueeze(0).expand(self.B, -1)
        elif isinstance(actions, torch.Tensor):
            a = actions.to(self.device).long()
            if a.dim() == 1:
                a = a.unsqueeze(0).expand(self.B, -1)
        else:
            raise TypeError("actions must be dict or torch.Tensor")

        # increment time
        self.curr_steps += 1

        # Spawn any items that should appear at this step
        self._maybe_spawn_items()

        # Are agents allowed to move/pick yet? (frozen for t <= freeze_dur)
        can_act_env = (self.curr_steps > cfg.freeze_dur)  # (B,)
        can_act = can_act_env.view(B, 1).expand(B, A)  # (B,A)

        # movement phase (0..3)
        move_mask = (a < 4) & can_act
        deltas = torch.zeros(self.B, A, 2, dtype=torch.long, device=self.device)
        for i in range(4):
            sel = (a == i) & move_mask
            if sel.any():
                deltas[sel] = self._delta[i]

        proposed = self.agent_pos + deltas
        proposed = torch.clamp(proposed, 0, G - 1)

        # occupancy at current positions
        occ_agents_now = torch.full((self.B, G, G), -1, dtype=torch.long, device=self.device)
        idx_flat_agents = self._make_index(self.agent_pos, G)
        b_idx = torch.arange(self.B, device=self.device).unsqueeze(1).expand_as(idx_flat_agents)
        occ_agents_now.view(self.B, -1)[b_idx, idx_flat_agents] = torch.arange(A, device=self.device).unsqueeze(0).expand_as(idx_flat_agents)

        # walls occupancy
        occ_walls = torch.zeros(self.B, G, G, dtype=torch.bool, device=self.device)
        if self.wall_pos.numel() > 0:
            idx_w = self._make_index(self.wall_pos, G)
            b_idx_w = torch.arange(self.B, device=self.device).unsqueeze(1).expand_as(idx_w)
            occ_walls.view(self.B, -1)[b_idx_w[:, :idx_w.size(1)], idx_w] = True

        # food occupancy (only spawned & not done)
        occ_foods = torch.zeros(self.B, G, G, dtype=torch.bool, device=self.device)
        active_food_mask = self.food_spawned & (~self.food_done)
        if active_food_mask.any():
            pos_f = self.food_pos.clone()
            pos_f[~active_food_mask] = 0
            idx_f = self._make_index(pos_f, G)
            b_idx_f = torch.arange(self.B, device=self.device).unsqueeze(1).expand_as(idx_f)
            occ_foods.view(self.B, -1)[b_idx_f[:, :idx_f.size(1)], idx_f] = True

        # invalid targets
        tgt_flat = self._make_index(proposed, G)
        tgt_is_wall = occ_walls.view(self.B, -1)[b_idx, tgt_flat]
        tgt_has_agent = (occ_agents_now.view(self.B, -1)[b_idx, tgt_flat] != -1)
        tgt_has_food = occ_foods.view(self.B, -1)[b_idx, tgt_flat]

        dup = tgt_flat.unsqueeze(2).eq(tgt_flat.unsqueeze(1))  # (B,A,A)
        dup = dup.triu(diagonal=1).any(-1)                     # (B,A)

        can_move = move_mask & (~tgt_is_wall) & (~tgt_has_agent) & (~tgt_has_food) & (~dup)

        # bumps: moving into another agent
        self.total_bump += (move_mask & tgt_has_agent).sum(dim=1)

        # apply movement
        self.agent_pos = torch.where(can_move.unsqueeze(-1), proposed, self.agent_pos)

        # pickup phase (action==4), within l2<=sqrt(2)
        pick_mask = (a == 4) & can_act
        picked_food = torch.full((self.B,), -1, dtype=torch.long, device=self.device)

        if pick_mask.any():
            # pairwise dist^2 between agents and active foods: (B,A,Fd)
            ay = self.agent_pos[..., 0].unsqueeze(-1)  # (B,A,1)
            ax = self.agent_pos[..., 1].unsqueeze(-1)
            fy = self.food_pos[..., 0].unsqueeze(1)    # (B,1,Fd)
            fx = self.food_pos[..., 1].unsqueeze(1)

            dy2 = (ay - fy).to(torch.float32).pow(2)
            dx2 = (ax - fx).to(torch.float32).pow(2)
            near = (dy2 + dx2) <= 2.0  # l2<=sqrt(2)

            active_food_mask_exp = active_food_mask.unsqueeze(1)  # (B,1,Fd)
            attempt = pick_mask.unsqueeze(-1) & near & active_food_mask_exp

            # combined strength per food
            strength_sum = attempt.to(torch.float32).sum(dim=1) * float(cfg.agent_strength)  # (B,Fd)
            can_lift = (strength_sum >= float(cfg.food_strength_required)) & active_food_mask  # (B,Fd)

            if can_lift.any():
                # choose a lifted food per env (lowest id if multiple)
                first_idx = torch.argmax(can_lift.to(torch.int64), dim=1)  # (B,)
                mask_any = can_lift.any(dim=1)
                picked_food[mask_any] = first_idx[mask_any]

        # rewards, termination
        rewards = torch.zeros(self.B, A, dtype=torch.float32, device=self.device)

        collected = (picked_food >= 0)
        if collected.any():
            target = self.current_target_food_id  # (B,)

            correct = collected & (picked_food == target)
            wrong = collected & (~correct)

            # wrong-order pickups → immediate failure
            if wrong.any():
                rewards[wrong] -= 1.0
                self.dones_batch = self.dones_batch | wrong

            # correct pickups
            if correct.any():
                pf = picked_food.clone()
                b_idx_c = torch.nonzero(correct, as_tuple=False).view(-1)
                self.food_done[b_idx_c, pf[b_idx_c]] = True

                stage = self.collection_stage.clone()
                # stage 0 → move to 1, set next target
                stage0 = (stage == 0) & correct
                if stage0.any():
                    b0 = torch.nonzero(stage0, as_tuple=False).view(-1)
                    self.collection_stage[b0] = 1
                    self.current_target_food_id[b0] = self.spawn_order[b0, 1]
                    rewards[b0] += 0.5
                    
                # stage 1 → success (both items collected in order)
                stage1 = (stage == 1) & correct
                if stage1.any():
                    b1 = torch.nonzero(stage1, as_tuple=False).view(-1)
                    self.collection_stage[b1] = 2
                    if cfg.time_pressure:
                        bonus = ((cfg.max_steps - self.curr_steps[b1]) / self._max_steps_f).clamp_min(0.0)
                    else:
                        bonus = torch.zeros_like(self.curr_steps[b1], dtype=torch.float32)
                    rewards[b1] += (0.5 + bonus).unsqueeze(-1)
                    self.dones_batch[b1] = True

        # timeout
        timed_out = (self.curr_steps >= self.cfg.max_steps)
        if timed_out.any():
            rewards[timed_out] -= 1.0
            self.trunc_batch = self.trunc_batch | timed_out
            self.dones_batch = self.dones_batch | timed_out

        finished = self.dones_batch | self.trunc_batch
        dones_out = self.dones_batch.clone()
        truncs_out = self.trunc_batch.clone()

        # normalize rewards and update stats
        rewards = rewards / float(cfg.reward_scale)
        self.cum_rewards += rewards
        self.episode_len += 1

        # infos (keep compatible with ScoreG env; extend if needed)
        infos: Dict[int, dict] = None

        if auto_reset and finished.any():
            self._reset_indices(finished)

        obs = self.observe()
        return obs, rewards, dones_out, truncs_out, infos

    def close(self):
        pass
