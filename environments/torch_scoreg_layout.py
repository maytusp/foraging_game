# torch_scoreg_layout.py
# 26 March 2026
# Add layout structure to the foraging game (default size would be 13x13)
# Limit Communication Range
# Limit Communication Step as the agents take so many steps and communication is hard to optimise
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn.functional as F
simple_layout_5x5 = """
AAAAA
FFFFF
.....
FFFFF
AAAAA
"""


warmup_layout_7x7 = """
WWWWWWWWW
WAAAAAAAW
WFFFFFFFW
WFFFFFFFW
W.......W
WFFFFFFFW
WFFFFFFFW
WAAAAAAAW
WWWWWWWWW
"""


simple_layout_7x7 = """
AAAAAAA
FFFFFFF
FFFFFFF
W..W..W
FFFFFFF
FFFFFFF
AAAAAAA
"""

simple_layout_9x9 = """
WWAAAAAWW
FFFFFFFFF
FFFFFFFFF
WW..W..WW
WW..W..WW
WW..W..WW
FFFFFFFFF
FFFFFFFFF
WWAAAAAWW
"""



simple_layout_13x13 = """
AAAAAAAAAAAAA
FFFFFFFFFFFFF
FFFFFWWWFFFFF
.............
.............
.............
WW.........WW
.............
.............
.............
FFFFFWWWFFFFF
FFFFFFFFFFFFF
AAAAAAAAAAAAA
"""


simple_layout_17x17 = """
AAAAAAAAAAAAAAAAA
FFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFF
.................
.................
......WWWWW......
.................
.................
WWWW.........WWWW
.................
.................
......WWWWW......
.................
.................
FFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFF
AAAAAAAAAAAAAAAAA
"""

simple_layout_21x21 = """
AAAAAAAAAAAAAAAAAAAAA
FFFFFFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFFFFFF
WWWWWW.........WWWWWW
.....................
.....................
......WWWWWWWWW......
.....................
.....................
WWWWWW.........WWWWWW
.....................
.....................
......WWWWWWWWW......
.....................
WWWWWW.........WWWWWW
.....................
FFFFFFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFFFFFF
FFFFFFFFFFFFFFFFFFFFF
AAAAAAAAAAAAAAAAAAAAA
"""

@dataclass
class EnvConfig:
    grid_size: int = 5
    image_size: int = 7            # local obs crop (odd)
    comm_field: int = 7
    communication_steps: int = 6
    num_channels: int = 2
    num_agents: int = 2
    num_foods: int = 2             # N_i
    num_walls: int = 0 # initial active walls
    max_walls: int = 20
    max_steps: int = grid_size**2
    agent_visible: bool = False    # show other agents in obs
    food_energy_fully_visible: bool = False
    identical_item_obs: bool = False
    time_pressure: bool = True
    n_words: int = 10
    message_length: int = 1
    mode: Literal["train", "test"] = "train"

    spawn_mode: int = 0  # 0: Easy, 1: Medium, 2: Hard # to make the training converge easier

    test_moderate_score: bool = False
    seed: int = 42
    # constants (match your code where possible)
    N_val: int = 255
    N_att: int = 1
    agent_strength: int = 3
    food_strength_required: int = 6
    reward_scale: float = 1.0
    use_compile: bool = True
    ascii_layout: Optional[str] = None

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
        self.agent_visible = cfg.agent_visible

        # --- RNG
        g = torch.Generator(device="cpu")
        g.manual_seed(cfg.seed)
        self._cpu_gen = g  # for random init on CPU then move to device

        # active wall count tensor
        self.active_wall_count = torch.zeros(1, device=self.device, dtype=torch.long)

        # initialize current layout
        self._set_layout(cfg.ascii_layout)

        # --- persistent tensors (allocated on device)
        self.agent_pos = torch.zeros(self.B, A, 2, dtype=torch.long, device=self.device)  # (y,x)
        self.agent_energy = torch.full((self.B, A), 20, dtype=torch.float32, device=self.device)
        self.food_pos = torch.zeros(self.B, Fd, 2, dtype=torch.long, device=self.device)
        self.food_done = torch.zeros(self.B, Fd, dtype=torch.bool, device=self.device)
        self.food_energy = torch.zeros(self.B, Fd, dtype=torch.float32, device=self.device)
        self.target_food_id = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.score_visible_to_agent = torch.zeros(self.B, Fd, dtype=torch.long, device=self.device)  # which agent sees which food


        
        self.spawn_mode = torch.tensor([cfg.spawn_mode], device=self.device, dtype=torch.long)

        self.curr_steps = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.dones_batch = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.trunc_batch = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.total_bump = torch.zeros(self.B, dtype=torch.long, device=self.device)

        self.comm_started = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.comm_steps_used = torch.zeros(self.B, dtype=torch.long, device=self.device)

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


    def set_spawn_mode(self, mode: int):
            """
            Set food spawn difficulty.
            0: Easy (Both same side, 0 or G-1)
            1: Medium (One edge 0/G-1, One middle G//2)
            2: Hard (Opposite sides, 0 and G-1)
            """
            assert mode in [0, 1, 2], "Mode must be 0, 1, or 2"
            self.spawn_mode.fill_(mode)

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
        if self._wall_mask_2d.any():
            occ += self._wall_mask_2d.unsqueeze(0).to(torch.float32) * self._occ_wall

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
        
    def _parse_ascii_layout(self, ascii_layout: Optional[str]) -> torch.Tensor:
        """
        Returns wall mask of shape (G, G), dtype=bool
        True = wall

        Also builds:
        self._agent_spawn_flat
        self._food_spawn_flat
        self._agent_spawn_top_flat
        self._agent_spawn_bottom_flat
        """
        G = self.cfg.grid_size

        if ascii_layout is None:
            mask = torch.zeros(G, G, dtype=torch.bool, device=self.device)

            all_cells = torch.arange(G * G, device=self.device, dtype=torch.long)
            self._agent_spawn_flat = all_cells
            self._food_spawn_flat = all_cells

            ys = all_cells // G
            self._agent_spawn_top_flat = all_cells[ys < (G // 2)]
            self._agent_spawn_bottom_flat = all_cells[ys > (G // 2)]
            if self._agent_spawn_top_flat.numel() == 0:
                self._agent_spawn_top_flat = all_cells
            if self._agent_spawn_bottom_flat.numel() == 0:
                self._agent_spawn_bottom_flat = all_cells

            return mask

        rows = [row.strip() for row in ascii_layout.strip().splitlines() if row.strip()]
        if len(rows) != G:
            raise ValueError(f"ascii_layout must have exactly {G} rows, got {len(rows)}")

        for i, row in enumerate(rows):
            if len(row) != G:
                raise ValueError(f"ascii_layout row {i} must have length {G}, got {len(row)}")

        mask = torch.zeros(G, G, dtype=torch.bool, device=self.device)
        agent_spawn = torch.zeros(G, G, dtype=torch.bool, device=self.device)
        food_spawn = torch.zeros(G, G, dtype=torch.bool, device=self.device)

        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == "W":
                    mask[y, x] = True
                elif ch == ".":
                    pass
                elif ch == "A":
                    # `A` marks agent spawn cells, but foods may also use those
                    # cells as long as the cell is not currently occupied.
                    agent_spawn[y, x] = True
                    food_spawn[y, x] = True
                elif ch == "F":
                    food_spawn[y, x] = True
                else:
                    raise ValueError(
                        f"Unsupported layout char '{ch}' at ({y}, {x}); use only 'W', '.', 'A', or 'F'"
                    )

        self._agent_spawn_flat = torch.nonzero(agent_spawn.view(-1), as_tuple=False).view(-1)
        self._food_spawn_flat = torch.nonzero(food_spawn.view(-1), as_tuple=False).view(-1)

        food_ys = self._food_spawn_flat // G
        self._food_spawn_top_flat = self._food_spawn_flat[food_ys < (G // 2)]
        self._food_spawn_bottom_flat = self._food_spawn_flat[food_ys > (G // 2)]

        # fall back to generic empty cells if A/F are not provided
        free_flat = torch.nonzero(~mask.view(-1), as_tuple=False).view(-1)
        if self._agent_spawn_flat.numel() == 0:
            self._agent_spawn_flat = free_flat
        if self._food_spawn_flat.numel() == 0:
            self._food_spawn_flat = free_flat

        if self._food_spawn_flat.numel() > 0:
            food_ys = self._food_spawn_flat // G
            self._food_spawn_top_flat = self._food_spawn_flat[food_ys < (G // 2)]
            self._food_spawn_bottom_flat = self._food_spawn_flat[food_ys > (G // 2)]

            if self._food_spawn_top_flat.numel() == 0:
                self._food_spawn_top_flat = self._food_spawn_flat
            if self._food_spawn_bottom_flat.numel() == 0:
                self._food_spawn_bottom_flat = self._food_spawn_flat

        agent_ys = self._agent_spawn_flat // G
        self._agent_spawn_top_flat = self._agent_spawn_flat[agent_ys < (G // 2)]
        self._agent_spawn_bottom_flat = self._agent_spawn_flat[agent_ys > (G // 2)]

        # fallback if one side is empty
        if self._agent_spawn_top_flat.numel() == 0:
            self._agent_spawn_top_flat = self._agent_spawn_flat
        if self._agent_spawn_bottom_flat.numel() == 0:
            self._agent_spawn_bottom_flat = self._agent_spawn_flat

        return mask

    def _sample_distinct_from_pool(self, pool_flat: torch.Tensor, n_envs: int, count: int) -> torch.Tensor:
        """
        Sample `count` distinct cells per env from a provided flat-index pool.
        Returns (n_envs, count, 2) as (y, x)
        """
        num_pool = pool_flat.numel()
        if count > num_pool:
            raise ValueError(
                f"Not enough cells in spawn pool: need {count}, have {num_pool}."
            )

        noise = torch.rand(n_envs, num_pool, device=self.device, generator=self.rng)
        perm = noise.argsort(dim=1)[:, :count]
        chosen_flat = pool_flat.unsqueeze(0).expand(n_envs, -1).gather(1, perm)

        G = self.cfg.grid_size
        y = chosen_flat // G
        x = chosen_flat % G
        return torch.stack([y, x], dim=-1).long()

    def _sample_distinct_free_cells(self, n_envs: int, count: int) -> torch.Tensor:
        """
        Sample `count` distinct positions per env from free cells.
        Returns: (n_envs, count, 2) as (y, x)
        """
        num_free = self._free_cells_flat.numel()
        if count > num_free:
            raise ValueError(
                f"Not enough free cells for sampling: need {count}, have {num_free}. "
                f"Reduce num_agents/num_foods or use a less dense wall layout."
            )

        # Random permutation per env, take first `count`
        noise = torch.rand(n_envs, num_free, device=self.device, generator=self.rng)
        perm = noise.argsort(dim=1)[:, :count]                          # (n_envs, count)
        chosen_flat = self._free_cells_flat.unsqueeze(0).expand(n_envs, -1).gather(1, perm)

        G = self.cfg.grid_size
        y = chosen_flat // G
        x = chosen_flat % G
        return torch.stack([y, x], dim=-1).long()

    def _set_layout(self, ascii_layout: Optional[str]):
        """
        Update the active wall layout for all environments.
        This does not reset episodes by itself.
        """
        G = self.cfg.grid_size

        # 2D wall mask used by stepping/observations
        self._wall_mask_2d = self._parse_ascii_layout(ascii_layout)   # (G, G), bool
        self._free_cells_flat = torch.nonzero(~self._wall_mask_2d.view(-1), as_tuple=False).view(-1)

        # keep wall_pos only for visualization / compatibility
        fixed_wall_coords = torch.nonzero(self._wall_mask_2d, as_tuple=False).long()  # (W, 2)
        self.fixed_num_walls = fixed_wall_coords.size(0)
        self.active_wall_count.fill_(self.fixed_num_walls)

        if self.fixed_num_walls > 0:
            self.wall_pos = fixed_wall_coords.unsqueeze(0).expand(self.B, -1, -1).clone()
        else:
            self.wall_pos = torch.empty(self.B, 0, 2, dtype=torch.long, device=self.device)


    def set_layout(self, ascii_layout: Optional[str], reset_now: bool = False):
        """
        Public API for training code.
        ascii_layout=None means no walls.
        """
        self.cfg.ascii_layout = ascii_layout
        self._set_layout(ascii_layout)

        if reset_now:
            full_mask = torch.ones((self.B,), dtype=torch.bool, device=self.device)
            self._reset_indices(full_mask)

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
        


        # Mask Attribute bt agents who can see those items
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

        # Mask observation of other agents
        if not(self.agent_visible):
            agent_map = torch.full((B, G, G), -1, dtype=torch.long, device=self.device)   # (B, G, G)
            agent_idx_flat = self._make_index(self.agent_pos, G)                 # (B, A) (A <= G*G)
            agent_bidx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, A)  # (B, A)
            agent_id   = torch.arange(A, device=self.device).unsqueeze(0).expand(B, A)  # (B, A)
            agent_map.view(B,-1)[agent_bidx, agent_idx_flat] = agent_id

            agent_one_hot = F.one_hot(agent_map.clamp_min(0), num_classes=A).to(occ_crops.dtype)  # (B,G,G,A)
            agent_one_hot = agent_one_hot * (agent_map >= 0).unsqueeze(-1).to(agent_one_hot.dtype) # set non-agent position to be zero
            agent_one_hot_crop = self._crop_around(agent_one_hot, self.agent_pos, pad_val=0.0) # (B,A,A,K,K)
            agent_channel_idx = torch.arange(A, device=self.device).view(1,A,1,1,1)
            agent_channel_idx = agent_channel_idx.expand(B,-1,1,K,K)
            occ_mask = agent_one_hot_crop.gather(2, agent_channel_idx) # (B,A,K,K)
            occ_crops = occ_mask * occ_crops

        img = torch.cat([occ_crops, attr_crops], dim=2)  # (B,A,C,K,K)
        pos = self.agent_pos.to(torch.float32)             # (B,A,2)


        comm_inrange = self._compute_comm_inrange()  # (B,A,A)

        # communication is active only after first meeting, and only for limited steps
        comm_active = self.comm_started & (self.comm_steps_used <= self.cfg.communication_steps)  # (B,)
        comm_mask = comm_inrange & comm_active.view(B, 1, 1)

        return img, pos, comm_mask

    def _compute_comm_inrange(self) -> torch.Tensor:
        """
        Returns raw pairwise communication range mask of shape (B, A, A),
        based only on distance and excluding self-communication.
        """
        A = self.cfg.num_agents

        ay = self.agent_pos[..., 0].unsqueeze(2)  # (B,A,1)
        ax = self.agent_pos[..., 1].unsqueeze(2)
        by = self.agent_pos[..., 0].unsqueeze(1)  # (B,1,A)
        bx = self.agent_pos[..., 1].unsqueeze(1)

        cheb = torch.maximum((ay - by).abs(), (ax - bx).abs())  # (B,A,A)

        R = (self.cfg.comm_field - 1) // 2
        comm_inrange = (cheb <= R)

        eye = torch.eye(A, dtype=torch.bool, device=self.device).unsqueeze(0)
        comm_inrange = comm_inrange & (~eye)
        return comm_inrange

    def _reset_indices(self, mask: torch.Tensor):
        cfg, B, A, Fd, G, nb = self.cfg, self.B, self.cfg.num_agents, self.cfg.num_foods, self.cfg.grid_size, self.num_neigh
        mask = mask.bool()
        if not mask.any(): return
        idxs = torch.nonzero(mask, as_tuple=False).view(-1)
        n = idxs.numel()

        score_pool = self._score_list  # (M,)
        M = score_pool.numel()

        # per-env choose Fd distinct scores (vectorized "perm" via argsort of random noise)
        noise = torch.rand(n, M, device=self.device, generator=self.rng)
        perm_per_env = noise.argsort(dim=1, descending=False)[:, :Fd]   # (n, Fd)
        self.food_energy[idxs] = score_pool[perm_per_env]               # (n, Fd)

        # per-env order of Fd items
        order_noise = torch.rand(n, Fd, device=self.device, generator=self.rng)
        order = order_noise.argsort(dim=1, descending=False)            # (n, Fd)
        self.score_visible_to_agent[idxs] = order % A                   # (n, Fd)

        # target food = argmax
        self.target_food_id[idxs] = torch.argmax(self.food_energy[idxs], dim=1)

        # ------------------------------
        # Agent spawn: only on A cells, and opposite sides
        # agent 0 / agent 1 swap top-bottom randomly per env
        # ------------------------------
        if A != 2:
            raise ValueError("Current opposite-side A spawning assumes num_agents == 2")

        top_pos = self._sample_distinct_from_pool(self._agent_spawn_top_flat, n, 1).squeeze(1)       # (n,2)
        bottom_pos = self._sample_distinct_from_pool(self._agent_spawn_bottom_flat, n, 1).squeeze(1) # (n,2)

        top_first = torch.randint(0, 2, (n,), device=self.device, generator=self.rng).bool()

        agent0 = torch.where(top_first.unsqueeze(-1), top_pos, bottom_pos)
        agent1 = torch.where(top_first.unsqueeze(-1), bottom_pos, top_pos)

        self.agent_pos[idxs, 0] = agent0
        self.agent_pos[idxs, 1] = agent1

        # ------------------------------
        # Food spawn: for A=2, foods are split across top/bottom according to
        # which agent can see them. With num_foods=4 this gives 2 top + 2 bottom.
        # `A` cells are allowed in the food pool, but the currently occupied
        # agent cells are still excluded below.
        # ------------------------------
        top_agent_id = torch.where(
            top_first,
            torch.zeros(n, dtype=torch.long, device=self.device),
            torch.ones(n, dtype=torch.long, device=self.device),
        )  # (n,)

        owners = self.score_visible_to_agent[idxs]                  # (n, Fd)
        food_on_top = owners.eq(top_agent_id.unsqueeze(1))          # (n, Fd) bool

        food_pos = torch.empty(n, Fd, 2, dtype=torch.long, device=self.device)

        G = self.cfg.grid_size
        top_flat_all = self._food_spawn_top_flat
        bottom_flat_all = self._food_spawn_bottom_flat

        for e in range(n):
            # How many foods should go to each side in this env?
            top_mask_e = food_on_top[e]                             # (Fd,)
            n_top = int(top_mask_e.sum().item())
            n_bottom = Fd - n_top

            # Agent cells to exclude
            agent0_flat = self.agent_pos[idxs[e], 0, 0] * G + self.agent_pos[idxs[e], 0, 1]
            agent1_flat = self.agent_pos[idxs[e], 1, 0] * G + self.agent_pos[idxs[e], 1, 1]

            top_pool = top_flat_all[(top_flat_all != agent0_flat) & (top_flat_all != agent1_flat)]
            bottom_pool = bottom_flat_all[(bottom_flat_all != agent0_flat) & (bottom_flat_all != agent1_flat)]

            if n_top > top_pool.numel():
                raise ValueError(
                    f"Not enough top food spawn cells for num_foods={Fd}. "
                    f"Need {n_top}, have {top_pool.numel()}."
                )
            if n_bottom > bottom_pool.numel():
                raise ValueError(
                    f"Not enough bottom food spawn cells for num_foods={Fd}. "
                    f"Need {n_bottom}, have {bottom_pool.numel()}."
                )

            # Sample distinct top positions
            if n_top > 0:
                perm_top = torch.randperm(top_pool.numel(), device=self.device, generator=self.rng)[:n_top]
                chosen_top = top_pool[perm_top]
                top_y = chosen_top // G
                top_x = chosen_top % G
                top_pos_e = torch.stack([top_y, top_x], dim=-1).long()
                food_pos[e, top_mask_e] = top_pos_e

            # Sample distinct bottom positions
            if n_bottom > 0:
                perm_bottom = torch.randperm(bottom_pool.numel(), device=self.device, generator=self.rng)[:n_bottom]
                chosen_bottom = bottom_pool[perm_bottom]
                bottom_y = chosen_bottom // G
                bottom_x = chosen_bottom % G
                bottom_pos_e = torch.stack([bottom_y, bottom_x], dim=-1).long()
                food_pos[e, ~top_mask_e] = bottom_pos_e

        self.food_pos[idxs] = food_pos
        self.food_done[idxs] = False

        self.agent_energy[idxs] = 20.0
        self.curr_steps[idxs] = 0
        self.dones_batch[idxs] = False
        self.trunc_batch[idxs] = False
        self.total_bump[idxs] = 0
        self.cum_rewards[idxs] = 0
        self.episode_len[idxs] = 0

        self.comm_started[idxs] = False
        self.comm_steps_used[idxs] = 0

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
        occ_walls = self._wall_mask_2d.unsqueeze(0).expand(self.B, -1, -1)  # (B,G,G)

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

        # ---------------- Communication logic ----------------
        comm_inrange = self._compute_comm_inrange()          # (B,A,A)
        met_now = comm_inrange.flatten(start_dim=1).any(dim=1)                   # (B,)

        # start communication when agents first meet
        newly_started = (~self.comm_started) & met_now
        self.comm_started = self.comm_started | newly_started

        # once started, consume one communication step per env step
        self.comm_steps_used[self.comm_started] += 1

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
                bonus = ((self.cfg.max_steps - self.curr_steps[right]) / self._max_steps_f).clamp_min(0.0) if self.cfg.time_pressure else torch.tensor(0.0, device=self.device)
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
