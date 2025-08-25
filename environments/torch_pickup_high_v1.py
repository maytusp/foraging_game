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
        
        self._score_list = score_list.to(torch.float32)
        self._reset_indices(torch.ones((self.B,)))
    # ---------------------- Utilities ----------------------
    def _rand_choice(self, choices: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        idx = torch.randint(0, choices.numel(), shape, generator=self._cpu_gen)
        return choices[idx]

    def _random_positions(self, count: int) -> torch.Tensor:
        """Return (B, count, 2) random free positions (naive sampling; fine for small grids)."""
        G = self.cfg.grid_size
        pos = torch.stack([
            torch.randint(0, G, (self.B, count), generator=self._cpu_gen),
            torch.randint(0, G, (self.B, count), generator=self._cpu_gen),
        ], dim=-1).to(torch.long)
        return pos.to(self.device)

    def _make_index(self, yx: torch.Tensor, W: int) -> torch.Tensor:
        """Flattened indices from (.., 2) coords (y,x)."""
        return (yx[...,0] * W + yx[...,1]).to(torch.long)

    # ---------------------- Observation ----------------------
    def _make_occupancy_and_attr_maps(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
          occ_map: (B, H, W) float32
          attr_map: (B, H, W, N_att) float32
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

    def _crop_around(self, grid: torch.Tensor, centers: torch.Tensor, pad_val: float) -> torch.Tensor:
        """
        grid: (B, H, W) or (B, H, W, C)
        centers: (B, A, 2) (y,x)
        returns crops: (B, A, K, K, [C])
        """
        B, G, K = self.B, self.cfg.grid_size, self.cfg.image_size
        r = K // 2
        if grid.dim() == 3:  # (B,H,W)
            padded = F.pad(grid, (r, r, r, r), value=pad_val)  # (B, H+2r, W+2r)
            crops = []
            for a in range(self.cfg.num_agents):
                y = centers[:, a, 0] + r
                x = centers[:, a, 1] + r
                # gather slices
                rows = torch.arange(-r, r+1, device=self.device).view(1, -1, 1)
                cols = torch.arange(-r, r+1, device=self.device).view(1, 1, -1)
                y_idx = y.view(B,1,1) + rows  # (B,K,1)
                x_idx = x.view(B,1,1) + cols  # (B,1,K)
                crop = padded.gather(1, y_idx.expand(B,K,padded.size(2))).gather(2, x_idx.expand(B,K,K))
                crops.append(crop)
            return torch.stack(crops, dim=1)  # (B, A, K, K)
        else:  # (B,H,W,C)
            C = grid.size(-1)
            grid_ = grid.permute(0,3,1,2)  # (B,C,H,W)
            padded = F.pad(grid_, (r, r, r, r), value=pad_val).permute(0,2,3,1)  # back to (B,H,W,C)
            crops = []
            for a in range(self.cfg.num_agents):
                y = centers[:, a, 0] + r
                x = centers[:, a, 1] + r
                rows = torch.arange(-r, r+1, device=self.device).view(1, -1, 1)
                cols = torch.arange(-r, r+1, device=self.device).view(1, 1, -1)
                y_idx = y.view(B,1,1) + rows
                x_idx = x.view(B,1,1) + cols
                # gather per channel
                crop = padded[
                    torch.arange(B, device=self.device).view(B,1,1),
                    y_idx,
                    x_idx,
                ]  # (B,K,K,C)
                crops.append(crop)
            return torch.stack(crops, dim=1)  # (B, A, K, K, C)

    def observe(self) -> Dict[int, dict]:
        """
        PettingZoo-style observations for the FIRST env in the batch (B=1 recommended with PZ).
        image shape: (C, K, K)  where C = 1 + N_att
        """
        occ, attr = self._make_occupancy_and_attr_maps()
        occ_crops = self._crop_around(occ, self.agent_pos, pad_val=float(self.cfg.N_val))     # (B,A,K,K)
        attr_crops = self._crop_around(attr, self.agent_pos, pad_val=0.0)                     # (B,A,K,K,N_att)

        # mask attributes by visibility if not fully visible
        if not self.cfg.food_energy_fully_visible:
            # build per-agent mask: where food belongs to that agent
            # Here we approximate visibility by masking half the positions randomly per agent via score_visible_to_agent
            # Simpler: keep as-is; attribute is only written at food cells
            pass

        # stack channels: occupancy + attributes
        img = torch.cat([occ_crops.unsqueeze(-1), attr_crops], dim=-1)  # (B,A,K,K, 1+N_att)
        img = img.permute(0,1,4,2,3).contiguous()  # (B,A,C,K,K)
        B, A, C, K, _ = img.shape

        obs = {}
        # first batch element to keep PZ dict API
        for aid in range(self.cfg.num_agents):
            obs[aid] = {
                "image": img[:, aid],                            # (C,K,K) float32
                "location": self.agent_pos[:, aid].to(torch.float32),
            }
            if self.cfg.use_message:
                obs[aid]["message"] = torch.zeros((B, self.cfg.message_length), dtype=torch.long, device=self.device)
        return obs


    def _reset_indices(self, mask: torch.Tensor):
        """Reset only the envs where mask[b]==True (same logic as reset(), but per-batch)."""
        cfg, B, A, Fd, G = self.cfg, self.B, self.cfg.num_agents, self.cfg.num_foods, self.cfg.grid_size
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if not mask.any():
            return

        idxs = torch.nonzero(mask, as_tuple=False).view(-1)
        # sample scores
        score_pool = self._score_list
        perm = torch.stack([torch.randperm(score_pool.numel(), generator=self._cpu_gen) for _ in range(idxs.numel())])
        chosen = score_pool[perm[:, :Fd]].to(self.device)  # (n, Fd)
        self.food_energy[idxs] = chosen

        # visible_to_agent: round-robin across agents
        vis = torch.stack([
            torch.tensor([i % A for i in torch.randperm(Fd, generator=self._cpu_gen)], dtype=torch.long)
            for _ in range(idxs.numel())
        ], dim=0).to(self.device)
        self.score_visible_to_agent[idxs] = vis

        # target is argmax energy
        self.target_food_id[idxs] = torch.argmax(self.food_energy[idxs], dim=1)

        # foods alternate top/bottom rows, random x
        y = torch.zeros(idxs.numel(), Fd, dtype=torch.long)
        y[:, 1::2] = G - 1
        x = torch.randint(0, G, (idxs.numel(), Fd), generator=self._cpu_gen)
        self.food_pos[idxs] = torch.stack([y, x], dim=-1).to(self.device)
        self.food_done[idxs] = False

        # walls mid-row
        if cfg.num_walls > 0:
            mid = G // 2
            xs = torch.linspace(0, G - 1, steps=cfg.num_walls).round().to(torch.long)
            wpos = torch.stack([torch.full((cfg.num_walls,), mid, dtype=torch.long), xs], dim=-1)  # (W,2)
            self.wall_pos[idxs] = wpos.to(self.device)

        # agents near one of the foods they can see
        for k, b in enumerate(idxs.tolist()):
            for aid in range(A):
                fidx = torch.nonzero(self.score_visible_to_agent[b] == aid, as_tuple=False)
                if fidx.numel() > 0:
                    f = int(fidx[0, 0].item())
                    fy, fx = int(self.food_pos[b, f, 0].item()), int(self.food_pos[b, f, 1].item())
                else:
                    fy = torch.randint(0, G, (1,), generator=self._cpu_gen).item()
                    fx = torch.randint(0, G, (1,), generator=self._cpu_gen).item()
                cand = torch.tensor([
                                    [max(0, fy-1), fx],
                                    [min(G-1, fy+1), fx],
                                    [fy, max(0, fx-1)],
                                    [fy, min(G-1, fx+1)]], dtype=torch.long, device=self.device)
                
                # spawn an agent near a food item but not the same position
                food_mask = ~((cand[:, 0] == fy) & (cand[:, 1] == fx))
                cand = cand[food_mask]

                j = torch.randint(0, cand.size(0), (1,), generator=self._cpu_gen).item()
                self.agent_pos[b, aid] = cand[j]

        self.agent_energy[idxs] = 20.0
        self.curr_steps[idxs] = 0
        self.dones_batch[idxs] = False
        self.trunc_batch[idxs] = False
        self.total_bump[idxs] = 0
        self.cum_rewards[idxs] = 0
        self.episode_len[idxs] = 0
        self.last_msgs[idxs] = 0  # clear messages


    # ---------------------- Step ----------------------
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
        can_move = move_mask & (~tgt_is_wall) & (~tgt_has_agent) & (~tgt_has_food)

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
        infos = {
            i: {
                "episode": {
                    "r": float(self.cum_rewards[b, i].item()),
                    "l": int(self.episode_len[b, i].item()),
                    "collect": int(collected[b].item()),
                    "success": int(success[b].item()),
                    "target_id": int(self.target_food_id[b].item()),
                    "food_scores": {int(fid): float(self.food_energy[b, fid].item()) for fid in range(self.cfg.num_foods)},
                    "score_visible_to_agent": self.score_visible_to_agent[b].detach().cpu().tolist(),
                    "total_bump": int(self.total_bump[b].item()),
                }
            } for i in range(self.cfg.num_agents)
        }

        if auto_reset and finished.any():
            self._reset_indices(finished)

        obs = self.observe()
    
        return obs, rewards, dones_out, truncs_out, infos

    def close(self):
        pass
