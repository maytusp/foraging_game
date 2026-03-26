# visualize_torch_env.py
from __future__ import annotations
import os
import numpy as np
import torch
import pygame
from moviepy.editor import ImageSequenceClip

from constants import *              # expects: cell_size, WHITE, BLACK, agent_images, spinach, cattle (optional)
from keyboard_control_mm import *    # expects: get_agent_action(events, agent_id) -> int in [0..6]

from environments.mixed_motive_v1 import TorchForagingEnv, EnvConfig

# --------------------------- Pygame helpers ---------------------------

def init_pygame(grid_size: int):
    pygame.init()
    screen_size = grid_size * cell_size
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Torch Foraging Environment")
    font = pygame.font.SysFont(None, 22)
    return screen, font

def draw_grid(screen, grid_size: int):
    screen.fill(WHITE)
    size = grid_size * cell_size
    for x in range(0, size, cell_size):
        for y in range(0, size, cell_size):
            pygame.draw.rect(screen, BLACK, pygame.Rect(x, y, cell_size, cell_size), 1)

def blit_or_rect(screen, img, x, y, fallback_color=(0, 200, 0)):
    if img is not None:
        screen.blit(img, (x, y))
    else:
        pygame.draw.rect(screen, fallback_color, (x, y, cell_size, cell_size))

def draw_agent_with_heading(screen, x, y, heading, color=(0, 0, 255)):
    """
    Draws a triangle pointing in heading: 0 up, 1 right, 2 down, 3 left.
    (x,y) is top-left corner of the grid cell.
    """
    cx = x + cell_size // 2
    cy = y + cell_size // 2
    r  = max(6, cell_size // 3)
    if heading == 0:   # up
        pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
    elif heading == 1: # right
        pts = [(cx + r, cy), (cx - r, cy - r), (cx - r, cy + r)]
    elif heading == 2: # down
        pts = [(cx, cy + r), (cx - r, cy - r), (cx + r, cy - r)]
    else:              # left
        pts = [(cx - r, cy), (cx + r, cy - r), (cx + r, cy + r)]
    pygame.draw.polygon(screen, color, pts)

def draw_fov_rect(screen, G: int, ax: int, ay: int, heading: int, K: int, color=(255, 0, 0)):
    """
    Draw a KxK rectangle *in front of the agent* based on its heading.
    Near edge touches the agent cell; rectangle is clamped to the grid.
    heading: 0 up, 1 right, 2 down, 3 left
    """
    r = K // 2
    # Compute extents in grid coordinates
    if heading == 0:  # up
        top    = ay - K
        bottom = ay - 1
        left   = ax - r
        right  = ax + r
    elif heading == 1:  # right
        top    = ay - r
        bottom = ay + r
        left   = ax + 1
        right  = ax + K
    elif heading == 2:  # down
        top    = ay + 1
        bottom = ay + K
        left   = ax - r
        right  = ax + r
    else:  # left
        top    = ay - r
        bottom = ay + r
        left   = ax - K
        right  = ax - 1

    # Clamp to grid bounds
    top    = max(0, top)
    left   = max(0, left)
    bottom = min(G - 1, bottom)
    right  = min(G - 1, right)

    # Convert to pixel rectangle
    width_cells  = max(0, right - left + 1)
    height_cells = max(0, bottom - top + 1)
    if width_cells > 0 and height_cells > 0:
        rect = pygame.Rect(left * cell_size, top * cell_size, width_cells * cell_size, height_cells * cell_size)
        pygame.draw.rect(screen, color, rect, 2)

# --------------------------- Visualization ---------------------------

def visualize_torch_environment(env: TorchForagingEnv, step_i: int, screen, font, b: int = 0):
    """Render env b (usually 0) to pygame and return an RGB frame (H,W,3)."""
    pygame.display.set_caption(f"Torch Environment - Step {int(step_i)}")
    G   = env.cfg.grid_size
    K   = env.cfg.image_size
    A   = env.cfg.num_agents

    draw_grid(screen, G)

    # Walls
    if env.wall_pos.numel() > 0 and env.wall_pos.size(1) > 0:
        for w in range(env.wall_pos.size(1)):
            wy = int(env.wall_pos[b, w, 0].item()); wx = int(env.wall_pos[b, w, 1].item())
            pygame.draw.rect(screen, (120, 120, 120), (wx * cell_size, wy * cell_size, cell_size, cell_size))

    # Light items -> spinach
    spinach_img = spinach if 'spinach' in globals() else None
    L = env.cfg.num_light
    if L > 0:
        for f in range(L):
            if bool(env.light_done[b, f].item()):
                continue
            fy = int(env.light_pos[b, f, 0].item()); fx = int(env.light_pos[b, f, 1].item())
            x, y = fx * cell_size, fy * cell_size
            blit_or_rect(screen, spinach_img, x, y, fallback_color=(0, 200, 0))

    # Heavy animals -> cattle
    cattle_img = cattle if 'cattle' in globals() else None
    H = env.cfg.num_heavy
    if H > 0:
        for f in range(H):
            if bool(env.heavy_done[b, f].item()):
                continue
            fy = int(env.heavy_pos[b, f, 0].item()); fx = int(env.heavy_pos[b, f, 1].item())
            x, y = fx * cell_size, fy * cell_size
            blit_or_rect(screen, cattle_img, x, y, fallback_color=(230, 140, 20))

    # Agents + egocentric FOV (KxK *in front* of agent)
    tool_name = {0: "Sickle", 1: "Sword", 2: "Bait"}
    for aid in range(A):
        ay = int(env.agent_pos[b, aid, 0].item())
        ax = int(env.agent_pos[b, aid, 1].item())
        hd = int(env.agent_dir[b, aid].item()) if hasattr(env, "agent_dir") else 0
        tl = int(env.agent_tool[b, aid].item()) if hasattr(env, "agent_tool") else 0

        x, y = ax * cell_size, ay * cell_size

        # agent sprite or triangle with heading
        imgs = agent_images if 'agent_images' in globals() else None
        if imgs is not None and len(imgs) > aid and imgs[aid] is not None:
            screen.blit(imgs[aid], (x, y))
        else:
            draw_agent_with_heading(screen, x, y, hd, color=(0, 0, 255))

        # draw FOV rectangle in front (not centered)
        draw_fov_rect(screen, G, ax, ay, hd, K, color=(255, 0, 0))

        # HUD text: id, heading, tool
        txt = font.render(f"A{aid} dir:{hd} tool:{tool_name.get(tl, tl)}", True, (10, 10, 10))
        screen.blit(txt, (x + 2, y + 2))

    pygame.display.flip()
    frame = pygame.surfarray.array3d(screen)            # (W,H,3)
    return frame.transpose((1, 0, 2))                   # (H,W,3)

# --------------------------- Play loop ---------------------------

def run_human_play(
    env: TorchForagingEnv,
    num_episodes: int = 5,
    max_steps: int = None,
    visualize: bool = True,
    human_play: bool = True,
    save_videos: bool = True,
    save_dir: str = "logs_torch",
    fps: int = 5,
):
    os.makedirs(save_dir, exist_ok=True)
    assert env.B == 1, "Interactive play expects num_envs=1."

    screen, font = init_pygame(env.cfg.grid_size)
    clock = pygame.time.Clock()
    _ = env.observe()  # prime obs

    for ep in range(num_episodes):
        frames = []
        running = True

        steps_lim = max_steps if max_steps is not None else env.cfg.max_steps
        for step_i in range(steps_lim):
            if not running:
                break

            if visualize:
                frame = visualize_torch_environment(env, step_i, screen, font, b=0)
                frames.append(frame)

            # Collect actions for each agent
            if human_play:
                agent_actions = [None] * env.cfg.num_agents
                while not all(a is not None for a in agent_actions):
                    events = pygame.event.get()
                    for event in events:
                        if event.type == pygame.QUIT:
                            running = False
                            break
                    if not running:
                        break

                    for i in range(env.cfg.num_agents):
                        if agent_actions[i] is None:
                            a = get_agent_action(events, i)
                            if a is not None:
                                agent_actions[i] = int(a)
                    clock.tick(30)

                if not running:
                    break
                acts_t = torch.tensor(agent_actions, device=env.device, dtype=torch.long)
            else:
                acts_t = torch.randint(0, 7, (env.cfg.num_agents,), device=env.device)

            (img, locs, msg_masks), rew, done, trunc, info = env.step(acts_t)
            print("agent0 obs: \n", img[0,0])
            print("agent1 obs: \n", img[0,1])
            print("msg mask: \n", msg_masks)
            print("msg mask sum dim 1: \n", msg_masks.unsqueeze(-1).sum((1,2)).clamp(max=1))
            print("step reward:", rew[0].tolist())
            if done.any() or trunc.any():
                try:
                    print("step reward:", rew[0].tolist())
                except Exception:
                    pass
                break

            clock.tick(60)

        if visualize and save_videos and len(frames) > 0:
            out_path = os.path.join(save_dir, f"ep{ep}.mp4")
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(out_path, codec="libx264", audio=False, verbose=False, logger=None)
            print(f"Saved: {out_path}")

    pygame.quit()

# --------------------------- Run directly ---------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = EnvConfig(
        grid_size=10,
        image_size=5,           # egocentric 5x5
        comm_field=5,
        num_agents=2,
        num_light=4,
        num_heavy=2,
        num_walls=0,
        max_steps=50,
        mode="train",
        seed=42,
    )
    env = TorchForagingEnv(cfg, device=device, num_envs=1)

    run_human_play(
        env,
        num_episodes=5,
        max_steps=cfg.max_steps,
        visualize=True,
        human_play=True,
        save_videos=False,
        save_dir="logs_torch",
        fps=6,
    )
