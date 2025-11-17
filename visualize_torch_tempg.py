# visualize_torch_temporalg.py
from __future__ import annotations
import os
import numpy as np
import torch
import pygame
from moviepy.editor import ImageSequenceClip

from constants import *              # expects: cell_size, WHITE, BLACK, agent_images, food_images, etc.
from keyboard_control import *       # expects: get_agent_action(events, agent_id)

# Import TemporalG env
from environments.torch_pickup_temporal import TorchTemporalEnv, EnvConfig

# --------------------------- Pygame helpers ---------------------------

def init_pygame(grid_size: int):
    pygame.init()
    screen_size = grid_size * cell_size
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Torch TemporalG Environment")
    font = pygame.font.SysFont(None, 24)
    return screen, font

def draw_grid(screen, grid_size: int):
    screen.fill(WHITE)
    size = grid_size * cell_size
    for x in range(0, size, cell_size):
        for y in range(0, size, cell_size):
            pygame.draw.rect(screen, BLACK, pygame.Rect(x, y, cell_size, cell_size), 1)

def blit_or_rect(screen, img_list, idx, x, y, color=(0, 200, 0)):
    if img_list is not None and len(img_list) > idx and img_list[idx] is not None:
        screen.blit(img_list[idx], (x, y))
    else:
        pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))

# --------------------------- Visualization ---------------------------

def visualize_torch_environment(env: TorchTemporalEnv, step: int, screen, font, b: int = 0):
    """Render the FIRST env in the batch (b=0) to pygame and return an RGB frame (H,W,3)."""
    pygame.display.set_caption(f"Torch TemporalG Environment - Step {int(step)}")
    G   = env.cfg.grid_size
    K   = env.cfg.image_size
    A   = env.cfg.num_agents
    Fd  = env.cfg.num_foods
    r   = K // 2

    draw_grid(screen, G)

    # Walls
    if env.wall_pos.size(1) > 0:
        for w in range(env.wall_pos.size(1)):
            wy = int(env.wall_pos[b, w, 0].item())
            wx = int(env.wall_pos[b, w, 1].item())
            pygame.draw.rect(screen, (255, 0, 0), (wx * cell_size, wy * cell_size, cell_size, cell_size))

    # Foods (only spawned & not yet collected)
    for f in range(Fd):
        if not bool(env.food_spawned[b, f].item()):
            continue
        if bool(env.food_done[b, f].item()):
            continue
        fy = int(env.food_pos[b, f, 0].item())
        fx = int(env.food_pos[b, f, 1].item())
        x, y = fx * cell_size, fy * cell_size
        # use image if available, otherwise a green square
        if 'food_images' in globals():
            img_list = food_images
            idx = f % (len(food_images) if len(food_images) > 0 else 1)
        else:
            img_list = None
            idx = 0
        blit_or_rect(screen, img_list, idx, x, y)

    # Agents + FOV
    for aid in range(A):
        ay = int(env.agent_pos[b, aid, 0].item())
        ax = int(env.agent_pos[b, aid, 1].item())
        x, y = ax * cell_size, ay * cell_size
        blit_or_rect(screen, agent_images if 'agent_images' in globals() else None, aid, x, y, color=(0, 0, 255))

        # FOV square (KxK) centered on the agent
        square_size = K * cell_size
        top_left_x = x + cell_size // 2 - square_size // 2
        top_left_y = y + cell_size // 2 - square_size // 2
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(top_left_x, top_left_y, square_size, square_size), 2)

        # Optional overlay: show stage / spawn info if you want
        # stage = int(env.collection_stage[b].item())
        # txt = font.render(f"A{aid} S{stage}", True, BLACK)
        # screen.blit(txt, (x, y - 18))

    pygame.display.flip()
    frame = pygame.surfarray.array3d(screen)            # (W,H,3)
    return frame.transpose((1, 0, 2))                   # (H,W,3) for moviepy

# --------------------------- Play loop ---------------------------

def run_human_play(
    env: TorchTemporalEnv,
    num_episodes: int = 5,
    max_steps: int = None,
    visualize: bool = True,
    human_play: bool = True,
    save_videos: bool = True,
    save_dir: str = "logs_torch_temporalg",
    fps: int = 5,
):
    os.makedirs(save_dir, exist_ok=True)
    assert env.B == 1, "Interactive play expects num_envs=1."

    screen, font = init_pygame(env.cfg.grid_size)
    clock = pygame.time.Clock()

    # Initial observation (not strictly needed, but kept for parity)
    _ = env.observe()

    for ep in range(num_episodes):
        frames = []
        running = True

        for step in range(max_steps):
            if not running:
                break

            if visualize:
                frame = visualize_torch_environment(env, step, screen, font, b=0)
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
                            a = get_agent_action(events, i)  # typically 1..5 mapping
                            if a is not None:
                                agent_actions[i] = a
                    clock.tick(30)  # poll at 30 FPS
                if not running:
                    break

                # map to 0..4 (0:up,1:down,2:left,3:right,4:pick_up)
                actions = [int(a) - 1 for a in agent_actions]
                acts_t = torch.tensor(actions, device=env.device, dtype=torch.long)
            else:
                # random policy for testing
                acts_t = torch.randint(0, 5, (env.cfg.num_agents,), device=env.device)

            # Step the TemporalG env
            (imgs, locs, masks), rew, done, trunc, info = env.step(acts_t)
            print(f"Image \n  {imgs[0,0]}")
            print(f"Location \n  {locs[0]}")
            print(f"masks \n  {masks[0]}")
            # Optionally inspect reward / stage
            if done.any() or trunc.any():
                print(f"Episode {ep}, step {step}, reward: {rew}")
                break

            clock.tick(60)
        # # --- SAVE VIDEO (if any frames) ---
        # if visualize and save_videos and len(frames) > 0:
        #     # Make sure fps is a real number
        #     print(frames)
        #     out_path = os.path.join(save_dir, f"ep{ep}.mp4")
        #     clip = ImageSequenceClip(frames, fps=5)
        #     # And also pass fps explicitly to write_videofile
        #     clip.write_videofile(out_path, codec="libx264")
        #     print(f"Saved: {out_path}")



    pygame.quit()

# --------------------------- Run directly ---------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = EnvConfig(
        grid_size=9,
        image_size=5,
        num_agents=2,
        num_foods=2,
        num_walls=4,
        max_steps=40,       # TemporalG: T_max = 20
        agent_visible=False,
        mode="test",
        seed=42,
    )
    # num_envs=1 for interactive play
    env = TorchTemporalEnv(cfg, device=device, num_envs=1)

    run_human_play(
        env,
        num_episodes=20,
        max_steps=cfg.max_steps,
        visualize=True,
        human_play=True,
        save_videos=False,
        save_dir="logs_torch_temporalg",
        fps=5,
    )
