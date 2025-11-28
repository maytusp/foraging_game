# visualize_torch_scale.py
from __future__ import annotations
import os
import numpy as np
import torch
import pygame
from moviepy.editor import ImageSequenceClip

from constants import *              # expects: cell_size, WHITE, BLACK, agent_images, food_images, etc.
from keyboard_control import *       # expects: get_agent_action(events, agent_id)

# If you placed the class in torch_foraging_env.py:
from environments.torch_scoreg_scale import TorchForagingEnv, EnvConfig

# --------------------------- Pygame helpers ---------------------------

def init_pygame(grid_size: int):
    pygame.init()
    screen_size = grid_size * cell_size
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Torch Foraging Environment")
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

def visualize_torch_environment(env: TorchForagingEnv, step: int, screen, font, b: int = 0):
    """Render the FIRST env in the batch (b=0) to pygame and return an RGB frame (H,W,3)."""
    pygame.display.set_caption(f"Torch Environment - Step {int(step)}")
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

    # Foods
    for f in range(Fd):
        if bool(env.food_done[b, f].item()):
            continue
        fy = int(env.food_pos[b, f, 0].item())
        fx = int(env.food_pos[b, f, 1].item())
        x, y = fx * cell_size, fy * cell_size
        # use image if available, otherwise a green square
        blit_or_rect(screen, food_images if 'food_images' in globals() else None, f % (len(food_images) if 'food_images' in globals() else 1), x, y)

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

        # Optional overlay text
        # energy = float(env.agent_energy[b, aid].item())
        # txt = font.render(f"A{aid}", True, BLACK)
        # screen.blit(txt, (x, y - 18))

    pygame.display.flip()
    frame = pygame.surfarray.array3d(screen)            # (W,H,3)
    return frame.transpose((1, 0, 2))                   # (H,W,3) for moviepy

# --------------------------- Play loop ---------------------------

def run_human_play(
    env: TorchForagingEnv,
    num_episodes: int = 5,
    max_steps: int = 200,
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

            # Step the env
            (obs, locs, masks), rew, done, trunc, info = env.step(acts_t)
            print(f"----STEP {step}-----")
            print(obs[0,0])
            # (optional) debug prints like your CPU script
            # print(f"Agent0 occ channel:\n{obs[0]['image'][0]}")
            # if env.cfg.num_agents > 1: print(f"Agent1 occ channel:\n{obs[1]['image'][0]}")
            if done.any() or trunc.any():
                print(rew)
                break

            clock.tick(60)

        # if visualize and save_videos and len(frames) > 0:
        #     out_path = os.path.join(save_dir, f"ep{ep}.mp4")
        #     clip = ImageSequenceClip(frames, fps=fps)
        #     clip.write_videofile(out_path, codec="libx264")
        #     print(f"Saved: {out_path}")

    pygame.quit()

# --------------------------- Run directly ---------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = EnvConfig(
        grid_size=13,
        image_size=7,
        comm_field=7,
        num_agents=2,
        num_foods=4,
        num_walls=20,
        max_steps=50,
        agent_visible=True,
        food_energy_fully_visible=False,
        mode="train",
        seed=42,
    )
    # num_envs=1 for interactive play
    env = TorchForagingEnv(cfg, device=device, num_envs=1)

    run_human_play(
        env,
        num_episodes=5,
        max_steps=200,
        visualize=True,
        human_play=True,
        save_videos=True,
        save_dir="logs_torch",
        fps=5,
    )
