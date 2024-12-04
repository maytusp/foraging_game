from constants import *
from keyboard_control import *
from environment import *

import numpy as np
import random
import time

import pygame
from moviepy.editor import ImageSequenceClip

# Initialize and run the environment

# Pygame-based visualization function
def visualize_environment(environment, step):
    # Initialize pygame if not already initialized
    pygame.init()
    font = pygame.font.SysFont(None, 24)
    screen_size = environment.grid_size * cell_size
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption(f"Environment at Step {step}")

    # Fill background
    screen.fill(WHITE)
    
    # Draw grid
    for x in range(0, screen_size, cell_size):
        for y in range(0, screen_size, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(screen, BLACK, rect, 1)

    # Draw home area
    for i in range(HOME_SIZE):
        for j in range(HOME_SIZE):
            home_rect = pygame.Rect((HOME_POSITION[1] + j) * cell_size, (HOME_POSITION[0] + i) * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, HOME_COLOR, home_rect)

    # Draw agents
    for agent_id, agent in enumerate(environment.agent_maps):
        x, y = agent.position[1] * cell_size, agent.position[0] * cell_size
        screen.blit(agent_images[agent_id], (x, y))
        
        energy_text = font.render(f"Energy: {int(agent.energy)}", True, BLACK)
        screen.blit(energy_text, (x, y - 20))  # Display text slightly above the agent

        # Draw a square centered at the agent position representing visual fieldss
        square_size = 5 * cell_size
        top_left_x = x + cell_size // 2 - square_size // 2
        top_left_y = y + cell_size // 2 - square_size // 2
        square_rect = pygame.Rect(top_left_x, top_left_y, square_size, square_size)
        pygame.draw.rect(screen, (173, 216, 255), square_rect, 2)

    # Draw foods
    for food in environment.foods:
        x, y = food.position[1] * cell_size, food.position[0] * cell_size
        index = food.food_type - 1
        screen.blit(food_images[index], (x, y))
        
        if len(food.carried) > 0:
            # Display "pick up" message above the food
            pickup_text = font.render("Pick Up", True, (0, 255, 0))  # Green text
            screen.blit(pickup_text, (x, y - 20))  # Display text slightly above the food

    pygame.display.flip()

    frame = pygame.surfarray.array3d(screen)
    return frame
    
if __name__ == "__main__":
    NUM_STEPS = 10000
    NUM_EPISODES = 3
    HUMAN_PLAY = True
    VISUALIZE = True
    env = Environment()
    clock = pygame.time.Clock()
    for ep in range(NUM_EPISODES):
        observations = env.reset()
        frames = []
        # print("Obs", observations[0].shape)
        for step in range(NUM_STEPS):
            print(f"--- Step {step + 1} ---")
            agent_actions = [None] * env.num_agents  # Stores actions for each agent
            if VISUALIZE:
                frame = visualize_environment(env, step)
                frames.append(frame.transpose((1, 0, 2)))
            if HUMAN_PLAY:
                while not all(agent_actions):
                    events = pygame.event.get()
                    for event in events:
                        if event.type == pygame.QUIT:
                            running = False
                            break

                    # Get actions from keyboard for each agent
                    for i in range(env.num_agents):
                        agent_actions[i] = agent_actions[i] or get_agent_action(events, i)
            if env.num_agents == 1:
                agent_actions = agent_actions[0]
            observations, rewards, dones, _, _ = env.step(agent_actions, int_action=False)
            # print("reward", rewards)
            # print(observations)
            # if rewards[0] != 0 or rewards[1] != 0:
            #     print("reward", rewards)
            if dones[0]:
                print("return", env.cumulative_rewards)
                break

        if VISUALIZE:
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_videofile(f"vids/ep{ep}.mp4", codec="libx264")