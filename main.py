from constants import *
from keyboard_control import *
from environment import *

import numpy as np
import random
import time

import pygame
from moviepy.editor import ImageSequenceClip

# Initialize and run the environment
font = pygame.font.SysFont(None, 24)
# Pygame-based visualization function
def visualize_environment(environment, step):
    # Initialize pygame if not already initialized
    pygame.init()
    
    screen_size = GRID_SIZE * cell_size
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
    for agent_id, agent in enumerate(environment.agents):
        x, y = agent.position[1] * cell_size, agent.position[0] * cell_size
        screen.blit(agent_images[agent_id], (x, y))
        
        energy_text = font.render(f"Energy: {int(agent.energy)}", True, BLACK)
        screen.blit(energy_text, (x, y - 20))  # Display text slightly above the agent


    # Draw foods
    for food in environment.foods:
        x, y = food.position[1] * cell_size, food.position[0] * cell_size
        index = food.strength_required - 1
        screen.blit(food_images[index], (x, y))
        
        if len(food.carried) > 0:
            # Display "pick up" message above the food
            pickup_text = font.render("Pick Up", True, (0, 255, 0))  # Green text
            screen.blit(pickup_text, (x, y - 20))  # Display text slightly above the food

    pygame.display.flip()

    frame = pygame.surfarray.array3d(screen)
    return frame
env = Environment()
clock = pygame.time.Clock()
frames = []

for step in range(NUM_STEPS):
    print(f"--- Step {step + 1} ---")
    agent_actions = [None] * NUM_AGENTS  # Stores actions for each agent
    frame = visualize_environment(env, step)
    # Loop until both agents have provided input
    while not all(agent_actions):
        # Update the display to show environment
        
        # clock.tick(1)  # Adjust the loop frequency
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                break

        # Get actions from keyboard for each agent
        for i in range(NUM_AGENTS):
            agent_actions[i] = agent_actions[i] or get_agent_action(events, i)

    frames.append(frame.transpose((1, 0, 2)))
    done = env.step(agent_actions)
    if done:
        break

clip = ImageSequenceClip(frames, fps=10)
clip.write_videofile("simulation_video.mp4", codec="libx264")