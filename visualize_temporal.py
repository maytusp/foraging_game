# Edit: 20Dec2024
from constants import *
from keyboard_control import *
# from environments.environment_single import *

import numpy as np
import random
import time

import supersuit as ss
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
    # for i in range(environment.home_size):
    #     for j in range(environment.home_size):
    #         home_rect = pygame.Rect((environment.home_position[1] + j) * cell_size, (environment.home_position[0] + i) * cell_size, cell_size, cell_size)
    #         pygame.draw.rect(screen, HOME_COLOR, home_rect)

    # Draw agents
    for agent_id, agent in enumerate(environment.agent_maps):
        x, y = agent.position[1] * cell_size, agent.position[0] * cell_size
        screen.blit(agent_images[agent_id], (x, y))
        
        # energy_text = font.render(f"Agent: {int(agent.id)}", True, BLACK)
        # screen.blit(energy_text, (x, y - 20))  # Display text slightly above the agent

        # Draw a square centered at the agent position representing visual fieldss
        square_size = environment.image_size * cell_size
        top_left_x = x + cell_size // 2 - square_size // 2
        top_left_y = y + cell_size // 2 - square_size // 2
        square_rect = pygame.Rect(top_left_x, top_left_y, square_size, square_size)
        pygame.draw.rect(screen, (255, 0, 0), square_rect, 2)

    # Draw foods
    for food in environment.foods:
        if food.visible:
            x, y = food.position[1] * cell_size, food.position[0] * cell_size
            index = food.food_type - 1
            screen.blit(food_images[index], (x, y))
            
            if len(food.carried) > 0:
                # Display "pick up" message above the food
                pickup_text = font.render("Pick Up", True, (0, 255, 0))  # Green text
                screen.blit(pickup_text, (x, y+20))  # Display text slightly above the food
    # Draw wall
    for wall in environment.wall_list:
        x, y = wall.position[1] * cell_size, wall.position[0] * cell_size
           # Draw the rectangle
        pygame.draw.rect(screen, (255, 0, 0), (x, y, cell_size, cell_size))
        # screen.blit(, (x, y))
    pygame.display.flip()

    frame = pygame.surfarray.array3d(screen)
    return frame

def nonzero_sum_channels(obs):
    print(obs.shape)
    # Extract the last N_att channels (excluding the first channel)
    att_channels = obs[1:, :, :]  # Shape: (N_att, 5, 5)

    # Compute the sum over the (5,5) spatial dimensions
    channel_sums = np.sum(att_channels, axis=(1, 2))  # Shape: (N_att,)

    # Convert to binary indicator (1 if sum is nonzero, 0 otherwise)
    binary_mask = (channel_sums != 0).astype(int)

    return binary_mask

if __name__ == "__main__":
    NUM_STEPS = 10000
    NUM_EPISODES = 20
    HUMAN_PLAY = True
    VISUALIZE = True
    from environments.pickup_temporal import *
    # from environments.pickup_high_v1 import *
    # env = Environment(agent_visible=False, partner_food_visible=False)
    env = Environment(image_size=3, grid_size=5, N_i=2, agent_visible=True, use_message=True, num_walls=3)
    envs = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(envs, 1, num_cpus=0, base_class="gymnasium")
    clock = pygame.time.Clock()
    for ep in range(NUM_EPISODES):
        observations = env.reset()
        frames = []
        # print("Obs", observations[0].shape)
        for step in range(NUM_STEPS):
            print(f"--- Step {step + 1} ---")
            agent_actions = [None] * env.num_agents  # Stores actions for each agent
            agent_messages =  [None] * env.num_agents
            if VISUALIZE:
                single_env = envs.vec_envs[0].unwrapped.par_env
                frame = visualize_environment(single_env, step)
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
                        agent_messages[i] = np.random.randint(1,4)
            if env.num_agents == 1:
                agent_actions = agent_actions[0]
            agent_actions = list(np.array(agent_actions)-1)
            agent_messages = agent_messages
            observations, rewards, dones, _, infos = envs.step({"action": agent_actions, "message": agent_messages})
            # print("reward", rewards)
            # print(f"agent0: \n {nonzero_sum_channels(observations[0]['image'])}")
            print(f"Agent0 obs: \n {observations['image'][0][0]}")
            print(f"Agent1 obs: \n {observations['image'][1][0]}")
            print(f"Agent0 M: \n {observations['message']}")
            # print(f"Agent1 M: \n {observations['message']}")
            
            # print(f"score: {observations[1]['image']}")
            # print(f"Can communicate:  {observations[0]['is_m_sent']}")
            # print(f"agent1: \n {nonzero_sum_channels(observations[1]['image'])}")
            # if rewards[0] != 0 or rewards[1] != 0:
            #     print("reward", rewards)
            if isinstance(dones,bool):
                if dones:
                    break
            else: 
                if dones[0]:
                    for info in infos:
                        if "terminal_observation" in info:
                            # for info in each_infos:
                            if "episode" in info:
                                print(info["episode"]["r"])

                    break

        if VISUALIZE:
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_videofile(f"logs/ep{ep}.mp4", codec="libx264")