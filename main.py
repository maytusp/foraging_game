from constants import *
from keyboard_control import *

import numpy as np
import random
import time

import pygame
from moviepy.editor import ImageSequenceClip

# Environment Parameters
GRID_SIZE = 10  # Size of the grid world
NUM_AGENTS = 1  # Number of agents
NUM_FOODS = 8  # Number of foods
HOME_POSITION = (1, 1)  # Coordinates of the home
MAX_MESSAGE_LENGTH = 10  # Example message length limit
AGENT_ATTRIBUTES = [5, 5, 5, 5]  # All agents have the same attributes
AGENT_STRENGTH = 3
AGENT_ENERGY = 50
MAX_REQUIRED_STRENGTH = 6
HOME_SIZE = 2
# Define colors for agents and the home area
AGENT_COLOR = "green"
HOME_COLOR = "lightblue"

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

    # Draw foods
    for food in environment.foods:
        x, y = food.position[1] * cell_size, food.position[0] * cell_size
        index = food.strength_required - 1
        screen.blit(food_images[index], (x, y))

    pygame.display.flip()

    frame = pygame.surfarray.array3d(screen)
    return frame

# Define the classes
class Agent:
    def __init__(self, id, position, strength, max_energy):
        self.id = id
        self.position = position
        self.strength = strength
        self.energy = max_energy
        self.carrying_food = None
        self.done = False
        self.messages = []
        # Agent observation field adjusted to (24, 4) for the 5x5 grid view, exluding agent position
        self.perception = {"observation": np.zeros((24, 4)), "receive_message": np.zeros((NUM_AGENTS, MAX_MESSAGE_LENGTH))}
    
    def observe(self, environment):
        # Define the 5x5 field of view around the agent, excluding its center
        perception_data = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue  # Skip the agent's own position
                x, y = self.position[0] + dx, self.position[1] + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    obj = environment.grid[x, y]
                    if obj is None:
                        perception_data.append([0, 0, 0, 0])  # Empty grid
                    else:
                        perception_data.append(obj.attribute)  # Food or other object attributes
                else:
                    perception_data.append([0, 0, 0, 0])  # Out-of-bounds grid (treated as empty)
        
        self.perception["observation"] = np.array(perception_data)
    

    def send_message(self, env):
        return env.message[0,:] #TODO Use neural network to send message

class Food:
    def __init__(self, position, strength_required, id):
        self.position = position
        self.strength_required = strength_required
        self.carried = []
        self.attribute = self.generate_attributes(strength_required)
        self.energy_score = 10 * strength_required
        self.id = id
        self.done = False

    def generate_attributes(self, strength_required):
        # Return unique attributes based on the food's strength requirement
        attribute_mapping = {
            1: [1, 1, 1, 1],
            2: [1, 1, 2, 2],
            3: [1, 1, 3, 3],
            4: [2, 1, 1, 4],
            5: [3, 2, 4, 3],
            6: [3, 3, 2, 3],

        }
        return np.array(attribute_mapping.get(strength_required, [1, 1, 1, 1]))

class Environment:
    def __init__(self):
        self.grid = np.full((GRID_SIZE, GRID_SIZE), None)
        self.prev_pos_list = []
        # Initialize agents with uniform attributes
        self.agents = [Agent(i, self.random_position(), AGENT_STRENGTH, AGENT_ENERGY) for i in range(NUM_AGENTS)]
        for agent in self.agents:
            self.grid[agent.position[0], agent.position[1]] = agent

        self.foods = [Food(self.random_position(), random.randint(1, MAX_REQUIRED_STRENGTH), food_id) for food_id in range(NUM_FOODS)]

        for food in self.foods:
            self.grid[food.position[0], food.position[1]] = food

        self.collected_foods = []
        self.message = np.zeros((NUM_AGENTS, MAX_MESSAGE_LENGTH)) # Message that each agent sends, each agent receive N-1 agents' messages

    def update_grid(self):
        self.grid = np.full((GRID_SIZE, GRID_SIZE), None)
        for agent in self.agents:
            if not(agent.done): # If agent is alive
                self.grid[agent.position[0], agent.position[1]] = agent
        for food in self.foods:
            if not(food.done): # If food is not placed at home
                self.grid[food.position[0], food.position[1]] = food

    def min_dist(self,curr_pos, min_distance):
        satisfy = True
        for prev_pos in self.prev_pos_list:
            if self.compute_dist(curr_pos, prev_pos) < min_distance:
                satisfy = False
                break
        return satisfy

    def random_position(self):
        while True:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if self.grid[pos[0], pos[1]] is None and self.min_dist(pos,3):
                self.prev_pos_list.append(pos)
                return pos

    def compute_dist(self, pos1, pos2):
        pos1 = np.array([pos1[0], pos1[1]])
        pos2 = np.array([pos2[0], pos2[1]])
        return np.linalg.norm(pos1 - pos2)
    
    def step(self, agent_actions):
        # One step in the simulation
        # Gather each agent's chosen action for consensus on movement
        actions = []
        for i, agent in enumerate(self.agents):
            if agent.done:
                continue

            action = agent_actions[i]
            actions.append((agent, action))

        # Apply actions with consensus for carrying agents
        consensus_action = None
        carrying_agents = [agent for agent, action in actions if agent.carrying_food]
        if carrying_agents:
            # Ensure all carrying agents agree on the same direction
            consensus_action = actions[0][1] if all(a[1] == actions[0][1] for a in actions if a[0] in carrying_agents) else None
        
        # Process each agent’s action
        for agent, action in actions:
            print("action", action)
            if action in ["up", "down", "left", "right"]:
                # For carrying agents, move only if there's consensus on the direction
                if agent.carrying_food and action != consensus_action: #TODO
                    print(f"Agent {agent.id} couldn't move; consensus required.")
                    continue
                # agent.move(self, action)  # Move with or without food
                # Calculate the new position based on the movement direction
                new_position = {
                    "up": (max(1, agent.position[0] - 1), agent.position[1]),
                    "down": (min(GRID_SIZE - 2, agent.position[0] + 1), agent.position[1]),
                    "left": (agent.position[0], max(1, agent.position[1] - 1)),
                    "right": (agent.position[0], min(GRID_SIZE - 2, agent.position[1] + 1))
                }.get(action, agent.position)
                delta_position = np.subtract(new_position, agent.position)
                old_agent_position = np.array(agent.position)
                new_agent_position = old_agent_position + delta_position

                # Check if the new position is empty and move if it is
                if agent.carrying_food:
                    old_food_position = np.array(agent.carrying_food.position)
                    new_food_position = agent.carrying_food.position + delta_position
                    if (self.grid[new_agent_position[0], new_agent_position[1]] is None or self.grid[new_agent_position[0], new_agent_position[1]].id == agent.carrying_food.id) and \
                        (self.grid[new_food_position[0], new_food_position[1]] is None or self.grid[new_food_position[0], new_food_position[1]].id == agent.id):

                        print("try to mode food to", new_food_position)
                        # Move the food with the agent
                        agent.position = new_agent_position
                        agent.carrying_food.position = new_food_position
                        loss = (1 + 0.2*min(agent.strength, agent.carrying_food.strength_required)) #TODO Adjust the energy based on the actual strength that each agent uses
                        agent.energy -= loss # When carry food, agent lose more energy due to friction
    
                        print(f"Agent {agent.id} lose {loss}.")
                else:
                    if self.grid[new_agent_position[0], new_agent_position[1]] is None:
                        agent.energy -= 1
                        agent.position += delta_position

                if agent.energy <= 0:
                    agent.done = True
                    print(f"Agent {agent.id} died.")

                self.update_grid()
                print("grid", self.grid)

            elif action == "pick_up" and agent.carrying_food is None:
                for food in self.foods:
                    if (self.compute_dist(food.position, agent.position) <= np.sqrt(2)) and len(food.carried) == 0:
                        if food.strength_required <= agent.strength and not food.carried:
                            food.carried.append(agent.id)
                            agent.carrying_food = food
                            print(f"Agent {agent.id} picked up food at {food.position}")
                            break
                
            elif action == "drop" and agent.carrying_food:
                # If agent drops food at home
                if (agent.position[0] in range(HOME_POSITION[0], HOME_POSITION[0] + HOME_SIZE) and 
                    agent.position[1] in range(HOME_POSITION[1], HOME_POSITION[1] + HOME_SIZE)):
                    agent.energy += agent.carrying_food.energy_score
                    print(f"Agent {agent.id} brought food home, current energy: {agent.energy}")
                    agent.carrying_food.position = (-2000,-2000)
                    agent.carrying_food.done = True
                    self.collected_foods.append(agent.carrying_food)
                    
                agent.carrying_food.carried = []
                agent.carrying_food = None

            #TODO Pick up and drop by both agents not only single agent, this is probably difficult
            #TODO Update grid state every step

        # End if all agents are out of energy
        if all(agent.energy <= 0 for agent in self.agents):
            print("All agents have died. Game Over.")
            return True
        
        # End if all food items are collected
        if len(self.collected_foods) == len(self.foods):
            print("All food items are collected")
            return True
        return False

# Initialize and run the environment
env = Environment()
clock = pygame.time.Clock()
frames = []
# Run for a maximum of 100 steps or until all agents are dead
for step in range(200):
    print(f"--- Step {step + 1} ---")


    agent_actions = [None] * NUM_AGENTS  # Stores actions for each agent

    # Loop until both agents have provided input
    while not all(agent_actions):
        # Update the display to show environment
        frame = visualize_environment(env, step)
        frames.append(frame.transpose((1, 0, 2)))
        
        clock.tick(10)  # Adjust the loop frequency
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                break

        # Get actions from keyboard for each agent
        for i in range(NUM_AGENTS):
            agent_actions[i] = agent_actions[i] or get_agent_action(events, i)

    done = env.step(agent_actions)
    if done:
        break

clip = ImageSequenceClip(frames, fps=10)
clip.write_videofile("simulation_video.mp4", codec="libx264")