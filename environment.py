import pygame
import numpy as np
import random
import time

from constants import *
from keyboard_control import *

# Environment Parameters
GRID_SIZE = 10  # Size of the grid world
NUM_AGENTS = 1  # Number of agents
NUM_FOODS = 4  # Number of foods
HOME_POSITION = (1, 1)  # Coordinates of the home
MAX_MESSAGE_LENGTH = 10  # Example message length limit
AGENT_ATTRIBUTES = [10, 10, 10, 10]  # All agents have the same attributes
AGENT_STRENGTH = 3
AGENT_ENERGY = 30

MAX_REQUIRED_STRENGTH = 6
HOME_SIZE = 2

# Reward Hyperparameters
energy_punishment = -30
collect_all_reward = 200
pickup_reward = 10
drop_punishment = -10
drop_reward = 1 # multiplying with energy
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
    
    def observe(self, environment): #TODO Check this again
        # Define the 5x5 field of view around the agent, excluding its center
        perception_data = []
        for dy in range(-2, 3):
            row = []
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    row.append([0, 0, 0, 0])  # agent's own position
                    continue

                x, y = self.position[0] + dx, self.position[1] + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    obj = environment.grid[x, y]
                    if obj is None:
                        row.append([0, 0, 0, 0])  # Empty grid
                    elif isinstance(obj, Food): # Observe Food
                        row.append(obj.attribute)
                    elif isinstance(obj, Agent): # Observe Agent
                        row.append(AGENT_ATTRIBUTES)  
                else:
                    row.append([0, 0, 0, 0])  # Out-of-bounds grid (treated as empty)
            perception_data.append(row)
        
        return np.array(perception_data)

    def send_message(self, env):
        return env.message[0,:] #TODO Use neural network to send message

class Food:
    def __init__(self, position, strength_required, id):
        self.position = position
        self.strength_required = strength_required
        self.carried = [] # keep all agents that already picked up this food
        self.pre_carried = [] # keep all agents that try to pick up this food, no need to be successful
        self.attribute = self.generate_attributes(strength_required)
        self.energy_score = 5 * strength_required
        self.id = id
        self.done = False
        self.reduced_strength = 0

    def generate_attributes(self, strength_required):
        # Return unique attributes based on the food's strength requirement
        attribute_mapping = {
            1: [1, 1, 1, 1], # Spinach
            2: [1, 1, 2, 2], # Watermelon
            3: [1, 3, 3, 3], # Strawberry
            4: [2, 4, 1, 4], # Chicken
            5: [2, 6, 4, 3], # Pig
            6: [2, 3, 8, 3], # Cattle

        }
        return np.array(attribute_mapping.get(strength_required, [1, 1, 1, 1]))

class Environment:
    def __init__(self):
        self.reset()
    def reset(self):
        self.grid = np.full((GRID_SIZE, GRID_SIZE), None)
        self.prev_pos_list = []
        # Initialize agents with uniform attributes
        self.agents = [Agent(i, self.random_position(), AGENT_STRENGTH, AGENT_ENERGY) for i in range(NUM_AGENTS)]
        for agent in self.agents:
            self.grid[agent.position[0], agent.position[1]] = agent
        self.foods = [Food(self.random_position(), food_id+2, food_id) for food_id in range(NUM_FOODS)]
        for food in self.foods:
            self.grid[food.position[0], food.position[1]] = food

        self.collected_foods = set()
        self.message = np.zeros((NUM_AGENTS, MAX_MESSAGE_LENGTH)) # Message that each agent sends, each agent receive N-1 agents' messages
        return self.observe()

    def update_grid(self):
        '''
        Update grid position after agents move
        '''
        self.grid = np.full((GRID_SIZE, GRID_SIZE), None)
        for agent in self.agents:
            if not(agent.done): # If agent is alive
                self.grid[agent.position[0], agent.position[1]] = agent
        for food in self.foods:
            if not(food.done): # If food is not placed at home
                self.grid[food.position[0], food.position[1]] = food
                
    def update_food(self):
        '''
        All agents have to pick up food at the same time step.
        '''
        for food in self.foods:
            food.reduced_strength = 0 # clear reduced strenth due to combined strength
            food.pre_carried.clear() # clear list
            food.is_moved = False

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
            if self.grid[pos[0], pos[1]] is None and self.min_dist(pos,3) and self.compute_dist(pos, HOME_POSITION) > 4:
                self.prev_pos_list.append(pos)
                return pos

    def compute_dist(self, pos1, pos2):
        pos1 = np.array([pos1[0], pos1[1]])
        pos2 = np.array([pos2[0], pos2[1]])
        return np.linalg.norm(pos1 - pos2)
    
    def observe(self):
        agent_obs = []
        agent_loc = []
        for agent in self.agents:
            agent_obs.append(agent.observe(self))
            agent_loc.append(agent.position)
        return {"image": agent_obs,"location": agent_loc}

    def int_to_act(self, action):
        '''
        input: action integer tensor frm the moel, the value is from 0 to 5
        output: action string that matches environment
        '''
        action = action.detach().cpu().numpy()
        action_map = {0: "up", 
                    1: "down", 
                    2: "left",
                    3: "right", 
                    4: "pick_up",
                    5: "drop", }
        action_list = []
        for i in range(len(action)): # loop over agents
            action_int = action[i]
            action_list.append(action_map[action_int])

        return action_list


    def step(self, agent_actions):
        # Update food state: Clear all agents if not carried
        self.update_food()

        # One step in the simulation
        # Gather each agent's chosen action for consensus on movement
        actions = []
        self.rewards = np.array([0] * NUM_AGENTS)
        for i, agent in enumerate(self.agents):
            if agent.done:
                continue

            action = agent_actions[i]
            actions.append((agent, action))

        # Consensus action is for agents taking the same food items
        consensus_action = {}
        for food in self.foods:
            if len(food.carried) > 1:
                first_id = food.carried[0]
                consensus_action[food.id] = actions[first_id][1] if all(a[1] == actions[first_id][1] for a in actions if a[0].id in food.carried) else None
        # print("food taken by multiagent", consensus_action.keys())
        # Process each agent’s action
        for agent, action in actions:
            # If an agent is tied to other agents, i.e., picking the same food.
            # Consensus action has to be satisfied for all agents to perform action, move or drop food.  Otherwise, these actions have no effect.
            if agent.carrying_food:
                if agent.carrying_food.id in consensus_action:
                    if consensus_action[agent.carrying_food.id] is None:
                        print(f"Agent {agent.id} couldn't move; consensus required.")
                        continue
            if action in ["up", "down", "left", "right"]:
                agent.energy -= 1
                delta_pos = {'up': np.array([-1,0]),
                            'down': np.array([1,0]),
                            'left': np.array([0,-1]),
                            'right': np.array([0,1]),}
                old_agent_position = np.array(agent.position)
                new_agent_position = old_agent_position + delta_pos[action]

                # Check if the new position is empty and move if it is
                # Note: This code will iterate all the agents that carry the same food.
                # The condition not(agent.carrying_food.is_moved) will make sure all agents are iterated only one time
                # print(f"check {agent.id}")
                if agent.carrying_food and not(agent.carrying_food.is_moved):
                    # print(f"agent {agent.id} is not moved")
                    move = True
                    new_food_position = agent.carrying_food.position + delta_pos[action]
                    new_pos_list = [new_food_position]
                     # the new position of each element has to be unoccupieds
                    for agent_id in agent.carrying_food.carried:
                        new_pos_list.append(self.agents[agent_id].position + delta_pos[action])

                    for id,new_pos in enumerate(new_pos_list):
                        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] > GRID_SIZE-1 or new_pos[1] > GRID_SIZE-1:
                            # print(f"Bounded, {agent.id}'s move = False, because new_pos number {id}")
                            move = False                            
                            break

                        check_grid = self.grid[new_pos[0], new_pos[1]]
                        if isinstance(check_grid, Agent) and (check_grid.id not in agent.carrying_food.carried) or \
                            isinstance(check_grid, Food) and (check_grid.id != agent.carrying_food.id):
                            move = False
                            # print(f"{agent.id}'s move = False because new_pos number {id}")
                            # print(f"first condition= {isinstance(check_grid, Agent) and (check_grid.id not in agent.carrying_food.carried)}")
                            # print(f"second condition= {isinstance(check_grid, Food) and (check_grid.id != agent.carrying_food.id)}")
                            break


                    if move:
                        for agent_id in agent.carrying_food.carried:
                            old_position = self.agents[agent_id].position
                            new_position = self.agents[agent_id].position + delta_pos[action]
                            self.agents[agent_id].position = new_position
                            loss = 0.2*min(self.agents[agent_id].strength, self.agents[agent_id].carrying_food.strength_required)
                            self.agents[agent_id].energy -= loss # When carry food, agent lose more energy due to friction
                        if not(agent.carrying_food.is_moved):
                            # print(f"{agent.carrying_food.id} moves to {new_food_position}")
                            agent.carrying_food.position = new_food_position
                            agent.carrying_food.is_moved = True
                        

                elif not(agent.carrying_food):
                    if new_agent_position[0] < 0 or new_agent_position[1] < 0 or new_agent_position[0] > GRID_SIZE-1 or new_agent_position[1] > GRID_SIZE-1:
                        continue

                    if self.grid[new_agent_position[0], new_agent_position[1]] is None:
                        agent.position += delta_pos[action]

            elif action == "pick_up" and agent.carrying_food is None:
                for food in self.foods:
                    if (self.compute_dist(food.position, agent.position) <= np.sqrt(2)) and len(food.carried) == 0:
                        # If the combined strength satisfies the required strength, the food is picked up sucessfully
                        if food.strength_required - food.reduced_strength <= agent.strength and not food.carried:
                            food.carried += food.pre_carried
                            food.carried.append(agent.id)
                            for agent_id in food.carried:
                                self.agents[agent_id].carrying_food = food
                                self.rewards[agent_id] += pickup_reward
                            food.pre_carried.clear()
                            # print(f"Agents {food.carried} picked up food at {food.position}")
                            break

                        # If food is too heavy, the heaviness is reduced by the strength of the picking agent.
                        # Other agents can pick up if the combined strength satisfies the required strength
                        elif food.strength_required - food.reduced_strength > agent.strength and not food.carried:
                            food.reduced_strength += agent.strength
                            food.pre_carried.append(agent.id) # agent.id prepares to carry the food.

            elif action == "drop" and agent.carrying_food:
                # If agent drops food at home
                if (agent.carrying_food.position[0] in range(HOME_POSITION[0], HOME_POSITION[0] + HOME_SIZE) and 
                    agent.carrying_food.position[1] in range(HOME_POSITION[1], HOME_POSITION[1] + HOME_SIZE)):
                    
                    # Dismiss the dropped food item at home
                    agent.carrying_food.position = (-2000,-2000)
                    agent.carrying_food.done = True
                    self.collected_foods.add(agent.carrying_food.id)
                    
                    for agent_id in agent.carrying_food.carried:
                        self.agents[agent_id].energy += self.agents[agent_id].carrying_food.energy_score # TODO this is wrong another agent has to get energy too
                        self.rewards[agent_id] += self.agents[agent_id].carrying_food.energy_score * drop_reward # TODO this is wrong another agent has to get energy too
                        
                        self.agents[agent_id].carrying_food.carried = []
                        self.agents[agent_id].carrying_food = None

                    
                else:
                    self.rewards[agent.id] += drop_punishment
                    agent.carrying_food.carried = []
                    agent.carrying_food = None

            else:
                self.rewards[agent.id] -= 1 # Useless move punishment

            # Update grid state 
            self.update_grid()

            # End if any agent runs out of energy
            if agent.energy <= 0:
                agent.done = True
                self.rewards += np.array([energy_punishment] * NUM_AGENTS)
                return self.observe(), np.copy(self.rewards), True, None, None

        # End conditions
        # End if all food items are collected
        if len(self.collected_foods) == len(self.foods):
            self.rewards += np.array([collect_all_reward] * NUM_AGENTS)
            return self.observe(), np.copy(self.rewards), True, None, None

        return self.observe(), np.copy(self.rewards), False, None, None