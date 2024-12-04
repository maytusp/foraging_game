#TODO: Add energy as observation
import pygame
import numpy as np
import random
import time
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from constants import *
from keyboard_control import *

# Environment Parameters
NUM_FOODS = 6  # Number of foods
ENERGY_FACTOR = 20
HOME_POSITION = (0, 0)  # Coordinates of the home
HOME_SIZE = 2
HOME_GRID_X = {HOME_POSITION[0] + i for i in range(HOME_SIZE)}
HOME_GRID_Y = {HOME_POSITION[1] + i for i in range(HOME_SIZE)}

NUM_ACTIONS = 6

# print("HOME GRID X,Y", HOME_GRID_X, HOME_GRID_Y)
MAX_MESSAGE_LENGTH = 10  # Example message length limit
AGENT_ATTRIBUTES = [150]  # All agents have the same attributes
HOME_ATTRIBUTES = [100]
AGENT_STRENGTH = 3
AGENT_ENERGY = 30

MAX_REQUIRED_STRENGTH = 6


# Reward Hyperparameters
energy_punishment = 0
collect_all_reward = 0
pickup_reward = 0
drop_punishment = 0
drop_reward_factor = 0.1 # multiplying with energy
energy_reward_factor = 0.01

class Environment(ParallelEnv):
    metadata = {"name": "multi_foraging"}
    def __init__(self, truncated=False, torch_order=True, num_agents=2):
        self.possible_agents = [i for i in range(num_agents)]
        self.grid_size = 10
        self.image_size = 5
        self.num_channels = 1
        self.torch_order = torch_order
        self.truncated = {i:truncated for i in range(num_agents)}
        self.infos = {}
        self.image_shape = (self.num_channels,self.image_size,self.image_size) if self.torch_order else (self.image_size,self.image_size,self.num_channels)
        self.single_observation_space = spaces.Dict(
            {"image": spaces.Box(0, 255, shape=self.image_shape, dtype=np.float32),
                "location": spaces.Box(0, self.grid_size, shape=(2,), dtype=np.float32),
                "energy": spaces.Box(0, 500, shape=(1,), dtype=np.float32),
            })
        self.single_action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_spaces = spaces.Dict({i: self.single_observation_space for i in range(num_agents)})
        self.action_spaces = spaces.Dict({i: self.single_action_space for i in range(num_agents)})
        self.render_mode = None
        

    def reset(self, seed=42, options=None):
        self.episode_lengths = {i:0 for i in range(len(self.possible_agents))}
        self.cumulative_rewards = {i:0 for i in range(len(self.possible_agents))}
        self.dones = {i:False for i in range(len(self.possible_agents))}
        self.infos = {}

        self.grid = np.full((self.grid_size, self.grid_size), None)
        self.prev_pos_list = []
        # Initialize agents with uniform attributes
        self.agents = self.possible_agents[:]
        self.agent_maps = [EnvAgent(i, self.random_position(), AGENT_STRENGTH, AGENT_ENERGY, self.grid_size) for i in range(len(self.possible_agents))]
        

        for agent in self.agent_maps:
            self.grid[agent.position[0], agent.position[1]] = agent
        #  position, food_type, id)
        self.foods = [Food(position=self.random_position(), 
                            food_type = food_id+1,
                            id=food_id) for food_id in range(NUM_FOODS)
                    ]
        for food in self.foods:
            self.grid[food.position[0], food.position[1]] = food

        self.collected_foods = set()
        self.message = np.zeros((len(self.possible_agents), MAX_MESSAGE_LENGTH)) # Message that each agent sends, each agent receive N-1 agents' messages
        return self.observe(), self.infos

    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def action_space(self, agent_id):
        return self.action_spaces[agent_id]
    
    def update_grid(self):
        '''
        Update grid position after agents move
        '''
        self.grid = np.full((self.grid_size, self.grid_size), None)
        for agent in self.agent_maps:
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
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if self.grid[pos[0], pos[1]] is None and self.min_dist(pos,2) and self.compute_dist(pos, HOME_POSITION) > 2:
                self.prev_pos_list.append(pos)
                return pos

    def compute_dist(self, pos1, pos2):
        pos1 = np.array([pos1[0], pos1[1]])
        pos2 = np.array([pos2[0], pos2[1]])
        return np.linalg.norm(pos1 - pos2)
    
    def observe(self):
        '''
        torch_order: (C, W, H)
        '''
        if len(self.possible_agents)==1:
            image = self.agent_maps[0].observe(self)
            if self.torch_order:
                image = np.transpose(image, (2,0,1))
            return {"image": image, "location": self.agent_maps[0].position, "energy": self.agent_maps[0].energy}
        else:
            agent_obs = {i:{} for i in range(self.num_agents)}
            for i, agent in enumerate(self.agent_maps):
                image = agent.observe(self)
                if self.torch_order:
                    image = np.transpose(image, (2,0,1))
                agent_obs[i]['image'] = image
                agent_obs[i]['location'] = agent.position
                agent_obs[i]['energy'] = np.array([agent.energy])
            return agent_obs

    def int_to_act(self, action):
        '''
        input: action integer tensor frm the moel, the value is from 0 to 5
        output: action string that matches environment
        '''
        action_map = {0: "up", 
                    1: "down", 
                    2: "left",
                    3: "right", 
                    4: "pick_up",
                    5: "drop", }

        return action_map[action]



    def step(self, agent_actions, int_action=True):
        # Update food state: Clear all agents if not carried
        self.update_food()
    
        # One step in the simulation
        # Gather each agent's chosen action for consensus on movement
        actions = {}
        self.rewards = {i:0 for i in self.agents}
        for i, agent in enumerate(self.agent_maps):
            # End if any agent runs out of energy
            if agent.energy <= 0:
                agent.done = True
                for j in range(len(self.possible_agents)):
                    self.dones[j] = True
                break

            if int_action:
                if len(self.possible_agents)==1:
                    action = self.int_to_act(agent_actions)
                else:
                    action = self.int_to_act(agent_actions[i]) # integer action to string action
            else:
                
                if len(self.possible_agents)==1:
                    action = agent_actions
                else:
                    action = agent_actions[i] # integer action to string action
            actions[i] = (agent, action)

        # Consensus action is for agents taking the same food items
        consensus_action = {}
        for food in self.foods:
            if len(food.carried) > 1:
                first_id = food.carried[0]
                try:
                    consensus_action[food.id] = actions[first_id][1] if all(a[1] == actions[first_id][1] for k, a in actions.items() if a[0].id in food.carried) else None
                except: # In case agent.done
                    pass

        for action_key in actions.keys():
            (agent, action) = actions[action_key]
            failed_action = False
            # If an agent is tied to other agents, i.e., picking the same food.
            # Consensus action has to be satisfied for all agents to perform action, move or drop food.  Otherwise, these actions have no effect.
            if agent.carrying_food:
                if agent.carrying_food.id in consensus_action:
                    if consensus_action[agent.carrying_food.id] is None:
                        print(f"Agent {agent.id} couldn't move; consensus required.")
                        failed_action = True
                        agent.energy -= 1
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
                        new_pos_list.append(self.agent_maps[agent_id].position + delta_pos[action])

                    for id,new_pos in enumerate(new_pos_list):
                        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] > self.grid_size-1 or new_pos[1] > self.grid_size-1:
                            # print(f"Bounded, {agent.id}'s move = False, because new_pos number {id}")
                            move = False                            
                            break

                        check_grid = self.grid[new_pos[0], new_pos[1]]
                        if isinstance(check_grid, EnvAgent) and (check_grid.id not in agent.carrying_food.carried) or \
                            isinstance(check_grid, Food) and (check_grid.id != agent.carrying_food.id):
                            move = False
                            # print(f"{agent.id}'s move = False because new_pos number {id}")
                            # print(f"first condition= {isinstance(check_grid, Agent) and (check_grid.id not in agent.carrying_food.carried)}")
                            # print(f"second condition= {isinstance(check_grid, Food) and (check_grid.id != agent.carrying_food.id)}")
                            break


                    if move:
                        for agent_id in agent.carrying_food.carried:
                            old_position = self.agent_maps[agent_id].position
                            new_position = self.agent_maps[agent_id].position + delta_pos[action]
                            self.agent_maps[agent_id].position = new_position
                            loss = 0.2*min(self.agent_maps[agent_id].strength, self.agent_maps[agent_id].carrying_food.strength_required)
                            self.agent_maps[agent_id].energy -= loss # When carry food, agent lose more energy due to friction
                            # step_reward
                            # old_home_dist = self.compute_dist(old_position, HOME_POSITION)
                            # new_home_dist = self.compute_dist(new_position, HOME_POSITION)
                            # self.rewards[agent_id] += 0.2*(old_home_dist-new_home_dist)

                        if not(agent.carrying_food.is_moved):
                            # print(f"{agent.carrying_food.id} moves to {new_food_position}")
                            agent.carrying_food.position = new_food_position
                            agent.carrying_food.is_moved = True
                    else:
                        failed_action = True

                elif not(agent.carrying_food):
                    if new_agent_position[0] < 0 or new_agent_position[1] < 0 or new_agent_position[0] > self.grid_size-1 or new_agent_position[1] > self.grid_size-1:
                        failed_action = True
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
                                # step_reward
                                self.agent_maps[agent_id].carrying_food = food
                                self.rewards[agent_id] += pickup_reward
                            food.pre_carried.clear()
                            # print(f"Agents {food.carried} picked up food at {food.position}")
                            break

                        # If food is too heavy, the heaviness is reduced by the strength of the picking agent.
                        # Other agents can pick up if the combined strength satisfies the required strength
                        elif food.strength_required - food.reduced_strength > agent.strength and not food.carried:
                            food.reduced_strength += agent.strength
                            food.pre_carried.append(agent.id) # agent.id prepares to carry the food.
                if agent.carrying_food == None:
                    failed_action = True


            elif action == "drop" and agent.carrying_food:
                # If agent drops food at home
                if (agent.carrying_food.position[0] in range(HOME_POSITION[0], HOME_POSITION[0] + HOME_SIZE) and 
                    agent.carrying_food.position[1] in range(HOME_POSITION[1], HOME_POSITION[1] + HOME_SIZE)):
                    
                    # Dismiss the dropped food item at home
                    agent.carrying_food.position = (-2000,-2000)
                    agent.carrying_food.done = True
                    self.collected_foods.add(agent.carrying_food.id)
                    
                    for agent_id in agent.carrying_food.carried:
                        # step_reward
                        self.agent_maps[agent_id].energy += self.agent_maps[agent_id].carrying_food.energy_score # TODO this is wrong another agent has to get energy too
                        self.rewards[agent_id] += self.agent_maps[agent_id].carrying_food.energy_score * drop_reward_factor # TODO this is wrong another agent has to get energy too
                        
                        self.agent_maps[agent_id].carrying_food.carried = []
                        self.agent_maps[agent_id].carrying_food = None

                    
                else:
                    agent.carrying_food.carried = []
                    agent.carrying_food = None
                    failed_action = True

            # Useless action
            elif action == "pick_up" and agent.carrying_food is not None:
                failed_action = True

            # Useless action
            elif action == "drop" and not(agent.carrying_food):
                failed_action = True
            
            if failed_action:
                agent.energy -= 1 # Useless move punishment

            # Update grid state 
            self.update_grid()

        # End conditions
        # End if all food items are collected
        if len(self.collected_foods) == len(self.foods):
            average_energy = 0
            for agent in self.agent_maps:
                average_energy += agent.energy
            average_energy /= len(self.possible_agents)
            # terminal_reward
            for agent in self.agent_maps:
                self.rewards[agent.id] += energy_reward_factor * average_energy

                self.dones = {i:True for i in range(len(self.possible_agents))}
                
        for agent in self.agent_maps:
            self.cumulative_rewards[agent.id] += self.rewards[agent.id]
            self.episode_lengths[agent.id] += 1

            if self.dones[agent.id]:
                self.infos[agent.id] = {"episode": {
                                "r": self.cumulative_rewards[agent.id],
                                "l": self.episode_lengths[agent.id],
                                },
                            }
                                         
                                
        return self.observe(), self.rewards, self.dones, self.truncated, self.infos

# Define the classes
class EnvAgent:
    def __init__(self, id, position, strength, max_energy, grid_size):
        self.id = id
        self.position = position
        self.strength = strength
        self.energy = max_energy
        self.carrying_food = None
        self.done = False
        self.messages = []
        self.grid_size = grid_size
        # Agent observation field adjusted to (24, 4) for the 5x5 grid view, exluding agent position
    
    def observe(self, environment): #TODO Check this again
        # Define the 5x5 field of view around the agent, excluding its center
        perception_data = []
        for dx in range(-2, 3):
            row = []
            for dy in range(-2, 3):
                if dx == 0 and dy == 0: # agent's own position
                    if self.carrying_food is not None:
                        obs_attribute = list(map(lambda x:x+33, AGENT_ATTRIBUTES)) # if agent is carrying food
                    else:
                        obs_attribute = AGENT_ATTRIBUTES
                    row.append(obs_attribute)
                    continue

                x, y = self.position[0] + dx, self.position[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    obj = environment.grid[x, y]
                    if obj is None:
                        if x in HOME_GRID_X and y in HOME_GRID_Y:
                            row.append(HOME_ATTRIBUTES)  # home grid
                        else:
                            row.append([0])  # Empty grid
                    elif isinstance(obj, Food): # Observe Food
                        if len(obj.carried) > 0:
                            obs_attribute = list(map(lambda x:x+33, obj.attribute)) # if food is carried
                        else:
                            obs_attribute = obj.attribute

                        row.append(obs_attribute)
                    elif isinstance(obj, EnvAgent): # Observe another agent
                        if obj.carrying_food is not None:
                            obs_attribute = list(map(lambda x:x+33, AGENT_ATTRIBUTES)) # if agent is carrying food
                        else:
                            obs_attribute = AGENT_ATTRIBUTES
                        row.append(obs_attribute)
                else:
                    row.append([255])  # Out-of-bounds grid (treated as empty)
            perception_data.append(row)
        
        return np.array(perception_data)

    def send_message(self, env):
        return env.message[0,:] #TODO Use neural network to send message

class Food:
    def __init__(self, position, food_type, id):
        self.type_to_strength_map = {
                                    1:3, # Spinach
                                    2:3,  # Watermelon
                                    3:3, # Strawberry
                                    4:6,  # Chicken
                                    5:6,  # Pig
                                    6:6 # Cattle
                                    }
        self.position = position
        self.food_type = food_type
        self.strength_required = self.type_to_strength_map[food_type]
        self.carried = [] # keep all agents that already picked up this food
        self.pre_carried = [] # keep all agents that try to pick up this food, no need to be successful
        self.attribute = self.generate_attributes(food_type)
        self.energy_score = ENERGY_FACTOR * self.strength_required
        self.id = id
        self.done = False
        self.reduced_strength = 0

    def generate_attributes(self, food_type):
        attribute_mapping = {
            1: [10], # Spinach
            2: [20], # Watermelon
            3: [30], # Strawberry
            4: [40], # Chicken
            5: [50], # Pig
            6: [60], # Cattle

        }
        return np.array(attribute_mapping.get(food_type, [1, 1, 1, 1]))