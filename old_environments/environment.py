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
NUM_FOODS = 1  # Number of foods
ENERGY_FACTOR = 10
NUM_ACTIONS = 6

# print("HOME GRID X,Y", self.home_grid_x, self.home_grid_y)
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
drop_reward_factor = 1 # multiplying with energy
energy_reward_factor = 1

# energy parameter
pick_up_energy_factor = 0
step_punishment = False
class Environment(ParallelEnv):
    metadata = {"name": "multi_foraging"}
    def __init__(self, truncated=False, torch_order=True, num_agents=2, n_words=10, message_length=1, use_message=False, use_pointgoal=True, seed=42):
        np.random.seed(seed)
        self.use_message = use_message
        self.use_pointgoal = use_pointgoal # observed location is relatie to the home
        self.message_length = message_length
        self.possible_agents = [i for i in range(num_agents)]
        self.grid_size = 7
        self.image_size = 5
        self.num_channels = 1
        self.n_words = n_words
        self.torch_order = torch_order
        self.truncated = {i:truncated for i in range(num_agents)}
        self.infos = {}
        self.image_shape = (self.num_channels,self.image_size,self.image_size) if self.torch_order else (self.image_size,self.image_size,self.num_channels)
        self.single_observation_space = spaces.Dict(
            {"image": spaces.Box(0, 255, shape=self.image_shape, dtype=np.float32),
            "location": spaces.Box(0, self.grid_size, shape=(2,), dtype=np.float32),
            "energy": spaces.Box(0, 500, shape=(1,), dtype=np.float32)
            })
        if self.use_message:
            self.single_observation_space["message"] = spaces.Box(0, n_words-1, shape=(message_length,), dtype=np.int64)
            self.single_action_space = spaces.Dict({"action":spaces.Discrete(NUM_ACTIONS), "message":spaces.Discrete(n_words)})
        else:
            self.single_action_space = spaces.Discrete(NUM_ACTIONS)

        self.observation_spaces = spaces.Dict({i: self.single_observation_space for i in range(num_agents)})
        self.action_spaces = spaces.Dict({i: self.single_action_space for i in range(num_agents)})
        self.render_mode = None
        self.home_size = 3
        self.reward_denom = 100 # normalize reward
        self.reset()

    def reset(self, seed=42, options=None):
        home_rand_pos = np.random.randint(self.grid_size-self.home_size, size=2)
        self.home_position = (home_rand_pos[0], home_rand_pos[1])  # Coordinates of the home
        self.home_grid_x = {self.home_position[0] + i for i in range(self.home_size)}
        self.home_grid_y = {self.home_position[1] + i for i in range(self.home_size)}

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
        self.foods = [Food(position=self.random_food_position(), 
                            food_type = 6,
                            id=food_id) for food_id in range(NUM_FOODS)
                    ]
        for food in self.foods:
            self.grid[food.position[0], food.position[1]] = food

        self.collected_foods = set()
        self.sent_message = {i:np.zeros((1,)).astype(np.int64) for i in range(self.num_agents)} # Message that each agent sends, each agent receive N-1 agents' messages
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
            if self.l2_dist(curr_pos, prev_pos) < min_distance:
                satisfy = False
                break
        return satisfy

    def random_position(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if self.grid[pos[0], pos[1]] is None and self.min_dist(pos,2):
                self.prev_pos_list.append(pos)
                return pos

    def random_food_position(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if self.grid[pos[0], pos[1]] is None and self.min_dist(pos,2): #  and self.manhattan_dist(pos, self.home_position) > 3:
                self.prev_pos_list.append(pos)
                return pos

    def l2_dist(self, pos1, pos2):
        pos1 = np.array([pos1[0], pos1[1]])
        pos2 = np.array([pos2[0], pos2[1]])
        return np.linalg.norm(pos1 - pos2)

    def manhattan_dist(self, a, b):
        return sum(abs(val1-val2) for val1, val2 in zip(a,b))

    def a_minus_b(self, a, b):
        return (a[0]-b[0], a[1]-b[1])
        
    def observe(self):
        '''
        torch_order: (C, W, H)
        '''
        if len(self.possible_agents)==1:
            image = self.agent_maps[0].observe(self)
            if self.torch_order:
                image = np.transpose(image, (2,0,1))
            return {"image": image, "location": self.a_minus_b(self.home_position, self.agent_maps[0].position), "energy": self.agent_maps[0].energy}
        else:
            agent_obs = {i:{} for i in range(self.num_agents)}
            for i, agent in enumerate(self.agent_maps):
                image = agent.observe(self)
                if self.torch_order:
                    image = np.transpose(image, (2,0,1))
                agent_obs[i]['image'] = image
                agent_obs[i]['location'] = self.a_minus_b(self.home_position, agent.position)
                agent_obs[i]['energy'] = np.array([agent.energy])
                if self.use_message: #TODO this is for two agents seeing each other message but not seeing its message
                    agent_obs[i]['message'] = self.sent_message[i]
            # print("agent_obs", agent_obs)
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
        
    def extract_message(self, message, agent_id):
        received_message = [msg for i, msg in enumerate(message) if i != agent_id]
        return np.array(received_message)

    def normalize_reward(self, reward):
        norm_reward = {}
        for key, item in reward.items():
            norm_reward[key] = item / self.reward_denom
        return norm_reward

    def failed_action(self, agent):
        agent.energy -= 2 # Useless move punishment

    def step(self, agent_action_dict, int_action=True):
        # Update food state: Clear all agents if not carried
        self.update_food()
        # One step in the simulation
        # Gather each agent's chosen action for consensus on movement
        actions = {}
        self.rewards = {i:0 for i in self.agents}
        for i, agent in enumerate(self.agent_maps):
            if self.use_message: # Tuple TODO
                agent_actions, received_message = agent_action_dict[i]["action"], agent_action_dict
            else:
                agent_actions = agent_action_dict[i]
            # End if any agent runs out of energy
            if agent.energy <= 0: #TODO Change this to the end
                agent.done = True
                for j in range(len(self.possible_agents)):
                    self.dones[j] = True
                break
            
            if self.use_message and received_message is not None:
                self.sent_message[i] = self.extract_message(received_message, i)

            if int_action:
                if len(self.possible_agents)==1:
                    action = self.int_to_act(agent_actions)
                else:
                    action = self.int_to_act(agent_actions) # integer action to string action
            else:
                agent_actions = agent_action_dict
                if len(self.possible_agents)==1:
                    action = agent_actions
                else:
                    action = agent_actions[i] # integer action to string action
            actions[i] = (agent, action)

        # Consensus action is for agents taking the same food items
        consensual_action = {}
        for food in self.foods:
            if len(food.carried) > 1:
                first_id = food.carried[0]
                try:
                    consensual_action[food.id] = actions[first_id][1] if all(a[1] == actions[first_id][1] for k, a in actions.items() if a[0].id in food.carried) else None
                except: # In case agent.done
                    pass

        for action_key in actions.keys():
            (agent, action) = actions[action_key]
            # If an agent is tied to other agents, i.e., picking the same food.
            # Consensus action has to be satisfied for all agents to perform action, move or drop food.  Otherwise, these actions have no effect.
            if agent.carrying_food:
                if agent.carrying_food.id in consensual_action:
                    if consensual_action[agent.carrying_food.id] is None:
                        # print(f"Agent {agent.id} couldn't move; consensus required.")
                        self.failed_action(agent)
                        continue
                    
            if action in ["up", "down", "left", "right"]:
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
                        # multi_agent_reward = False
                        # if len(agent.carrying_food.carried) > 1: # reward is given if 2 agents successfully move food
                        #     multi_agent_reward = True
                        for agent_id in agent.carrying_food.carried:
                            old_position = self.agent_maps[agent_id].position
                            new_position = self.agent_maps[agent_id].position + delta_pos[action]
                            self.agent_maps[agent_id].position = new_position
                            loss = 0.2*min(self.agent_maps[agent_id].strength, self.agent_maps[agent_id].carrying_food.strength_required)
                            self.agent_maps[agent_id].energy += -loss-1 # When carry food, agent lose more energy due to friction
                            # if multi_agent_reward:
                            #     self.rewards[agent_id] += 0.2
                            if step_punishment:
                                self.rewards[agent_id] += -loss-1
                            # step_reward
                            # old_home_dist = self.l2_dist(old_position, self.home_position)
                            # new_home_dist = self.l2_dist(new_position, self.home_position)
                            # self.rewards[agent_id] += 0.2*(old_home_dist-new_home_dist)

                        if not(agent.carrying_food.is_moved):
                            # print(f"{agent.carrying_food.id} moves to {new_food_position}")
                            agent.carrying_food.position = new_food_position
                            agent.carrying_food.is_moved = True
                    else:
                        self.failed_action(agent)

                elif not(agent.carrying_food):
                    if new_agent_position[0] < 0 or new_agent_position[1] < 0 or new_agent_position[0] > self.grid_size-1 or new_agent_position[1] > self.grid_size-1:
                        self.failed_action(agent)

                    elif self.grid[new_agent_position[0], new_agent_position[1]] is None:
                        agent.position += delta_pos[action]
                        agent.energy -= 1
                        if step_punishment:
                            self.rewards[agent.id] -= 1
                    else:
                        self.failed_action(agent)

            elif action == "pick_up" and agent.carrying_food is None:
                hit = False
                for food in self.foods:
                    if (self.l2_dist(food.position, agent.position) <= np.sqrt(2)) and len(food.carried) == 0:
                        # If the combined strength satisfies the required strength, the food is picked up sucessfully
                        if food.strength_required - food.reduced_strength <= agent.strength and not food.carried:
                            food.carried += food.pre_carried
                            food.carried.append(agent.id)
                            for agent_id in food.carried:
                                # step_reward
                                self.agent_maps[agent_id].carrying_food = food
                                self.agent_maps[agent_id].energy -= pick_up_energy_factor*food.energy_score
                            food.pre_carried.clear()
                            # print(f"Agents {food.carried} picked up food at {food.position}")
                            hit = True
                            break

                        # If food is too heavy, the heaviness is reduced by the strength of the picking agent.
                        # Other agents can pick up if the combined strength satisfies the required strength
                        elif food.strength_required - food.reduced_strength > agent.strength and not food.carried:
                            food.reduced_strength += agent.strength
                            food.pre_carried.append(agent.id) # agent.id prepares to carry the food.
                            hit = True
                if not(hit):
                    self.failed_action(agent)


            elif action == "drop" and agent.carrying_food:
                # If agent drops food at home
                if (agent.carrying_food.position[0] in range(self.home_position[0], self.home_position[0] + self.home_size) and 
                    agent.carrying_food.position[1] in range(self.home_position[1], self.home_position[1] + self.home_size)):
                    
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
                    self.failed_action(agent)

            # Useless action
            elif action == "pick_up" and agent.carrying_food is not None:
                self.failed_action(agent)

            # Useless action
            elif action == "drop" and not(agent.carrying_food):
                self.failed_action(agent)


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

        # elif self.dones[0] or self.dones[1]: #TODO make it general for multiple agents
        #     # terminal reward if food is not placed at home
        #     for food in self.foods:
        #         if not(food.done) and len(food.carried) > 0: 
        #             for agent_id in food.carried:
        #                 self.rewards[agent_id] += ((self.grid_size-1)*2 - self.manhattan_dist(food.position, self.home_position))
                        


        # normalize reward
        self.rewards = self.normalize_reward(self.rewards)

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
                        if x in environment.home_grid_x and y in environment.home_grid_y:
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