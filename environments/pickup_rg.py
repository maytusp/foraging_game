# 9 April 2025
# Referential Game version of Foraging Game
# 2 players observe their items in obs grid but they cannot move
# 6 time steps in total. They have to send message at each time step to each other.
# At the t=6, both players can select pick up (1) or idle (0). 
# The positive reward is given if the target item is on the player that presses pick up. Otherwise, negative reward is given.

import pygame
import numpy as np
import random
import time
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import pickle

from constants import *
from keyboard_control import *


# Environment Parameters
NUM_ACTIONS = 3

AGENT_STRENGTH = 3
AGENT_ENERGY = 20

class Environment(ParallelEnv):
    metadata = {"name": "goal_cond_pickup"}
    def __init__(self, truncated=False, torch_order=True, num_agents=2, n_words=10, message_length=1, use_message=False, 
                                                                                                        seed=42, 
                                                                                                        agent_visible=False,
                                                                                                        food_ener_fully_visible=False, 
                                                                                                        identical_item_obs=False,
                                                                                                        N_i = 2,
                                                                                                        grid_size=5,
                                                                                                        image_size=3,
                                                                                                        max_steps=5,
                                                                                                        mode="train",
                                                                                                        ):
        np.random.seed(seed)
        self.mode = mode
        self.use_message = use_message
        self.agent_visible = agent_visible
        self.message_length = message_length
        self.possible_agents = [i for i in range(num_agents)]
        self.grid_size = grid_size # environment size
        self.image_size = image_size # receptive field size
        self.N_val = 255 # number of possible values, 255 is like standard RGB image
        self.N_att = 1 # number of attributes
        self.N_i = N_i # number of food items
        self.num_channels = 1 + self.N_att
        self.identical_item_obs = identical_item_obs
        self.n_words = n_words
        self.torch_order = torch_order
        self.truncated = {i:truncated for i in range(num_agents)}
        self.infos = {}
        self.image_shape = (self.num_channels, self.image_size, self.image_size) if self.torch_order else (self.image_size,self.image_size,self.num_channels)
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
        if mode == "train":
            self.score_unit = 25
            self.start_steps = 0
            self.last_steps = 10
            self.score_list = [(i+1)*self.score_unit for i in range(self.start_steps, self.last_steps)] # each food item will have one of these energy scores, assigned randomly.
        elif mode == "test":
            self.score_unit = 20
            self.start_steps = 0
            self.last_steps = 12
            self.score_list = [(i+1)*self.score_unit for i in range(self.start_steps, self.last_steps) if (i+1) % 25 != 0]

        self.max_score = self.N_val
        self.food_ener_fully_visible = food_ener_fully_visible

        self.max_steps = max_steps
        self.food_type2name =  {
                                    1: "spinach",
                                    2: "watermelon",
                                    3: "strawberry",
                                    4: "chicken",
                                    5: "pig",
                                    6: "cattle",
                                }
        self.agent_spawn_range = {0:((0, 0), (1, self.grid_size-1)), 1:((self.grid_size-2, 0), (self.grid_size-1, self.grid_size-1))}
        self.food_spawn_range = {0:((0, 0), (0, self.grid_size-1)), 1:((self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1))} # TODO add condition for 4 and 6 items
        self.reset()
        
    
    def reset(self, seed=42, options=None):
        self.curr_steps = 0
        self.cue_step = False
        self.episode_lengths = {i:0 for i in range(len(self.possible_agents))}
        self.cumulative_rewards = {i:0 for i in range(len(self.possible_agents))}
        self.dones = {i:False for i in range(len(self.possible_agents))}
        self.infos = {}

        self.grid = np.full((self.grid_size, self.grid_size), None)
        self.prev_pos_list = []
        self.reg_food_spawn_range = {}
        self.reg_agent_spawn_range = {}
        #  position, food_type, id)

        self.selected_score = np.random.choice(self.score_list, size=self.N_i, replace=False)
        self.target_food_id = np.argmax(self.selected_score)
        self.score_visible_to_agent = np.random.choice([0]* (self.N_i//2) + [1]*(self.N_i//2), size=self.N_i, replace=False)

        self.foods = [Food(position=self.random_food_position(food_id), 
                            food_type = food_id+1,
                            id=food_id,
                            energy_score=self.selected_score[food_id],
                            visible_to_agent=self.score_visible_to_agent[food_id],
                            identical_item_obs=self.identical_item_obs) for food_id in range(self.N_i) # id runs from 0 to N_i-1
                    ]
        for food in self.foods:
            self.grid[food.position[0], food.position[1]] = food

        self.agents = self.possible_agents[:]
        self.agent_maps = [EnvAgent(i, self.random_agent_position(agent_id=i), 
                            AGENT_STRENGTH, AGENT_ENERGY, 
                            self.grid_size, self.agent_visible,
                            self.food_ener_fully_visible) for i in range(len(self.possible_agents))]
        
        
        for agent in self.agent_maps:
            self.grid[agent.position[0], agent.position[1]] = agent

        self.collected_foods = []
        self.sent_message = {i:np.zeros((1,)).astype(np.int64) for i in range(self.num_agents)} # Message that each agent sends, each agent receive N-1 agents' messages

        return self.observe(), self.infos

    def generate_food_attribute(self):
        distance_set = set()
        distance_list = []
        generated_food_attributes = []
        for i in range(self.N_i):
            stop = False
            while not(stop):
                curr_attribute_idx = np.random.choice(self.attribute_combinations_inds)
                curr_attribute = self.attribute_combinations[curr_attribute_idx]
                curr_dist = self.l2_dist(curr_attribute, self.goal_attribute)
                if curr_dist not in distance_set and curr_dist != 0:
                    distance_set.add(curr_dist)
                    distance_list.append(curr_dist)
                    generated_food_attributes.append(curr_attribute)
                    stop = True
        return generated_food_attributes, np.argmin(distance_list)
    
    def generate_goal_attribute(self):
        rand_idx = np.random.choice(self.attribute_combinations_inds)
        goal_attribute = self.attribute_combinations[rand_idx]
        # print(f"{rand_idx} {goal_attribute}")
        return goal_attribute

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
            if not(agent.done):
                self.grid[agent.position[0], agent.position[1]] = agent
        for food in self.foods:
            if not(food.done):
                self.grid[food.position[0], food.position[1]] = food
                
    def min_dist(self,curr_pos, min_distance):
        satisfy = True
        for prev_pos in self.prev_pos_list:
            if self.manhattan_dist(curr_pos, prev_pos) < min_distance:
                satisfy = False
                break
        return satisfy


    def random_position(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if self.grid[pos[0], pos[1]] is None and self.min_dist(pos,3):
                self.prev_pos_list.append(pos)
                return pos

    def random_agent_position(self, agent_id):
        # Select spawn range / side (left or right) # Inefficient, back to this later
        selected_side = self.reg_agent_spawn_range[agent_id]
        seen_food_id =  np.where(self.score_visible_to_agent == agent_id)[0][0]
        food_pos =  self.foods[seen_food_id].position
        min_xy, max_xy = self.agent_spawn_range[selected_side]
        min_x, min_y = min_xy[0], min_xy[1]
        max_x, max_y = max_xy[0], max_xy[1]
        while True:
            pos = (random.randint(min_x, max_x), random.randint(min_y, max_y))
            if self.grid[pos[0], pos[1]] is None and self.manhattan_dist(pos, food_pos) < 2:
                return pos

    def random_food_position(self, food_id):
        # Select spawn range / side (left or right) # Inefficient, back to this later
        if food_id > 0:
            prev_selected_side = self.reg_food_spawn_range[0]
            selected_side = {0:1, 1:0}[prev_selected_side]
        else:
            selected_side = np.random.binomial(1, 0.5, 1)[0]
            
        min_xy, max_xy = self.food_spawn_range[selected_side]
        min_x, min_y = min_xy[0], min_xy[1]
        max_x, max_y = max_xy[0], max_xy[1]
        self.reg_food_spawn_range[food_id] = selected_side

        # Register agent's side based on the foods it sees
        agent_id = self.score_visible_to_agent[food_id]
        if agent_id not in self.reg_agent_spawn_range:
            self.reg_agent_spawn_range[agent_id] = selected_side
            
        while True:
            pos = (random.randint(min_x, max_x), random.randint(min_y, max_y))
            if self.grid[pos[0], pos[1]] is None and self.min_dist(pos,3):
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
            return {"image": image, "location": self.agent_maps[0].position, "energy": self.agent_maps[0].energy}
        else:
            agent_obs = {i:{} for i in range(self.num_agents)}
            for i, agent in enumerate(self.agent_maps):
                if self.cue_step:
                    image = np.ones_like(agent.observe(self)) * 100
                else:
                    image = agent.observe(self)
                    # Sanity check: agents see their items' scores
                    # if np.max(image[:,:,1]) == 0: # image (W,H,C)
                    #     print("ERROR: Agents do not see the score")

                if self.torch_order:
                    image = np.transpose(image, (2,0,1))
                agent_obs[i]['image'] = image
                agent_obs[i]['location'] = agent.position
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
        action_map = {
                    0 : "idle",
                    1 : "not_pick_up",
                    2:  "pick_up",
                    }
        return action_map[action]
        
    def extract_message(self, message, agent_id):
        received_message = [v[1]['message'] for k, v in enumerate(message.items()) if k != agent_id]
        received_message = np.array(received_message)
        return received_message



    def failed_action(self, agent):
        pass

    def step(self, agent_action_dict, int_action=True):
        success = 0
        episode_successs = True
        self.curr_steps+=1
        
        if self.curr_steps == 1:
            self.agent_obs = self.observe()
        elif self.curr_steps == self.max_steps-1: # Give visual cue to make decision
            self.cue_step = True
            self.agent_obs = self.observe()

        actions = {}
        self.rewards = {i:0 for i in self.agents}
        
        for i, agent in enumerate(self.agent_maps):
            if self.use_message:
                agent_actions, received_message = agent_action_dict[i]["action"], agent_action_dict
            else:
                agent_actions = agent_action_dict[i]
            
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
            

        # End conditions
        # One food is collected
        # Reward if food has highest energy among others
        # Punishment otherwise
        if self.curr_steps == self.max_steps:
            # print(f"final step {self.curr_steps} obs {self.agent_obs}")
            self.pickup_agent_id = self.score_visible_to_agent[self.target_food_id]
            self.idle_agent_id = {0:1, 1:0}[self.pickup_agent_id]
            self.dones = {i:True for i in range(len(self.possible_agents))}
            # terminal_reward
            for action_key in actions.keys():
                (agent, action) = actions[action_key]

                # failed action
                # if action == "idle":
                #     episode_successs = False
                if agent.id == self.pickup_agent_id and action != "pick_up":
                    episode_successs = False
                elif agent.id == self.idle_agent_id and action != "not_pick_up":
                    episode_successs = False

        else:
            for action_key in actions.keys():
                (agent, action) = actions[action_key]

                if action != "idle":
                    episode_successs = False
                    self.dones = {i:True for i in range(len(self.possible_agents))}
                    break

                    
        if self.dones[0]:
            for agent in self.agent_maps:
                if episode_successs:
                    self.rewards[agent.id] += 1
                else:
                    self.rewards[agent.id] -= 1

        for agent in self.agent_maps:
            self.cumulative_rewards[agent.id] += self.rewards[agent.id]
            self.episode_lengths[agent.id] += 1

            if self.dones[agent.id]:
                self.infos[agent.id] = {"episode": {
                                "r": self.cumulative_rewards[agent.id],
                                "l": self.episode_lengths[agent.id],
                                "collect": len(self.collected_foods),
                                "success": episode_successs,
                                "target_id": self.target_food_id,
                                "food_scores": {f.id: f.energy_score for f in self.foods},
                                "score_visible_to_agent" : self.score_visible_to_agent,
                                },
                            }
    
        return self.agent_obs, self.rewards, self.dones, self.truncated, self.infos
# Original Code: Non-vectorise
# # Define the classes
class EnvAgent:
    def __init__(self, id, position, strength, max_energy, grid_size, agent_visible, fully_visible):
        self.id = id
        self.position = position
        self.strength = strength
        self.energy = max_energy
        self.carrying_food = None
        self.done = False
        self.grid_size = grid_size
        self.agent_visible = agent_visible
        self.fully_visible = fully_visible
        # Agent observation field adjusted to (24, 4) for the 5x5 grid view, exluding agent position
    
    def observe(self, environment): #TODO Check this again
        # Define the 5x5 field of view around the agent, excluding its center
        occupancy_data = []
        food_attribute_data = np.zeros((environment.image_size, environment.image_size, environment.N_att))
        ob_range = environment.image_size // 2
        begin = -ob_range
        end = ob_range + 1
        agent_occupancy = [environment.N_val // 2]
        wall_occupancy = [environment.N_val]
        food_occupancy = [environment.N_val // 3]
        carry_add = environment.N_val // 10
        
        for dx in range(begin, end):
            row = []
            for dy in range(begin, end):
                if dx == 0 and dy == 0: # agent's own position
                    if self.carrying_food is not None:
                        obs_occupancy = list(map(lambda x:x+carry_add, agent_occupancy)) # if agent is carrying food
                    else:
                        obs_occupancy = agent_occupancy
                    row.append(obs_occupancy)
                    continue

                x, y = self.position[0] + dx, self.position[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    obj = environment.grid[x, y]
                    if obj is None:
                        row.append([0])  # Empty grid
                    elif isinstance(obj, Food): # Observe Food
                        if len(obj.carried) > 0:
                            obs_occupancy = list(map(lambda x:x+carry_add, food_occupancy)) # if food is carried
                        else:
                            obs_occupancy = food_occupancy
                        row.append(obs_occupancy)

                        # observe food's attribute
                        if self.fully_visible: # If agent can see all attributes
                            food_attribute_data[dx+ob_range, dy+ob_range] = obj.attribute
                        else: # This is default case where agent can see some attributes
                            mask = (obj.visible_to_agent == self.id)  # Creates a boolean mask
                            food_attribute_data[dx+ob_range, dy+ob_range] = mask * obj.attribute

                    elif isinstance(obj, EnvAgent) and self.agent_visible: # Observe another agent
                        if obj.carrying_food is not None:
                            obs_occupancy = list(map(lambda x:x+carry_add, agent_occupancy)) # if agent is carrying food
                        else:
                            obs_occupancy = agent_occupancy
                        row.append(obs_occupancy)
                    else:
                        row.append([0])  # Empty grid
                else:
                    row.append(wall_occupancy)  # Out-of-bounds grid (treated as empty)
            occupancy_data.append(row)
        occupancy_data = np.array(occupancy_data)
        obs_out = np.concatenate((occupancy_data, food_attribute_data), axis=2)
        return obs_out

class Food:
    def __init__(self, position, food_type, id, energy_score, visible_to_agent, identical_item_obs):
        self.type_to_strength_map = {
                                    1:6, # Spinach
                                    2:6,  # Watermelon
                                    3:6, # Strawberry
                                    4:6,  # Chicken
                                    5:6,  # Pig
                                    6:6 # Cattle
                                    }
        self.identical_item_obs = identical_item_obs
        self.position = position
        self.food_type = food_type
        self.strength_required = self.type_to_strength_map[food_type]
        self.carried = [] # keep all agents that already picked up this food
        self.pre_carried = [] # keep all agents that try to pick up this food, no need to be successful
        self.energy_score = energy_score
        self.id = id
        self.done = False
        self.reduced_strength = 0
        self.visible_to_agent = visible_to_agent
        self.attribute  = energy_score # TODO add another channel referring to item category
        

if __name__ == "__main__":
    env = Environment()
    for i in range(100):
        env.reset()
        print(f"target id: {env.target_food_id}")