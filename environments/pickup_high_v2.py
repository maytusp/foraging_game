# 28 Feb 2025 Pickup high v2 for studying communication through behaviour:
# The only difference to v1 is items are randomly placed across the map
# Agents are spawn on the different sides with their seen-score items
# This is for study experiment1: what is encoded in communication?
# Visible vs Invisible Agents
# Different Receptive Fields
# Different Environment Size
# Different Number of Items
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
NUM_ACTIONS = 5

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
                                                                                                        image_size=5,
                                                                                                        max_steps=10,
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
        self.reward_scale = 1 # normalize reward
        if mode == "train":
            self.score_unit = 5
            self.start_steps = 0
            self.last_steps = 50
            self.score_list = [(i+1)*self.score_unit for i in range(self.start_steps, self.last_steps)] # each food item will have one of these energy scores, assigned randomly.
        elif mode == "test":
            self.score_unit = 2
            self.start_steps = 0
            self.last_steps = 125
            self.score_list = [(i+1)*self.score_unit for i in range(self.start_steps, self.last_steps) if (i+1) % 5 != 0]

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

        self.foods = [Food(position=self.random_position(), 
                            food_type = food_id+1,
                            id=food_id,
                            energy_score=self.selected_score[food_id],
                            visible_to_agent=self.score_visible_to_agent[food_id],
                            identical_item_obs=self.identical_item_obs) for food_id in range(self.N_i) # id runs from 0 to N_i-1
                    ]
        for food in self.foods:
            self.grid[food.position[0], food.position[1]] = food

        self.agents = self.possible_agents[:]
        self.agent_maps = [EnvAgent(i, self.random_position(), 
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
                image = agent.observe(self)
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
        action_map = {0: "up", 
                    1: "down", 
                    2: "left",
                    3: "right", 
                    4: "pick_up",
                    }
        return action_map[action]
        
    def extract_message(self, message, agent_id):
        received_message = [v[1]['message'] for k, v in enumerate(message.items()) if k != agent_id]
        received_message = np.array(received_message)
        return received_message


    def normalize_reward(self, reward):
        norm_reward = {}
        for key, item in reward.items():
            norm_reward[key] = item / self.reward_scale
        return norm_reward

    def failed_action(self, agent):
        pass

    def step(self, agent_action_dict, int_action=True):
        success = 0
        self.curr_steps+=1
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


        for action_key in actions.keys():
            (agent, action) = actions[action_key]
            # If an agent is tied to other agents, i.e., picking the same food.
            # Consensus action has to be satisfied for all agents to perform action, move or drop food.  Otherwise, these actions have no effect
                    
            if action in ["up", "down", "left", "right"]:
                delta_pos = {'up': np.array([-1,0]),
                            'down': np.array([1,0]),
                            'left': np.array([0,-1]),
                            'right': np.array([0,1]),}
                old_agent_position = np.array(agent.position)
                new_agent_position = old_agent_position + delta_pos[action]

                if new_agent_position[0] < 0 or new_agent_position[1] < 0 or new_agent_position[0] > self.grid_size-1 or new_agent_position[1] > self.grid_size-1:
                    self.failed_action(agent)

                elif self.grid[new_agent_position[0], new_agent_position[1]] is None:
                    agent.position += delta_pos[action]
                    # agent.energy -= 1
                    # self.rewards[agent.id] -= 0.1
                else:
                    self.failed_action(agent)

            elif action == "pick_up":
                hit = False
                for food in self.foods:
                    if (self.l2_dist(food.position, agent.position) <= np.sqrt(2)):
                        # If the combined strength satisfies the required strength, the food is picked up sucessfully
                        if food.strength_required - food.reduced_strength <= agent.strength and not food.carried:
                            # cancel the step punishment for agents that pick up previously if the item is successfully picked
                            # for agent_id in food.pre_carried:
                                # self.agent_maps[agent_id].energy += 1
                                # self.rewards[agent_id] += 0.1
                            food.carried += food.pre_carried
                            food.carried.append(agent.id)
                            food.pre_carried.clear()
                            # Dismiss the dropped food item at home
                            food.position = (-2000,-2000)
                            food.done = True
                            self.collected_foods.append(food.id)
                            hit = True
                            break

                        # If food is too heavy, the heaviness is reduced by the strength of the picking agent.
                        # Other agents can pick up if the combined strength satisfies the required strength
                        elif food.strength_required - food.reduced_strength > agent.strength and not food.carried:
                            food.reduced_strength += agent.strength
                            food.pre_carried.append(agent.id) # agent.id prepares to carry the food.
                            hit = True
                            # step punishment if another agent doesn't pick up
                            # agent.energy -= 1
                            # self.rewards[agent.id] -= 0.1

                if not(hit):
                    self.failed_action(agent)

            # Update grid state 
            self.update_grid()

            # If max steps
            if self.curr_steps == self.max_steps: #TODO Change this to the end
                agent.done = True
                for j in range(len(self.possible_agents)):
                    self.dones[j] = True
                    self.rewards[j] -= 1
                break
        # End conditions
        # One food is collected
        # Reward if food has highest energy among others
        # Punishment otherwise
        if len(self.collected_foods) == 1:
            # terminal_reward
            for agent in self.agent_maps:
                if self.collected_foods[0] == self.target_food_id:
                    self.rewards[agent.id] += 1
                    self.rewards[agent.id] += ((self.max_steps - self.curr_steps) / self.max_steps) # get more reward if use fewer steps
                    success = 1
                else:
                    self.rewards[agent.id] -= 1
                self.dones = {i:True for i in range(len(self.possible_agents))}

        # normalize reward
        self.norm_rewards = self.normalize_reward(self.rewards)

        for agent in self.agent_maps:
            self.cumulative_rewards[agent.id] += self.rewards[agent.id]
            self.episode_lengths[agent.id] += 1

            if self.dones[agent.id]:
                self.infos[agent.id] = {"episode": {
                                "r": self.cumulative_rewards[agent.id],
                                "l": self.episode_lengths[agent.id],
                                "collect": len(self.collected_foods),
                                "success": success,
                                "target_id": self.target_food_id,
                                "food_scores": {f.id: f.energy_score for f in self.foods},
                                "score_visible_to_agent" : self.score_visible_to_agent,
                                },
                            }
    
        return self.observe(), self.norm_rewards, self.dones, self.truncated, self.infos
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


# # Optimised code: Vectorise, does not speed up with vectorise gym
# class EnvAgent:
#     def __init__(self, id, position, strength, max_energy, grid_size, agent_visible, fully_visible):
#         self.id = id
#         self.position = np.array(position)  # Convert to NumPy array for fast calculations
#         self.strength = strength
#         self.energy = max_energy
#         self.carrying_food = None
#         self.done = False
#         self.grid_size = grid_size
#         self.agent_visible = agent_visible
#         self.fully_visible = fully_visible

#     def observe(self, environment):
#         ob_range = environment.image_size // 2
#         begin, end = -ob_range, ob_range + 1
        
#         # Define occupancy values
#         agent_occupancy = environment.N_val // 2
#         wall_occupancy = environment.N_val
#         food_occupancy = environment.N_val // 3
#         carry_add = environment.N_val // 10

#         # Create empty observation arrays
#         occupancy_data = np.zeros((environment.image_size, environment.image_size, 1), dtype=int)
#         food_attribute_data = np.zeros((environment.image_size, environment.image_size, environment.N_att))

#         # Compute valid indices in one step
#         x_range = self.position[0] + np.arange(begin, end)
#         y_range = self.position[1] + np.arange(begin, end)

#         valid_x = (x_range >= 0) & (x_range < self.grid_size)
#         valid_y = (y_range >= 0) & (y_range < self.grid_size)

#         valid_x_range = x_range[valid_x]
#         valid_y_range = y_range[valid_y]

#         sub_grid = environment.grid[np.ix_(valid_x_range, valid_y_range)]

#         is_food = np.vectorize(lambda obj: isinstance(obj, Food) if obj else False)(sub_grid)
#         is_agent = np.vectorize(lambda obj: isinstance(obj, EnvAgent) if obj else False)(sub_grid) & self.agent_visible

#         # Get carried food status
#         food_carried = np.vectorize(lambda obj: len(obj.carried) > 0 if isinstance(obj, Food) else False)(sub_grid)
#         agent_carrying = np.vectorize(lambda obj: obj.carrying_food is not None if isinstance(obj, EnvAgent) else False)(sub_grid)

#         rhs = (
#             is_food * (food_occupancy + food_carried * carry_add) +
#             is_agent * (agent_occupancy + agent_carrying * carry_add)
#         )

#         rhs = np.expand_dims(rhs, axis=2)

#         # Assign occupancy values
#         occupancy_data[np.ix_(valid_x, valid_y)] += rhs

#         # Assign food attributes
#         if self.fully_visible:
#             food_attributes = np.vectorize(lambda obj: obj.attribute if isinstance(obj, Food) else np.zeros(environment.N_att))(sub_grid)
#         else:
#             food_attributes = np.vectorize(lambda obj: obj.attribute * (obj.visible_to_agent == self.id) if isinstance(obj, Food) else np.zeros(environment.N_att))(sub_grid)
        
#         food_attribute_data[np.ix_(valid_x, valid_y)] = np.expand_dims(food_attributes, axis=2)

#         # Mark agent's own position
#         occupancy_data[ob_range, ob_range, 0] = agent_occupancy + (carry_add if self.carrying_food else 0)

#         # Combine occupancy and attribute data
#         obs_out = np.concatenate((occupancy_data, food_attribute_data), axis=2)
#         return obs_out


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