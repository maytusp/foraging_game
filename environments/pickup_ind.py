# 24 Sep 2025: No Cooperation. Each agent picks up an item separately.
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
    metadata = {"name": "independent_pickup_v1"}

    def __init__(
        self,
        truncated=False,
        torch_order=True,
        num_agents=2,
        n_words=10,
        message_length=1,
        use_message=False,
        seed=42,
        agent_visible=False,
        food_ener_fully_visible=False,
        identical_item_obs=False,
        N_i=None,                 # ignored: number of items is forced to 2 * num_agents
        grid_size=5,
        image_size=5,
        max_steps=10,
        mode="train",             # kept for API compatibility; no effect now
        use_unseen_loc=False,     # kept for compatibility
        time_pressure=False,      # disabled in independent setting
        ablate_message=False,
        test_moderate_score=False,# kept for compatibility
        num_walls=0,
    ):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Core flags (some retained for API compatibility)
        self.mode = mode
        self.use_message = use_message
        self.agent_visible = agent_visible
        self.message_length = message_length
        self.food_ener_fully_visible = food_ener_fully_visible
        self.identical_item_obs = identical_item_obs
        self.n_words = n_words
        self.num_walls = num_walls
        self.use_unseen_loc = use_unseen_loc
        self.torch_order = torch_order
        self.ablate_message = ablate_message
        self.time_pressure = False  # no time pressure in the independent setting
        self.test_moderate_score = False

        # Dimensions / spaces
        self.possible_agents = [i for i in range(num_agents)]
        self.n_agents = num_agents
        self.grid_size = grid_size
        self.image_size = image_size
        self.N_val = 255
        self.N_att = 1                     # keep 1 channel for attributes (zeros); preserves shape
        self.num_channels = 1 + self.N_att # occupancy + attribute
        self.reward_scale = 1

        # Items: force to 2 * num_agents
        self.N_i = 2 * self.n_agents

        # Spaces
        self.image_shape = (
            (self.num_channels, self.image_size, self.image_size)
            if self.torch_order
            else (self.image_size, self.image_size, self.num_channels)
        )
        self.single_observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=self.image_shape, dtype=np.float32),
                "location": spaces.Box(0, self.grid_size, shape=(2,), dtype=np.float32),
                "energy": spaces.Box(0, 500, shape=(1,), dtype=np.float32),
            }
        )
        if self.use_message:
            self.single_observation_space["message"] = spaces.Box(
                0, n_words - 1, shape=(message_length,), dtype=np.int64
            )
            self.single_action_space = spaces.Dict(
                {"action": spaces.Discrete(NUM_ACTIONS), "message": spaces.Discrete(n_words)}
            )
        else:
            self.single_action_space = spaces.Discrete(NUM_ACTIONS)

        self.observation_spaces = spaces.Dict({i: self.single_observation_space for i in range(num_agents)})
        self.action_spaces = spaces.Dict({i: self.single_action_space for i in range(num_agents)})

        # Positions
        self.all_position_list = [[x, y] for x in range(grid_size) for y in range(grid_size)]

        # Walls spawn range (full grid by default)
        self.wall_spawn_range = ((0, 0), (self.grid_size - 1, self.grid_size - 1))

        # Episode / bookkeeping
        self.render_mode = None
        self.truncated = {i: truncated for i in range(num_agents)}
        self.infos = {}
        self.max_steps = max_steps

        self.reset()

    # --------- Helpers ---------
    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def l2_dist(self, pos1, pos2):
        pos1 = np.array([pos1[0], pos1[1]])
        pos2 = np.array([pos2[0], pos2[1]])
        return np.linalg.norm(pos1 - pos2)

    def manhattan_dist(self, a, b):
        return sum(abs(v1 - v2) for v1, v2 in zip(a, b))

    def int_to_act(self, action):
        action_map = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
            4: "pick_up",
        }
        return action_map[action]

    def normalize_reward(self, reward):
        return {k: v / self.reward_scale for k, v in reward.items()}

    def update_grid(self):
        self.grid = np.full((self.grid_size, self.grid_size), None)
        # agents first so they occupy their cells
        for agent in self.agent_maps:
            if not agent.done:
                self.grid[agent.position[0], agent.position[1]] = agent
        # foods that remain
        for food in self.foods:
            if not food.done:
                self.grid[food.position[0], food.position[1]] = food
        # walls (if any)
        for wall in self.wall_list:
            self.grid[wall.position[0], wall.position[1]] = wall

    def random_wall_position(self):
        (min_x, min_y), (max_x, max_y) = self.wall_spawn_range
        while True:
            pos = (random.randint(min_x, max_x), random.randint(min_y, max_y))
            if self.grid[pos[0], pos[1]] is None:
                return pos

    # --------- Core API ---------
    def reset(self, seed=None, options=None):

        # Sample food positions
        self.food_positions = random.sample(self.all_position_list, self.N_i)

        # Sample agent positions on remaining cells
        self.agent_position_list = [pos for pos in self.all_position_list if pos not in self.food_positions]
        self.agent_positions = random.sample(self.agent_position_list, self.n_agents)

        # State trackers
        self.curr_steps = 0
        self.episode_lengths = {i: 0 for i in self.possible_agents}
        self.cumulative_rewards = {i: 0 for i in self.possible_agents}
        self.dones = {i: False for i in self.possible_agents}
        self.infos = {}
        self.personal_collections = {i: 0 for i in self.possible_agents}

        # Grid
        self.grid = np.full((self.grid_size, self.grid_size), None)

        # Foods: identical, no scores, no cooperation fields used by logic
        self.foods = [Food(position=tuple(self.food_positions[i]), id=i) for i in range(self.N_i)]
        for food in self.foods:
            self.grid[food.position[0], food.position[1]] = food

        # Agents
        self.agents = self.possible_agents[:]
        self.agent_maps = [
            EnvAgent(i, tuple(self.agent_positions[i]), AGENT_STRENGTH, AGENT_ENERGY,
                     self.grid_size, self.agent_visible, self.food_ener_fully_visible)
            for i in range(self.n_agents)
        ]
        for agent in self.agent_maps:
            self.grid[agent.position[0], agent.position[1]] = agent

        # Walls (optional)
        self.wall_list = []
        for wall_id in range(self.num_walls):
            wall_pos = self.random_wall_position()
            self.wall_list.append(Wall(wall_id, wall_pos))
            self.grid[wall_pos[0], wall_pos[1]] = self.wall_list[wall_id]

        # Messaging placeholder (kept for API)
        if self.use_message:
            self.sent_message = {i: np.zeros((1,)).astype(np.int64) for i in range(self.n_agents)}

        return self.observe(), self.infos

    def observe(self):
        if len(self.possible_agents) == 1:
            image = self.agent_maps[0].observe(self)
            if self.torch_order:
                image = np.transpose(image, (2, 0, 1))
            return {
                "image": image,
                "location": self.agent_maps[0].position,
                "energy": self.agent_maps[0].energy,
            }
        else:
            agent_obs = {i: {} for i in range(self.n_agents)}
            for i, agent in enumerate(self.agent_maps):
                image = agent.observe(self)
                if self.torch_order:
                    image = np.transpose(image, (2, 0, 1))
                agent_obs[i]["image"] = image
                agent_obs[i]["location"] = agent.position
                agent_obs[i]["energy"] = np.array([agent.energy])
                if self.use_message:
                    if self.ablate_message:
                        agent_obs[i]["message"] = np.array([0])
                    else:
                        agent_obs[i]["message"] = self.sent_message[i]
            return agent_obs

    def step(self, agent_action_dict, int_action=True):
        self.curr_steps += 1
        self.rewards = {i: 0.0 for i in self.agents}

        # Prepare actions
        actions = {}
        for i, agent in enumerate(self.agent_maps):
            if self.use_message:
                agent_actions, received_message = agent_action_dict[i]["action"], agent_action_dict
                # In independent setting, we don’t use messages, but keep compatibility:
                self.sent_message[i] = self.extract_message(received_message, i)
            else:
                agent_actions = agent_action_dict[i]
            action = self.int_to_act(agent_actions) if int_action else agent_actions
            actions[i] = (agent, action)

        # Execute actions
        for agent, action in actions.values():
            if action in ["up", "down", "left", "right"]:
                delta_pos = {
                    "up": np.array([-1, 0]),
                    "down": np.array([1, 0]),
                    "left": np.array([0, -1]),
                    "right": np.array([0, 1]),
                }
                old = np.array(agent.position)
                newp = old + delta_pos[action]
                # bounds and collision with walls/agents/foods
                if 0 <= newp[0] < self.grid_size and 0 <= newp[1] < self.grid_size:
                    # Only move if destination is empty (no agent, no wall, no food)
                    if self.grid[newp[0], newp[1]] is None:
                        agent.position = (int(newp[0]), int(newp[1]))

            elif action == "pick_up":
                # If any food is within sqrt(2) (8-neighborhood including diagonals), pick one
                for food in self.foods:
                    if not food.done and self.l2_dist(food.position, agent.position) <= np.sqrt(2):
                        food.done = True
                        self.rewards[agent.id] += 1.0            # individual reward only
                        self.personal_collections[agent.id] += 1
                        # Remove the food from grid immediately
                        food.position = (-2000, -2000)
                        break

        # Update grid occupancy after all actions
        self.update_grid()

        # Termination: all items collected or max steps reached
        all_collected = all(f.done for f in self.foods)
        if self.curr_steps >= self.max_steps or all_collected:
            self.dones = {i: True for i in self.possible_agents}
        else:
            self.dones = {i: False for i in self.possible_agents}

        # (Optional) normalize rewards (kept for API parity)
        self.norm_rewards = self.normalize_reward(self.rewards)

        # Book-keeping
        for agent in self.agent_maps:
            self.cumulative_rewards[agent.id] += self.rewards[agent.id]
            self.episode_lengths[agent.id] += 1
            if self.dones[agent.id]:
                self.infos[agent.id] = {
                    "episode": {
                        "r": self.cumulative_rewards[agent.id],
                        "l": self.episode_lengths[agent.id],
                        "personal_collected": self.personal_collections[agent.id],
                        "total_remaining": sum(1 for f in self.foods if not f.done),
                    }
                }

        return self.observe(), self.norm_rewards, self.dones, self.truncated, self.infos

    # ---- Message extraction kept for API compatibility ----
    def extract_message(self, message, agent_id):
        received_message = [v[1]["message"] for k, v in enumerate(message.items()) if k != agent_id]
        return np.array(received_message)

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
                    elif isinstance(obj, Wall):
                        row.append(wall_occupancy)
                    else:
                        row.append([0])  # Empty grid
                else:
                    row.append(wall_occupancy)  # Out-of-bounds grid (treated as empty)
            occupancy_data.append(row)
        occupancy_data = np.array(occupancy_data)
        obs_out = np.concatenate((occupancy_data, food_attribute_data), axis=2)
        return obs_out


class Wall:
     def __init__(self, id, position):
        self.id = id
        self.position = position


class Food:
    """
    Independent collectible item with no score.
    Kept minimal, but exposes fields used by the visual observation code.
    """
    def __init__(self, position, id):
        self.position = position
        self.id = id
        self.done = False
        # Fields referenced by EnvAgent.observe (kept to avoid changing that logic)
        self.carried = []                 # never used in independent mode
        self.attribute = 0                # no score/attribute — stays 0 in the attribute channel
        self.visible_to_agent = -1        # never matches any agent id (so attribute stays hidden if masking is used)
