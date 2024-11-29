"""Port of the Chipmunk robot demo. Showcase a topdown robot driving towards the
mouse, and hitting obstacles on the way.
"""
from pygame.color import THECOLORS
import random
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import os
import torch
# Render Parameter
fps = 30
render_speed = 4

# Environment Parameters
human_play = False


pi = math.pi
width = 640
height = 480
norm_dist = (640**2 + 480**2)**0.5
collision_types = {
    "robot": 1,
    "sensor": 2,
    "obstacle": 3,
    "target": 4,
}
action_map = {0: "turn left", 1: "turn right", 2: "go forward"}# , 3: "stop"}

class NavigationSearchEnv(gym.Env):
    def __init__(self, device, difficulty=1, render_mode=True):
        # Diffulty (Level)
        difficulty_to_n_obstacles = {1:1, 2:10, 3:20}
        self.num_obstacles  = difficulty_to_n_obstacles[difficulty]

        # Hardware
        self.render_mode = render_mode
        self.device = device
        self.num_envs = 1
        
        # Reward
        self.box_size = 5
        self.num_visit_boxes = (width // self.box_size) * (height // self.box_size)
        self.turn = False

        # Agent
        self.num_skip = 4 # Frame skip
        self.num_steps = 1000
    
        # Robot
        self.robot_body = None
        self.robot_control_body = None
        self.sensor_body = None
        self.obs = None
        # self.reach_all_target = False
        self.hit_obstacle = False
        self.robot_pos_x = None
        self.robot_pos_y = None
        
        
        # Depth Sensor
        self.num_sensors = 9
        self.sensor_resolution = 8 # each sensor reading is 3 bit
        self.sensor_range = 80
        self.sensor_spread = int(self.sensor_range // self.sensor_resolution) # Default spread.
        self.sensor_init_distance = 10  # Gap before first sensor.
        self.obstacle_color = (169, 169, 169, 255)

        # Proprioception Sensor
        self.proprio_size = 3

        self.observation_space = spaces.Dict(
            {
                "depth": spaces.Box(0, 1, shape=(self.num_sensors,), dtype=np.float32),
                "proprioception": spaces.Box(0, 1, shape=(self.proprio_size,), dtype=np.float32),
            }
        )

        # We have 3 actions, corresponding to "right", "up", "left"
        self.action_space = spaces.Discrete(3)        
        
        self.initialise()
    
    def step(self, action):
        for _ in range(self.num_skip):
            self.update(1 / fps, action)
        self.turn = False
        self.prev_visits = len(self.visit_boxes)
        visit_x, visit_y = self.robot_body.position
        visit_x = int(visit_x // self.box_size)
        visit_y = int(visit_y // self.box_size)
        self.visit_boxes.add((visit_x, visit_y))
        self.current_visits = len(self.visit_boxes)
        self.done = self.is_episode_end()
        self.obs = self._get_obs()
        self.hit_obstacle = self.check_hit_obstacle()
        reward = self.get_reward()
        self.count_step += 1

        truncated, info = False, None
        return self.obs, reward, self.done, truncated, info
    
    def reset(self):
        self.initialise()
        obs = self._get_obs()
        info = None
        return obs, info
    
    def close(self):
        if self.render_mode:
            pygame.display.quit()
            pygame.quit()

    def update(self, dt, action):
            
        if human_play:
            for event in pygame.event.get():
                if (
                    event.type == pygame.QUIT
                    or event.type == pygame.KEYDOWN
                    and (event.key in [pygame.K_ESCAPE, pygame.K_q])
                ):
                    exit()            
            if self.render_mode:
                self.render()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                dv = Vec2d(30.0, 0.0)
                self.robot_body.velocity = self.robot_body.rotation_vector.cpvrotate(dv)
                self.robot_body.position = self.robot_body.position
            elif keys[pygame.K_RIGHT]:
                self.robot_body.angular_velocity = 1
            elif keys[pygame.K_LEFT]:
                self.robot_body.angular_velocity = -1               
            else:
                self.robot_body.angular_velocity = 0
                self.robot_body.velocity = 0, 0
            
        else:
            if self.render_mode:
                self.render()
            action_label = action_map[action]

            if action_label == "go forward":
                dv = Vec2d(30.0, 0.0)
                self.robot_body.velocity = self.robot_body.rotation_vector.cpvrotate(dv)
                self.robot_body.position = self.robot_body.position
                self.robot_body.angular_velocity = 0
            elif action_label == "turn right":
                self.robot_body.velocity = 0, 0
                self.robot_body.angular_velocity = 1.2
                self.turn = True
            elif action_label == "turn left":
                self.robot_body.velocity = 0, 0
                self.robot_body.angular_velocity = -1.2
                self.turn = True
            elif action_label == "stop":
                self.robot_body.velocity = 0, 0
                self.robot_body.angular_velocity = 0
                
        self.space.step(dt)


    def _get_obs(self):
        # Get the current location and the readings there.
        # Update sensor measurement
        x, y = self.robot_body.position
        visual_state = self.get_sensor_readings(x, y, self.robot_body.angle)
        proprio_state = self.get_absolute_proprioception()
        # norm_readings = [(self.sensor_resolution-i) / self.sensor_resolution for i in visual_state]
        norm_readings = [(i) / self.sensor_resolution for i in visual_state]
        return {'depth': np.array(norm_readings), 'proprioception': proprio_state}
        
    def get_distance(self, pos1, pos2):
        dist = abs(pos1 - pos2)
        return dist / norm_dist
    
    # Reward function
    def get_reward(self):
        reward = 0
        new_visit = self.current_visits - self.prev_visits
        if self.done:
            if self.hit_obstacle:
                reward -= 10
        else:
            if new_visit >= 1:
                reward += 1
            else:
                reward -= 1
        return reward

    # The episode will be ended if the agent hit an obstacle, 
    # the agent finds a target, or the agent reaches maximum steps
    def is_episode_end(self):
        if self.count_step == self.num_steps - 1:
            return True
        if self.hit_obstacle:
            return True
        # if len(self.found_target) == self.num_targets: # Reach all target
        #     return True
        return False
    
    def check_hit_obstacle(self):
        x, y = self.robot_body.position
        offset_hit = self.offset + 10
        if x < offset_hit or y < offset_hit or x > width - offset_hit or y > height - offset_hit:
            return True
        else:
            # The complexity can be reduced by looping over only obstacles nearby
            for obstacle in self.obstacles:
                obstacle_shapes = [s for s in obstacle.shapes]
                if self.robot_shape.shapes_collide(obstacle_shapes[0]).points != []:
                    return True
            
        return False

    # Proprioception
    def get_absolute_proprioception(self):
        x, y = self.robot_body.position
        angle = math.fmod(self.robot_body.angle + (2*pi), 2*pi)
        x_norm, y_norm = x / width, y / height
        angle_norm = angle / (2 * pi)
        return np.array([x_norm, y_norm, angle_norm])
    
    def add_box(self, space, size, mass):
        radius = Vec2d(size[0], size[1]).length

        body = pymunk.Body()
        space.add(body)

        # body.position = Vec2d(
        #     random.random() * (width - 2 * radius) + radius,
        #     random.random() * (height - 2 * radius) + radius,
        # )
        body_pos = (random.randint(3*size[0], int(width - 3*size[0])), random.randint(3*size[1], int(height-3*size[1])))
        body.position = body_pos[0], body_pos[1]
        body.angle = 0
        
        shape = pymunk.Poly.create_box(body, (size[0], size[1]), 0.0)
        shape.mass = mass
        shape.friction = 0.7
        shape.collision_type = collision_types["obstacle"]
        space.add(shape)
        if not(self.load_level):
            self.obstacle_position.append([body.position[0], body.position[1]])
        return body

    def init_robot_pos(self):
        # Set up robot position
        self.robot_init_pos = (random.randint(30, width-30), random.randint(30, height-30))
        self.min_robot_to_obstacle = self.get_min_distance_robot_to_obstacle(self.robot_init_pos)

    def initialise(self):
        self.done = False
        # Set up Level
        # self.target_init_red = 210
        # self.num_targets = 0
        self.visit_boxes = set()
        # self.target_map = {} # Key is color tuple, item is target
        self.object_position_list = []
        # self.found_target = set() # a set of found target
        # self.target_colors = set([(self.target_init_red + i, 0, 0, 255) for i in range(self.num_targets)])

        self.load_level = False
        self.level_path = "data/level1.npy"
        self.obstacle_position = np.load(self.level_path) if self.load_level else []

        # Dummy screen is for checking each point's color
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height+50))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)        
            self.font = pygame.font.Font(None, 24)
            self.text = "Observation:"
            self.text = self.font.render(self.text, True, pygame.Color("white"))
        else:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        self.dummy_screen = pygame.display.set_mode((width, height+50))
        # Initialise vairables
        self.count_step = 0
        
        # add obstacle
        self.obstacles = []
        self.space = pymunk.Space()
        self.space.iterations = 10
        self.space.sleep_time_threshold = 0.5
        self.offset = 10

        self.static_body = self.space.static_body

        # Create segments around the edge of the screen.
        self.shape = pymunk.Segment(self.static_body, (self.offset, self.offset), (self.offset, height-self.offset), 4)
        self.shape.color = self.obstacle_color
        self.space.add(self.shape)
        self.shape.elasticity = 1
        self.shape.friction = 1

        self.shape = pymunk.Segment(self.static_body, (width-self.offset, self.offset), (width-self.offset, height-self.offset), 4)
        self.shape.color = self.obstacle_color
        self.space.add(self.shape)
        self.shape.elasticity = 1
        self.shape.friction = 1

        self.shape = pymunk.Segment(self.static_body, (self.offset, self.offset), (width-self.offset, self.offset), 4)
        self.shape.color = self.obstacle_color
        self.space.add(self.shape)
        self.shape.elasticity = 1
        self.shape.friction = 1

        self.shape = pymunk.Segment(self.static_body, (self.offset, height-self.offset), (width-self.offset, height-self.offset), 4)
        self.shape.color = self.obstacle_color
        self.space.add(self.shape)
        self.shape.elasticity = 1
        self.shape.friction = 1


        for i in range(self.num_obstacles):
            body = self.add_box(self.space, (20, 20), 1e6)
            for s in body.shapes: 
                s.color = self.obstacle_color
            if self.load_level:
                body.position = self.obstacle_position[i][0], self.obstacle_position[i][1]
            self.obstacles.append(body)
            self.object_position_list.append(body.position)

        # Save level
        if not(self.load_level):
            # print(f"Save a new level with positions {obstacle_position}")
            np.save(self.level_path, np.array(self.obstacle_position))

        self.robot_body = self.add_box(self.space, (10,10), 1)
        self.init_robot_pos()
        
        # This part assures that the robot will not appear at the same position as any target
        while self.min_robot_to_obstacle < 0.03:
            self.init_robot_pos()

        self.robot_body.position = self.robot_init_pos[0], self.robot_init_pos[1]
        for s in self.robot_body.shapes:
            self.robot_shape = s
        self.robot_shape.color = (0, 255, 255, 255)
        self.robot_shape.collision_type = collision_types['robot']

        # Add a target
        # for target_idx in range(self.num_targets):
        #     self.target = self.add_box(self.space, (10,10), 10)
        #     self.target.position = Vec2d(
        #         50 + random.random() * (width - 50),
        #         50 + random.random() * (height - 50),
        #     )      
        #     # Pushing targets away from each other
        #     if len(self.object_position_list) >= 1:
        #         min_distance = self.compute_min_distance(self.target)
        #         while min_distance < 0.03:
        #             self.target.position = Vec2d(
        #                 50 + random.random() * (width - 50),
        #                 50 + random.random() * (height - 50),
        #             )      
        #             min_distance = self.compute_min_distance(self.target)

        #     target_color = (self.target_init_red+target_idx, 0, 0, 255)
        #     for s in self.target.shapes:
        #         s.color = target_color
        #         s.sensor = True
        #     self.target_map[target_color] = self.target
        #     self.object_position_list.append(self.target.position)
    
    # For generating targets that are appropriately far from each other
    # def compute_min_distance(self, current_target):
    #     min_distance = 1000
    #     for other_target_pos in self.object_position_list:
    #         target_pos = current_target.position
    #         temp_distance = self.get_distance(target_pos, other_target_pos)
    #         min_distance = min(min_distance, temp_distance)
    #     return min_distance

    def get_min_distance_robot_to_obstacle(self, robot_pos):
        min_distance = 1000
        for obs_pos in self.object_position_list:
            temp_distance = self.get_distance(robot_pos, obs_pos)
            min_distance = min(min_distance, temp_distance)
        return min_distance    


    def get_track_or_not(self, reading):
        reading_tuple = (reading[0], reading[1], reading[2], reading[3])
        # print(THECOLORS['black'])
        if reading == self.obstacle_color:
            return 1
        else:
            return 0        
        # elif reading_tuple in self.target_colors and not(reading_tuple in self.found_target):
        #     temp = self.target_map[reading_tuple]
        #     self.found_target.add(reading_tuple)
        #     for s in temp.shapes:
        #         s.color = (0,210,0,255)
        #     # print("Target found:", len(self.found_target))
        #     return 0

        
    def get_sensor_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sensor readings
        simply return N "distance" readings, one for each sensor
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sensor "arm" is non-zero, then that arm returns a distance of 5.
        """
        num_sensors = self.num_sensors # Odd number
        # fov = pi / 4 # Field of view in radian
        fov = pi / 1.5 # Field of view in radian
        delta_ang = fov / (num_sensors-1)
        offset_list = [(i*delta_ang) - (fov / 2) for i in range(num_sensors)]
        # Make our arms.
        
        arm_dict = {}
        for i in range(num_sensors):
            arm_dict[str(i)] = self.make_sensor_arm(x, y)
            offset = offset_list[i]

            # Rotate them and get readings.
            readings.append(self.get_arm_distance(arm_dict[str(i)], x, y, angle, offset))

        if self.render_mode:
            pygame.display.update()

        return readings
    
    def make_sensor_arm(self, x, y):
        spread = self.sensor_spread  # Default spread.
        distance = self.sensor_init_distance  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, self.sensor_resolution+1):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points
    
    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = self.dummy_screen.get_at(rotated_p)
                if self.get_track_or_not(obs) == 1:
                    return i

            if self.render_mode:
                pygame.draw.circle(self.screen, (255, 255, 255), (rotated_p), 1)
        # Return the distance for the arm.
        return i
    
    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = y_change + y_1
        return int(new_x), int(new_y)
    
    def show_perception(self, obs):
        # Set the starting position for the squares
        x = 260
        y = 490
        max_intensity = 255
        num_sensors = len(obs)
        # Loop through and draw 9 squares
        for i in range(num_sensors):
            intensity = int(obs[i] * max_intensity)
            square_color = (intensity, intensity, intensity)
            # Draw a square at the current position
            pygame.draw.rect(self.screen, square_color, pygame.Rect(x, y, 20, 20))
            pygame.display.update()
            # Move the position for the next square
            x += 21

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        obs = self._get_obs()
        self.show_perception(obs['depth'])
        self.screen.fill(pygame.Color("black"))
        self.space.debug_draw(self.draw_options)
        self.screen.blit(self.text, (140, 493))
        pygame.display.flip()
        self.clock.tick(fps * render_speed)

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO_RNN as PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, CategoricalMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from navsearch import NavigationSearchEnv

import torch
import torch.nn as nn


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 num_envs=1, num_layers=3, hidden_size=256, sequence_length=128, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]
        
        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), {"rnn": [rnn_states[0], rnn_states[1]]}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=128):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), {"rnn": [rnn_states[0], rnn_states[1]]}


# load and wrap the gymnasium environment
# env = gym.vector.make("PendulumNoVel-v1", num_envs=4, asynchronous=False)
mode = "train"
difficulty = 3
save_dir = "logs/level3_ppo_lstm"
render_mode = True if mode == "eval" else False
# render_mode = True
total_agent_steps = int(2e5)
memory_size = int(2048)
sequence_length = 256
device =  'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = NavigationSearchEnv(device=device, difficulty=difficulty, render_mode=render_mode)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, sequence_length=sequence_length)
models["value"] = Value(env.observation_space, env.action_space, device, num_envs=env.num_envs, sequence_length=sequence_length)

# print(models["policy"])

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = memory_size  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 4
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 10000
cfg["experiment"]["directory"] = save_dir

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": total_agent_steps, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()