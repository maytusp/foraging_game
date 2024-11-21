import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import time

from nets import *
from constants import *
from keyboard_control import *
from environment import *

import wandb


# Constants

ACTION_DIM = 6  # Action space size
MESSAGE_DIM = 10  # Length of the message vector
SEQ_LENGTH = 5  # Sequence length for LSTM
BATCH_SIZE = 16
GAMMA = 0.99
LR = 1e-4
REPLAY_SIZE = 1000
# EPSILON_DECAY = 0.99999 # 0.995
MAX_EPSILON = 0.8
MIN_EPSILON = 0.2
EXPLORE_STEPS = 1e4

# Visual Observation
INPUT_CHANNELS = 4
IMAGE_SIZE = 5

# Message
VOCAB_SIZE = 32
EMBED_DIM = 64

HIDDEN_DIM = 128

# Define the LSTM-based Q-Network without message
class LSTM_QNetwork(nn.Module):
    def __init__(self, input_channels, image_size, hidden_dim, action_dim, vocab_size, message_dim):
        super(LSTM_QNetwork, self).__init__()
        self.observation_encoder = CNNEncoder(input_channels, hidden_dim)
        # For calculating input size
        obs_feat = self.observation_encoder(torch.zeros(1, input_channels, image_size, image_size))
        obs_feat_dim = obs_feat.shape[1]
        loc_feat_dim = obs_feat_dim // 2
        self.location_encoder = MLP(dims=[2, loc_feat_dim])  # Location encoding
        input_dim = obs_feat_dim + loc_feat_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = MLP([hidden_dim, hidden_dim, action_dim])
        
    # without message
    def forward(self, obs, location, hidden=None):
        # obs: (batch_size, L, 5, 5, C)
        # location: (batch_size, L, 2)
        
        if obs.shape[-1] == INPUT_CHANNELS: # If obs shape = [B, L, W, H, C] change to [B, L, C, W, H]
            obs = torch.permute(obs, (0,1,4,2,3))

        # both encoder are CNN/MLP and they do not support temporal dim
        # we have to merge temporal dim into batch dim to parallelly process input
        B, T = obs.shape[0], obs.shape[1]

        obs = obs.contiguous().view(B * T, obs.shape[2], obs.shape[3], obs.shape[4])
        location = location.contiguous().view(B * T, location.shape[2])
        
        obs_encoded = self.observation_encoder(obs)
        loc_encoded = self.location_encoder(location)

        # Bring temporal dimension back
        obs_encoded = obs_encoded.view(B, T, -1)
        loc_encoded = loc_encoded.view(B, T, -1)


        combined = torch.cat((obs_encoded, loc_encoded), dim=2)
        lstm_out, hidden = self.lstm(combined, hidden)
        lstm_out = torch.relu(self.fc(lstm_out[:, -1, :]))

        action_q = self.action_head(lstm_out)
        return action_q, None, hidden

# Replay Buffer for Sequential Data
class SequentialReplayBuffer:
    def __init__(self, size, seq_length):
        self.buffer = deque(maxlen=size)
        self.seq_length = seq_length
        self.Transition = namedtuple('Transition',
                        ('images', 'locations', 'actions', 'rewards', 'next_images', 'next_locations', 'dones'))

    def add(self, *args):
        self.buffer.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# LSTM DQN Agent
class LSTMDQLAgent:
    def __init__(self, action_dim, message_dim, hidden_dim=128):
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.epsilon = 1.0
        self.step = 0

        self.q_network = LSTM_QNetwork(INPUT_CHANNELS, IMAGE_SIZE, HIDDEN_DIM, action_dim, VOCAB_SIZE, message_dim).to(device)
        self.target_network = LSTM_QNetwork(INPUT_CHANNELS, IMAGE_SIZE, HIDDEN_DIM, action_dim, VOCAB_SIZE, message_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.replay_buffer = SequentialReplayBuffer(REPLAY_SIZE, SEQ_LENGTH)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, image_seq, loc_seq):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            message = np.random.rand(self.message_dim)  # Random message
            message = None

        else:
            image_seq_input = torch.tensor(image_seq, dtype=torch.float32).unsqueeze(0).to(device)
            loc_seq_input = torch.tensor(loc_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_q, message_q, _ = self.q_network(image_seq_input, loc_seq_input)
            action = torch.argmax(action_q).item()
            message = None
        return torch.tensor(action, dtype=torch.int32, device=device).unsqueeze(0), message

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        transitions = self.replay_buffer.sample(BATCH_SIZE)
        batch = self.replay_buffer.Transition(*zip(*transitions))
        images = torch.cat(batch.images).to(device)
        locations = torch.cat(batch.locations)
        actions = torch.cat(batch.actions).to(device)
        rewards = torch.cat(batch.rewards).to(device)
        next_images = torch.cat(batch.next_images).unsqueeze(1).to(device)
        next_locations = torch.cat(batch.next_locations).unsqueeze(1).to(device)
        dones = torch.cat(batch.dones).to(torch.int).to(device)
        
        # print("__________________________")
        # print("batch.images", images.shape)
        # print("batch.locations", locations.shape)
        # print("batch.actions", actions.shape)
        # print("batch.rewards", rewards.shape)
        # print("batch.next_images", next_images.shape)
        # print("batch.next_locations", next_locations.shape)
        # print("__________________________")

        # Compute Q-targets
        with torch.no_grad():
            prev_images = images[:, -SEQ_LENGTH+1:, :, :, :]
            prev_locations = locations[:, -SEQ_LENGTH+1:, :]
            next_images_input = torch.cat((prev_images, next_images), dim=1) # concat along time dimension
            next_locations_input = torch.cat((prev_locations, next_locations), dim=1) # concat along time dimension

            next_action_q, _, _ = self.target_network(next_images_input, next_locations_input)
            max_next_q = torch.max(next_action_q, dim=1)[0]
            q_targets = rewards + (1 - dones) * GAMMA * max_next_q

        # Compute Q-values
        action_q, _, _ = self.q_network(images, locations) # This is Q(s,)
        q_values = action_q.gather(1, actions).squeeze(1) # Compute Q(s,a)

        # Loss and Optimization
        loss = nn.MSELoss()(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        # Epsilon decay
        # self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
        self.epsilon = max(MIN_EPSILON, ((EXPLORE_STEPS - self.step)/EXPLORE_STEPS)*MAX_EPSILON)
        return loss.item()


# Environment Interaction
def train_lstm_dql(env, num_episodes):
    agent = LSTMDQLAgent(ACTION_DIM, MESSAGE_DIM)
    for episode in range(num_episodes):
        obs = env.reset()
        image_seq = [obs['image'][0]]
        loc_seq = [obs['location'][0]]
        done = False
        total_reward = 0
        step = 0
        while not done:
            action, message = agent.select_action(image_seq[-SEQ_LENGTH:], loc_seq[-SEQ_LENGTH:])
            env_action = env.int_to_act(action)
            next_obs, rewards, done, _, _ = env.step(env_action)
            if len(image_seq) >= SEQ_LENGTH:
                agent.replay_buffer.add(torch.tensor(np.array(image_seq[-SEQ_LENGTH:]), dtype=torch.float32).unsqueeze(0),
                                                    torch.tensor(np.array(loc_seq[-SEQ_LENGTH:]), dtype=torch.float32).unsqueeze(0),
                                                    torch.tensor(action, dtype=torch.int64).unsqueeze(0), 
                                                    torch.tensor(rewards[0], dtype=torch.float32).unsqueeze(0), 
                                                    torch.tensor(np.array(next_obs["image"][0]), dtype=torch.float32).unsqueeze(0), 
                                                    torch.tensor(np.array(next_obs["location"][0]), dtype=torch.float32).unsqueeze(0), 
                                                    torch.tensor(done, dtype=torch.bool).unsqueeze(0),
                                        )

            image_seq.append(next_obs["image"][0])
            loc_seq.append(next_obs["location"][0])
            loss = agent.train()

            total_reward += sum(rewards)
            step += 1

        agent.update_target_network()
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, loss: {loss}")
        

# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Environment()
train_lstm_dql(env, 500)