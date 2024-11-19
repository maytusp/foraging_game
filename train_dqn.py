import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import time

from constants import *
from keyboard_control import *
from environment import *



# Constants
STATE_DIM = 30  # Example state dimension (location + visual + messages)
ACTION_DIM = 6  # Action space size
MESSAGE_DIM = 10  # Length of the message vector
SEQ_LENGTH = 8  # Sequence length for LSTM
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
REPLAY_SIZE = 10000
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        num_layers = len(dims) - 1
        for l in range(num_layers):
            self.model.add_module(f"fc{l}", nn.Linear(dims[l], dims[l+1]))
            self.model.add_module(f"relu{l}", nn.ReLU())

    def forward(self, x):
        return self.model(x)

# Define the LSTM-based Q-Network
class LSTM_QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, message_dim):
        super(LSTM_QNetwork, self).__init__()
        self.message_encoder = MLP(dims=[message_dim, message_dim*4, message_dim*2, message_dim]) #TODO Use embedding and trasnformer to extract
        self.location_encoder = MLP(dims=[2, message_dim]) # make location important as message (same size)
        self.observation_encoder = MLP(dims=[input_dim, input_dim*2, input_dim*4, input_dim*2, input_dim])

        self.message_head = MLP([hidden_dim, hidden_dim, message_dim]) #TODO Use transformer decoder
        self.action_head = MLP([hidden_dim, hidden_dim, action_dim])
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, hidden=None):
        # LSTM expects inputs of shape [batch_size, seq_len, input_dim]
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = torch.relu(self.fc(lstm_out[:, -1, :]))  # Use last hidden state
        action_q = self.action_head(lstm_out)
        message_q = self.message_head(lstm_out)
        return action_q, message_q, hidden


# Replay Buffer for Sequential Data
class SequentialReplayBuffer:
    def __init__(self, size, seq_length):
        self.buffer = deque(maxlen=size)
        self.seq_length = seq_length
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

    def add(self, *args):
        self.buffer.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# LSTM DQN Agent
class LSTMDQLAgent:
    def __init__(self, state_dim, action_dim, message_dim, hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.epsilon = 1.0

        self.q_network = LSTM_QNetwork(state_dim, hidden_dim, action_dim, message_dim).to(device)
        self.target_network = LSTM_QNetwork(state_dim, hidden_dim, action_dim, message_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.replay_buffer = SequentialReplayBuffer(REPLAY_SIZE, SEQ_LENGTH)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state_seq):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            message = np.random.rand(self.message_dim)  # Random message
        else:
            state_seq_input = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_q, message_q, _ = self.q_network(state_seq_input)
            action = torch.argmax(action_q).item()
            message = message_q.cpu().numpy()
        return action, message

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        transitions = self.replay_buffer.sample(BATCH_SIZE)
        batch = self.replay_buffer.Transition(*zip(*transitions))
        print(batch.state)
        states = torch.tensor(batch.state, dtype=torch.float32).to(device)
        actions = torch.tensor(batch.action, dtype=torch.long).to(device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(device)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32).to(device)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(device)

        # Compute Q-targets
        with torch.no_grad():
            next_action_q, _, _ = self.target_network(next_states)
            max_next_q = torch.max(next_action_q, dim=1)[0]
            q_targets = rewards[:, -1] + (1 - dones[:, -1]) * GAMMA * max_next_q

        # Compute Q-values
        action_q, _, _ = self.q_network(states)
        q_values = action_q.gather(1, actions[:, -1].unsqueeze(1)).squeeze(1)

        # Loss and Optimization
        loss = nn.MSELoss()(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)


# Environment Interaction
def train_lstm_dql(env, num_episodes):
    agent = LSTMDQLAgent(STATE_DIM, ACTION_DIM, MESSAGE_DIM)
    for episode in range(num_episodes):
        obs = env.reset()
        state_seq = [obs]
        done = False
        total_reward = 0
        while not done:
            actions = []
            messages = []

            for i in range(NUM_AGENTS):
                action, message = agent.select_action(state_seq[-SEQ_LENGTH:])
                actions.append(action)
                messages.append(message)
            next_state, rewards, done, _, _ = env.step(actions)
            
            
            for i in range(NUM_AGENTS):
                agent.replay_buffer.add(torch.tensor(np.squeeze(np.array(state_seq[-SEQ_LENGTH:]))), actions[i], rewards[i], next_state[i], done)

            state_seq.append(next_state)
            agent.train()

            total_reward += sum(rewards)

        agent.update_target_network()
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Environment()
train_lstm_dql(env, 500)
