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
from buffer import *

import wandb
import os




# Save parameters
MODEL_SAVE_EVERY = 20000 # 500 steps
MODEL_SAVED_DIR = "checkpoints"
os.makedirs(MODEL_SAVED_DIR, exist_ok=True)

# Constants
MAX_STEPS = 30
ACTION_DIM = 6  # Action space size
MESSAGE_DIM = 10  # Length of the message vector
SEQ_LENGTH = MAX_STEPS // 3 # Sequence length for LSTM
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
REPLAY_SIZE = 1000 # 1000 episodes
# EPSILON_DECAY = 0.99999 # 0.995
MAX_EPSILON = 0.8
MIN_EPSILON = 0.2
EXPLORE_STEPS = 3e5
UPDATE_TARGET_EVERY = 10

# Visual Observation
INPUT_CHANNELS = 4
IMAGE_SIZE = 5

# Message
VOCAB_SIZE = 32
EMBED_DIM = 64

HIDDEN_DIM = 128
NUM_LSTM_LAYER = 1
#EVAL
VISUALIZE = True
VIDEO_SAVED_DIR = "vids/drqn_config1/"
os.makedirs(VIDEO_SAVED_DIR, exist_ok=True)


if VISUALIZE:
    from visualize import *
    from moviepy.editor import *

# Define the LSTM-based Q-Network without message
class LSTM_QNetwork(nn.Module):
    def __init__(self, input_channels, image_size, hidden_dim, action_dim, vocab_size, message_dim):
        super(LSTM_QNetwork, self).__init__()
        self.observation_encoder = CNNEncoder(input_channels, hidden_dim)
        # For calculating input size
        obs_feat = self.observation_encoder(torch.zeros(1, input_channels, image_size, image_size))
        obs_feat_dim = obs_feat.shape[1]
        loc_feat_dim = obs_feat_dim // 2
        self.location_encoder = MLP(dims=[2, 64, loc_feat_dim])  # Location encoding
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

        lstm_out = lstm_out.contiguous().view(B*T, lstm_out.shape[2])
        lstm_out = torch.relu(self.fc(lstm_out))

        action_q = self.action_head(lstm_out)
        action_q = action_q.view(B, T, action_q.shape[1])

        return action_q, None, hidden




# LSTM DQN Agent
class DQNAgent:
    def __init__(self, action_dim, message_dim, hidden_dim=128):
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.epsilon = MAX_EPSILON
        self.grad_step = 0

        self.q_network = LSTM_QNetwork(INPUT_CHANNELS, IMAGE_SIZE, HIDDEN_DIM, action_dim, VOCAB_SIZE, message_dim).to(device)
        self.target_network = LSTM_QNetwork(INPUT_CHANNELS, IMAGE_SIZE, HIDDEN_DIM, action_dim, VOCAB_SIZE, message_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.replay_buffer = EpisodeReplayBuffer(REPLAY_SIZE, MAX_STEPS, SEQ_LENGTH)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, image_seq, loc_seq, hidden, explore=True):
        # Forward to update hidden state
        image_seq_input = torch.tensor(image_seq, dtype=torch.float32).unsqueeze(0).to(device)
        loc_seq_input = torch.tensor(loc_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_q, message_q, hidden = self.q_network(image_seq_input, loc_seq_input, hidden)
        action = torch.argmax(action_q).item()
        message = None

        # If the action is random, an agent ignores the output of RNN
        if random.random() < self.epsilon and explore:
            action = random.randint(0, self.action_dim - 1)

        return torch.tensor(action, dtype=torch.int32, device=device).unsqueeze(0), message, hidden

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        transitions, episode_length = self.replay_buffer.sample(BATCH_SIZE)
        images = []
        locations = []
        actions = []
        rewards = []
        next_images = []
        next_locations = []
        dones = []

        for i in range(BATCH_SIZE):
            transition_image = transitions[i]["image"]
            images.append(transitions[i]["image"])
            locations.append(transitions[i]["loc"])
            actions.append(transitions[i]["acts"])
            rewards.append(transitions[i]["rews"])
            next_images.append(transitions[i]["next_image"])
            next_locations.append(transitions[i]["next_loc"])
            dones.append(transitions[i]["done"])

        images = torch.tensor(np.array(images), dtype=torch.float32).to(device)
        locations =  torch.tensor(np.array(locations), dtype=torch.float32).to(device)
        actions =  torch.tensor(np.array(actions), dtype=torch.int64).to(device)
        rewards =  torch.tensor(np.array(rewards)).to(device)
        next_images =  torch.tensor(np.array(next_images), dtype=torch.float32).to(device)
        next_locations =  torch.tensor(np.array(next_locations), dtype=torch.float32).to(device)
        dones =  torch.tensor(np.array(dones)).to(torch.int).to(device)
        
        B = images.shape[0]
        T = images.shape[1]
        # print("__________________________")
        # print("batch.images", images.shape)
        # print("batch.locations", locations.shape)
        # print("batch.actions", actions.shape)
        # print("batch.rewards", rewards.shape)
        # print("batch.next_images", next_images.shape)
        # print("batch.next_locations", next_locations.shape)
        # print("dones", dones.shape)
        # print("__________________________")

        # Compute Q-targets
        with torch.no_grad():
            next_action_q, _, _ = self.target_network(next_images, next_locations)
            # print(f"next_images {next_images.shape}")
            # print(f"next_action_q {next_action_q.shape}")
            max_next_q = torch.max(next_action_q, dim=2)[0] 
            # print(f"max_next_q {max_next_q.shape}")
            q_targets = rewards + (1 - dones) * GAMMA * max_next_q

        # Compute Q-values
        action_q, _, _ = self.q_network(images, locations) # This is Q(s,)
        # print(f"action_q {action_q.shape}")
        actions = actions.contiguous().view(B*T, -1) # (B, T) --> (B*T)
        action_q = action_q.contiguous().view(B*T, -1) # (B, T, num actions) --> (B*T, num actions)
        q_values = action_q.gather(1, actions).squeeze(1) # Compute Q(s,a)
        q_values = q_values.view(B,T)
        # print(f"q_targets {q_targets.shape}")
        # print(f"q_values {q_values.shape}")
        # Loss and Optimization
        loss = nn.MSELoss()(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.grad_step += 1
        # Epsilon decay
        # self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
        self.epsilon = max(MIN_EPSILON, ((EXPLORE_STEPS - self.grad_step)/EXPLORE_STEPS)*MAX_EPSILON)

        if self.grad_step % UPDATE_TARGET_EVERY == 0:
            self.update_target_network()

        if self.grad_step % MODEL_SAVE_EVERY == 0:
            torch.save(self.q_network.state_dict(), os.path.join(MODEL_SAVED_DIR, f"ckpt_{self.grad_step}.pth"))

        return loss.item()


# Environment Interaction
def train_drqn(env, num_episodes):
    wandb.init(
        entity="maytusp",
        # set the wandb project where this run will be logged
        project="train_drqn",
        name=f"done if useless action",
        # track hyperparameters and run metadata
        config={
            "batch_size": BATCH_SIZE,
            "seq_length" : SEQ_LENGTH,
            "exploration_steps" : EXPLORE_STEPS,
            "buffer_size" : REPLAY_SIZE,
            "max_eps" : MAX_EPSILON,
            "min_eps" : MIN_EPSILON,
        }
    )
    agent = DQNAgent(ACTION_DIM, MESSAGE_DIM)
    for episode in range(num_episodes):
        episode_data = EpisodeData()
        obs = env.reset()

        # image_seq = [obs['image'][0]]
        # loc_seq = [obs['location'][0]]
        image = [obs['image'][0]]
        loc = [obs['location'][0]]

        done = False
        total_reward = 0
        cum_loss = 0
        grad_step = 0
        ep_step = 0

        # init hidden
        h = torch.randn(1, NUM_LSTM_LAYER, HIDDEN_DIM).to(device)
        c = torch.randn(1, NUM_LSTM_LAYER, HIDDEN_DIM).to(device)
        while not done:
            action, message, (h,c) = agent.select_action(image, loc, (h,c), explore=True)

            env_action = env.int_to_act(action)
            next_obs, rewards, done, _, _ = env.step(env_action)
            
            rec_action = action.detach().cpu().numpy()[0]
            episode_data.put((obs['image'][0], obs['location'][0], rec_action, rewards[0], 
                            next_obs['image'][0], next_obs['location'][0], done))

            loss = agent.train()
            
            if loss is not None:
                total_reward += sum(rewards)
                cum_loss += loss
                grad_step += 1

            if not(done):
                image = [next_obs["image"][0]]
                loc = [next_obs["location"][0]]

            ep_step += 1

        # each replay sample contains full episode
        agent.replay_buffer.add(episode_data)
                                
        if grad_step > 0:
            wandb.log(
                {"loss": cum_loss / grad_step,
                "reward": total_reward,
                "epsilon": agent.epsilon}
            )

        
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, loss: {loss}")


def test_drqn(env, num_episodes, checkpoint_path, visualize=True):
    wandb.init(
        entity="maytusp",
        # set the wandb project where this run will be logged
        project="test_drqn",
        name=f"done if useless action",
        # track hyperparameters and run metadata
        config={
            "batch_size": BATCH_SIZE,
            "seq_length" : SEQ_LENGTH,
            "exploration_steps" : EXPLORE_STEPS,
            "buffer_size" : REPLAY_SIZE,
            "max_eps" : MAX_EPSILON,
            "min_eps" : MIN_EPSILON,
        }
    )
    agent = DQNAgent(ACTION_DIM, MESSAGE_DIM)
    agent.q_network.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.q_network.eval()

    for episode in range(num_episodes):
        frames = []
        episode_data = EpisodeData()
        obs = env.reset()

        # image_seq = [obs['image'][0]]
        # loc_seq = [obs['location'][0]]
        image = [obs['image'][0]]
        loc = [obs['location'][0]]

        done = False
        total_reward = 0
        cum_loss = 0
        grad_step = 0
        ep_step = 0
        if visualize:
            frame = visualize_environment(env, ep_step)
            frames.append(frame.transpose((1, 0, 2)))

        # init hidden
        h = torch.randn(1, NUM_LSTM_LAYER, HIDDEN_DIM).to(device)
        c = torch.randn(1, NUM_LSTM_LAYER, HIDDEN_DIM).to(device)

        while not done:
            action, message, (h,c) = agent.select_action(image, loc, (h,c), explore=True)

            env_action = env.int_to_act(action)
            next_obs, rewards, done, _, _ = env.step(env_action)
            
            rec_action = action.detach().cpu().numpy()[0]

            if visualize:
                frame = visualize_environment(env, ep_step)
                frames.append(frame.transpose((1, 0, 2)))

            if not(done):
                image = [next_obs["image"][0]]
                loc = [next_obs["location"][0]]

            ep_step += 1
            total_reward += sum(rewards)

        if grad_step > 0:
            wandb.log(
                {
                "reward": total_reward,
               }
            )

        
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        if visualize and total_reward > -30:
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_videofile(os.path.join(VIDEO_SAVED_DIR, f"ep_{episode + 1}.mp4"), codec="libx264")
            
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Environment()
    train_drqn(env, 100000)
    # test_drqn(env, 100, "checkpoints/ckpt_480000.pth")