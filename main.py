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

mode = "train" # train, test
exp_name = "home_ob"
wandb_log = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = f"checkpoints/{exp_name}/ckpt_80000.pth"

# Save parameters
MODEL_SAVE_EVERY = 10000 # 500 steps
MODEL_SAVED_DIR = f"checkpoints/{exp_name}/"
os.makedirs(MODEL_SAVED_DIR, exist_ok=True)

# Constants
MAX_STEPS = 30
ACTION_DIM = 6  # Action space size
MESSAGE_DIM = 10  # Length of the message vector
SEQ_LENGTH = 10 # Sequence length for LSTM
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
REPLAY_SIZE = 1000 # episodes
MAX_EPSILON = 1.0
MIN_EPSILON = 0.2
EXPLORE_STEPS = 3e4
UPDATE_TARGET_EVERY = 20

# Visual Observation
INPUT_CHANNELS = 4
IMAGE_SIZE = 5

# Message
VOCAB_SIZE = 32
EMBED_DIM = 64

HIDDEN_DIM = 64
NUM_LSTM_LAYER = 1
#EVAL
VISUALIZE = True
VIDEO_SAVED_DIR = f"vids/{exp_name}/"
os.makedirs(VIDEO_SAVED_DIR, exist_ok=True)


if VISUALIZE:
    from visualize import *
    from moviepy.editor import *

# Define the LSTM-based Q-Network without message
class LSTM_QNetwork(nn.Module):
    def __init__(self, input_channels, image_size, hidden_dim, action_dim, vocab_size, message_dim, cnn_skip=True):
        super(LSTM_QNetwork, self).__init__()
        self.observation_encoder = CNNEncoder(input_channels, hidden_dim)
        # For calculating input size
        obs_feat = self.observation_encoder(torch.zeros(1, input_channels, image_size, image_size))
        obs_feat_dim = obs_feat.shape[1]
        loc_feat_dim = obs_feat_dim
        self.location_encoder =  nn.Linear(2, loc_feat_dim, bias=False) # Location encoding
        input_dim = obs_feat_dim + loc_feat_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.cnn_skip = cnn_skip
        if cnn_skip:
            self.action_head = MLP([hidden_dim+obs_feat_dim+loc_feat_dim, hidden_dim, action_dim])
        else:
            self.action_head = MLP([hidden_dim, hidden_dim, action_dim])
        

    def input_norm(self, obs, location):
        obs = obs / 255
        location = location / GRID_SIZE
        return obs, location

    # without message
    def forward(self, obs, location, hidden=None):
        # obs: (batch_size, L, 5, 5, C)
        # location: (batch_size, L, 2)
        obs, location = self.input_norm(obs, location)
        
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

        lstm_out = lstm_out.contiguous().view(B*T, -1)
        lstm_out = torch.relu(self.fc(lstm_out))

        if self.cnn_skip:
            combined = combined.contiguous().view(B*T, -1)
            head_input = torch.cat((lstm_out, combined), dim=1)
            action_q = self.action_head(head_input)
            action_q = action_q.view(B, T, action_q.shape[1])
        else:
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
        rewards =  torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
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
            max_next_q = torch.max(next_action_q, dim=2)[0] 
            q_targets = rewards + (1 - dones) * GAMMA * max_next_q
            # print(dones)
            # print((1 - dones) * GAMMA * max_next_q)

        # Compute Q-values
        action_q, _, _ = self.q_network(images, locations) # This is Q(s,)
        actions = actions.contiguous().view(B*T, -1) # (B, T) --> (B*T)
        action_q = action_q.contiguous().view(B*T, -1) # (B, T, num actions) --> (B*T, num actions)
        q_values = action_q.gather(1, actions).squeeze(1) # Compute Q(s,a)
        q_values = q_values.view(B,T)

        # Loss and Optimization
        loss = nn.MSELoss()(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.grad_step += 1

        # Epsilon decay
        self.epsilon = max(MIN_EPSILON, ((EXPLORE_STEPS - self.grad_step)/EXPLORE_STEPS)*MAX_EPSILON)

        if self.grad_step % UPDATE_TARGET_EVERY == 0:
            self.update_target_network()

        if self.grad_step % MODEL_SAVE_EVERY == 0:
            torch.save(self.q_network.state_dict(), os.path.join(MODEL_SAVED_DIR, f"ckpt_{self.grad_step}.pth"))

        return loss.item()


# Environment Interaction
def train_drqn(env, num_episodes):
    if wandb_log:
        wandb.init(
            entity="maytusp",
            # set the wandb project where this run will be logged
            project="train_drqn",
            name=f"{exp_name}",
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
        while not done or ep_step == MAX_STEPS:
            action, message, (h,c) = agent.select_action(image, loc, (h,c), explore=True)

            next_obs, rewards, done, _, _ = env.step(action)
            
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
            # print(f"step:{ep_step}: {rewards}")
            ep_step += 1

        # each replay sample contains full episode
        agent.replay_buffer.add(episode_data)

        if grad_step > 0 and wandb_log:
            wandb.log(
                {"loss": cum_loss / grad_step,
                "reward": total_reward,
                "epsilon": agent.epsilon}
            )

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, loss: {loss}")


def test_drqn(env, num_episodes, checkpoint_path, visualize=True):
    if wandb_log:
        wandb.init(
            entity="maytusp",
            # set the wandb project where this run will be logged
            project="test_drqn",
            name=f"{exp_name}",
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
            time.sleep(2)

        # init hidden
        h = torch.randn(1, NUM_LSTM_LAYER, HIDDEN_DIM).to(device)
        c = torch.randn(1, NUM_LSTM_LAYER, HIDDEN_DIM).to(device)

        while not done or ep_step == MAX_STEPS:
            
            action, message, (h,c) = agent.select_action(image, loc, (h,c), explore=True)

            next_obs, rewards, done, _, _ = env.step(action)
            
            rec_action = action.detach().cpu().numpy()[0]


            if not(done):
                image = [next_obs["image"][0]]
                loc = [next_obs["location"][0]]

            if visualize:
                print("image \n", np.sum(image[0], axis=2))
                print("loc \n", next_obs["location"][0])
                frame = visualize_environment(env, ep_step)
                frames.append(frame.transpose((1, 0, 2)))
                time.sleep(1)
            
            ep_step += 1
            total_reward += sum(rewards)

        if grad_step > 0 and wandb_log:
            wandb.log(
                {
                "reward": total_reward,
               }
            )

        
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        if visualize: # and total_reward > 5:
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_videofile(os.path.join(VIDEO_SAVED_DIR, f"ep_{episode + 1}.mp4"), codec="libx264")
            
if __name__ == "__main__":
    env = Environment()
    if mode == "train":
        train_drqn(env, 100000)
    elif mode == "test":
        test_drqn(env, 1000, ckpt_path)