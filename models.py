import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

from nets import *


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOLSTMAgent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.nonlinear = nn.Sequential(nn.Flatten(), # (1,5,5) to (25)
                                        nn.Linear(25, 256), 
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        )       
        self.lstm = nn.LSTM(256, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        # print(f"network input {x.shape}")
        # hidden = self.nonlinear(self.network(x / 255.0)) # CNN
        hidden = self.nonlinear(x / 255.0) # MLP
        # print("hidden before", hidden.shape)
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        # print("lstm_state[0]", lstm_state[0].shape)
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        # print("hidden after", hidden.shape)
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

class PPOLSTMCommAgent(nn.Module):
    '''
    Agent with communication
    Observations: [image, location, energy, message]
    '''
    def __init__(self, num_actions, grid_size=10, max_energy=200, n_words=10, n_embedding=3):
        super().__init__()
        self.grid_size = grid_size
        self.max_energy = max_energy
        self.n_words = n_words
        self.n_embedding = n_embedding
        self.image_feat_dim = 256
        self.loc_dim = 2
        self.energy_dim = 1
        self.visual_encoder = nn.Sequential(nn.Flatten(), # (1,5,5) to (25)
                                        nn.Linear(25, 256), 
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        )   
        self.message_encoder =  nn.Embedding(n_words, n_embedding) # Contains n_words tensor of size n_embedding
        self.lstm = nn.LSTM(self.image_feat_dim+self.loc_dim+self.energy_dim+self.n_embedding, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)
        self.message_head = layer_init(nn.Linear(128, n_words), std=0.01)

    def get_states(self, input, lstm_state, done):
        batch_size = lstm_state[0].shape[1]
        image, location, energy, message = input
        image_feat = self.visual_encoder(image / 255.0) # (L*B, feat_dim)
        location = location / self.grid_size # (L*B,2)
        energy = energy / self.max_energy # (L*B,1)
        message_feat = self.message_encoder(message) # (L*B,1)
        message_feat = message_feat.view(-1, self.n_embedding)

        # print(f"image_feat {image_feat.shape}, location {location.shape}, energy {energy.shape}, message_feat {message_feat.shape}")
        hidden = torch.cat((image_feat, location, energy, message_feat), axis=1)
        # print("hidden", hidden.shape)
        # LSTM logic
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, input, lstm_state, done, action=None, message=None):
        image, location, energy, received_message = input
        hidden, lstm_state = self.get_states((image, location, energy, received_message), lstm_state, done)

        action_logits = self.actor(hidden)
        action_probs = Categorical(logits=action_logits)
        if action is None:
            action = action_probs.sample()

        message_logits = self.actor(hidden)
        message_probs = Categorical(logits=message_logits)
        if message is None:
            message = message_probs.sample()

        # print(f"action_logits {action_logits.shape}")
        # print(f"action_probs {action_probs}")
        # print(f"action {action.shape}")
        # print(f"action_probs.log_prob(action) {action_probs.log_prob(action).shape}")
        # print("___________________________")
        # print(f"message_logits {message_logits.shape}")
        # print(f"message_probs {message_probs}")
        # print(f"message {message.shape}")
        # print(f"message_probs.log_prob(message) { message_probs.log_prob(message).shape}")
        return action, action_probs.log_prob(action), action_probs.entropy(), message, message_probs.log_prob(message), message_probs.entropy(), self.critic(hidden), lstm_state