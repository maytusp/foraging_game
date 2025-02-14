
# Created: 14 Feb
# Goal-Conditioned Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np

from nets import *


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOLSTMCommAgentGoal(nn.Module):
    '''
    Agent with communication
    Observations: [image, location, message]
    '''
    def __init__(self, num_actions, grid_size=10, n_words=16, embedding_size=16, num_channels=None, N_val=None, N_att=None, image_size=None):
        super().__init__()
        self.grid_size = grid_size
        self.n_words = n_words
        self.embedding_size = embedding_size
        self.image_feat_dim = 64
        self.loc_dim = 8
        self.N_val = N_val # the highest value of the observation
        self.num_channels = num_channels
        self.visual_encoder = nn.Sequential(nn.Flatten(), # (1,5,5) to (25)
                                        nn.Linear(image_size * image_size * num_channels, 256), 
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, self.image_feat_dim),
                                        nn.ReLU(),
                                        )

        self.message_encoder =  nn.Sequential(nn.Embedding(n_words, embedding_size), # Contains n_words tensor of size embedding_size
                                        nn.Linear(embedding_size, embedding_size), 
                                        nn.ReLU(),
                                        )
        self.location_encoder = nn.Linear(2, self.loc_dim)
        self.feature_fusion = nn.Sequential(nn.Linear(self.image_feat_dim+self.loc_dim+self.embedding_size+N_att, 256), 
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
        self.message_head = layer_init(nn.Linear(128, n_words), std=0.01)

    def get_states(self, input, lstm_state, done, tracks=None):
        batch_size = lstm_state[0].shape[1]
        image, location, message, goal = input
        image_feat = self.visual_encoder(image / self.N_val) # (L*B, feat_dim)
        location = location / self.grid_size # (L*B,2)
        location_feat = self.location_encoder(location)
        message_feat = self.message_encoder(message) # (L*B,1)
        message_feat = message_feat.view(-1, self.embedding_size)
        goal_feat = goal / self.N_val

        hidden = torch.cat((image_feat, location_feat, message_feat, goal_feat), axis=1)
        
        # print("hidden", hidden.shape)
        # LSTM logic
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        if tracks is not None:
            tracks = tracks.reshape((-1, batch_size))
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

    def get_action_and_value(self, input, lstm_state, done, action=None, message=None, tracks=None, pos_sig=False, pos_lis=False):
        image, location, received_message = input
        hidden, lstm_state = self.get_states((image, location, received_message), lstm_state, done, tracks)

        action_logits = self.actor(hidden)
        action_probs = Categorical(logits=action_logits)
        action_pmf = nn.Softmax(dim=1)(action_logits) 
        if action is None:
            action = action_probs.sample()

        message_logits = self.message_head(hidden)
        message_probs = Categorical(logits=message_logits)
        message_pmf = nn.Softmax(dim=1)(message_logits) # probability mass function of message

        if pos_lis:
            # create counterfactual case where message is zero
            zero_message = torch.zeros_like(received_message).to(received_message.device)
            hidden_cf, _ = self.get_states((image, location, zero_message), lstm_state, done, tracks)
            action_cf_logits = self.actor(hidden_cf)
            action_cf_pmf = nn.Softmax(dim=1)(action_cf_logits)

        # For positive signalling and listening
        if message is None:
            message = message_probs.sample()
        if pos_sig and not(pos_lis):
            return action, action_probs.log_prob(action), action_probs.entropy(), message, message_probs.log_prob(message), message_probs.entropy(), self.critic(hidden), lstm_state, message_pmf
        elif pos_lis and not(pos_sig):
            return action, action_probs.log_prob(action), action_probs.entropy(), message, message_probs.log_prob(message), message_probs.entropy(), self.critic(hidden), lstm_state, action_pmf, action_cf_pmf
        elif pos_sig and pos_lis: 
            return action, action_probs.log_prob(action), action_probs.entropy(), message, message_probs.log_prob(message), message_probs.entropy(), self.critic(hidden), lstm_state, action_pmf, action_cf_pmf, message_pmf
        else:
            return action, action_probs.log_prob(action), action_probs.entropy(), message, message_probs.log_prob(message), message_probs.entropy(), self.critic(hidden), lstm_state
