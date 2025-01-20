# Edit: 30ec2024: Deeper Message Encoder and Decoder
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


class PPOLSTMAgent(nn.Module):
    '''
    Agent with communication
    Observations: [image, location, energy, message]
    '''
    def __init__(self, num_actions, grid_size=10, max_energy=200, n_words=10, n_embedding=32):
        super().__init__()
        self.grid_size = grid_size
        self.max_energy = max_energy
        self.n_words = n_words
        self.n_embedding = n_embedding
        self.image_feat_dim = 32
        self.loc_dim = 32
        self.energy_dim = 32
        self.visual_encoder = nn.Sequential(nn.Flatten(), # (1,5,5) to (25)
                                        nn.Linear(25, 256), 
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, self.image_feat_dim),
                                        nn.ReLU(),
                                        )   
        self.message_encoder =  nn.Embedding(n_words, n_embedding) # Contains n_words tensor of size n_embedding
        self.energy_encoder = nn.Linear(1, self.energy_dim)
        self.location_encoder = nn.Linear(2, self.loc_dim)
        self.lstm = nn.LSTM(self.image_feat_dim+self.loc_dim+self.energy_dim+self.n_embedding, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, input, lstm_state, done):
        batch_size = lstm_state[0].shape[1]
        image, location, energy = input
        image_feat = self.visual_encoder(image / 255.0) # (L*B, feat_dim)
        location = location / self.grid_size # (L*B,2)
        energy = energy / self.max_energy # (L*B,1)
        energy_feat = self.energy_encoder(energy)
        location_feat = self.location_encoder(location)
        # print(f"image_feat {image_feat.shape}, location {location.shape}, energy {energy.shape}, message_feat {message_feat.shape}")
        hidden = torch.cat((image_feat, location_feat, energy_feat), axis=1)
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
        image, location, energy = input
        hidden, lstm_state = self.get_states((image, location, energy), lstm_state, done)

        action_logits = self.actor(hidden)
        action_probs = Categorical(logits=action_logits)
        if action is None:
            action = action_probs.sample()

        return action, action_probs.log_prob(action), action_probs.entropy(), self.critic(hidden), lstm_state

class PPOLSTMCommAgent(nn.Module):
    '''
    Agent with communication
    Observations: [image, location, energy, message]
    '''
    def __init__(self, num_actions, grid_size=10, max_energy=200, n_words=10, n_embedding=4, num_channels=1):
        super().__init__()
        self.grid_size = grid_size
        self.max_energy = max_energy
        self.n_words = n_words
        self.n_embedding = n_embedding
        self.image_feat_dim = 16
        self.loc_dim = 4
        self.energy_dim = 4
        self.num_channels = num_channels
        self.visual_encoder = nn.Sequential(nn.Flatten(), # (1,5,5) to (25)
                                        nn.Linear(25*num_channels, 256), 
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, self.image_feat_dim),
                                        nn.ReLU(),
                                        )

        self.message_encoder =  nn.Sequential(nn.Embedding(n_words, n_embedding), # Contains n_words tensor of size n_embedding
                                        nn.Linear(n_embedding, n_embedding), 
                                        nn.ReLU(),
                                        )

        self.energy_encoder = nn.Linear(1, self.energy_dim)
        self.location_encoder = nn.Linear(2, self.loc_dim)
        self.lstm = nn.LSTM(self.image_feat_dim+self.loc_dim+self.energy_dim+self.n_embedding, 128)
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
        image, location, energy, message = input
        image_feat = self.visual_encoder(image / 255.0) # (L*B, feat_dim)
        location = location / self.grid_size # (L*B,2)
        energy = energy / self.max_energy # (L*B,1)
        energy_feat = self.energy_encoder(energy)
        location_feat = self.location_encoder(location)
        message_feat = self.message_encoder(message) # (L*B,1)
        message_feat = message_feat.view(-1, self.n_embedding)

        # print(f"image_feat {image_feat.shape}, location {location.shape}, energy {energy.shape}, message_feat {message_feat.shape}")
        hidden = torch.cat((image_feat, location_feat, energy_feat, message_feat), axis=1)
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
        image, location, energy, received_message = input
        hidden, lstm_state = self.get_states((image, location, energy, received_message), lstm_state, done, tracks)

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
            hidden_cf, _ = self.get_states((image, location, energy, zero_message), lstm_state, done, tracks)
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


class PPOLSTMDIALAgent(nn.Module):
    '''
    Agent with communication
    Observations: [image, location, energy, message]
    '''
    def __init__(self, num_actions, grid_size=10, max_energy=200, n_embedding=4, num_channels=1):
        super().__init__()
        self.grid_size = grid_size
        self.n_embedding = n_embedding
        self.image_feat_dim = 16
        self.loc_dim = 4
        self.num_channels = num_channels
        self.visual_encoder = nn.Sequential(nn.Flatten(), # (1,5,5) to (25)
                                        nn.Linear(25*num_channels, 256), 
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, self.image_feat_dim),
                                        nn.ReLU(),
                                        )

        self.message_encoder =  nn.Sequential(nn.Linear(1, self.n_embedding), 
                                            nn.ReLU(),
                                        )

        self.location_encoder = nn.Linear(2, self.loc_dim)
        self.lstm = nn.LSTM(self.image_feat_dim+self.loc_dim+self.n_embedding, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)
        self.message_head = nn.Linear(128, 1)

        self.sigma = 1

    def get_states(self, input, lstm_state, done, tracks=None):
        batch_size = lstm_state[0].shape[1]
        image, location, message = input

        # reset message to zero when a new episode starts. Without this line, the last message of the previous episode will be the first message here
        message = (1.0-done).view(-1,1) * message 

        image_feat = self.visual_encoder(image / 255.0) # (L*B, feat_dim)
        location = location / self.grid_size # (L*B,2)

        location_feat = self.location_encoder(location)
        message_feat = self.message_encoder(message) # (L*B,1)
        message_feat = message_feat.view(-1, self.n_embedding)
       
        hidden = torch.cat((image_feat, location_feat, message_feat), axis=1)

        # LSTM logic
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        if tracks is not None:
            tracks = tracks.reshape((-1, batch_size))
        new_hidden = []
        print(f"hidden {hidden.shape}")
        print(f"len zip {len(tuple(zip(hidden, done)))}")
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

    def get_action_and_value(self, input, lstm_state, done, past_action=None, past_message=None, tracks=None, train_mode=False):
        # Train mode: image and location have size of (T*B, *). lstm_state has size of (1, B, N_h)
        # Interaction mode: image and location have size of (B, *) lstm_state has size of (1, B, N_h)
        # received_message has size of (B, 1)

        image, location, received_message = input # if train_mode = True we only need the first message

        if train_mode:
            batch_size = image.shape[1]
            seq_len = image.shape[0]
            hiddens = []
            message = []
            message_logits = []
            # image = image.view(seq_len, batch_size, -1)
            # location = location.view(seq_len, batch_size, -1)
            # print(f"image {image.shape}")
            # print(f"location {location.shape}")

            for t in range(seq_len):
                hidden, lstm_state = self.get_states((image[t], location[t], received_message), lstm_state, done[t], tracks)
                m, m_logit, _ = self.get_message(hidden, train_mode=True) # m --> (B,1)
                received_message = self.swap_message(m)
                # print(f"infer step {t} {hidden.shape}")

                hiddens.append(hidden)
                message.append(m)
                message_logits.append(m_logit)
            hiddens = torch.cat(hiddens, dim=0)
            # message = torch.cat(message, dim=0)
            message_logits = torch.cat(message_logits, dim=0)
            message_probs = Normal(message_logits, self.sigma)

        else:
            hiddens, _ = self.get_states((image, location, received_message), lstm_state, done, tracks)
            message, _, message_probs = self.get_message(hiddens, train_mode=False)
        

        action_logits = self.actor(hiddens)
        action_probs = Categorical(logits=action_logits)
        action_pmf = nn.Softmax(dim=1)(action_logits)

        if past_action is None:
            action = action_probs.sample()
        else:
            action = past_action

        # TODO We have to check this very carefully
        if past_message is not(None):
            message = past_message
            
        return action, action_probs.log_prob(action), action_probs.entropy(), message, message_probs.log_prob(message), message_probs.entropy(), self.critic(hiddens), lstm_state

    def get_message(self, hidden, train_mode=False):
        message_logits = self.message_head(hidden)

        if train_mode:
            # DRU Operation: follows https://github.com/minqi/learning-to-communicate-pytorch/
            # Regularization based on RIAL/DIAL paper (Foerster et. al.)
            message = message_logits + torch.randn(message_logits.size()).to(message_logits.device) * self.sigma
            message = torch.sigmoid(message).float()
            message_probs = None
        else:
            message_probs = Normal(message_logits, self.sigma)
            message = message_probs.sample()
            # Discretization based on RIAL/DIAL paper (Foerster et. al.)
            message =  (message.gt(0.5).float() - 0.5).sign().float()

        return message, message_logits, message_probs

    def swap_message(self, s_message):
        '''
        Input: s_message with the size of (B, 1)
        Output: r_message with the size of (B, 1)

        Description:
        s_message is in the form of [a1_message1, a2_message1, a1_message2, a2_message2, ...]
        r_message is swap betweeen a1 and a2 in the way that
        [a2_message1, a1_message1, a2_message1, a1_message1, ...]
        '''
        # Reshape s_message to (B/2, 2, 1) assuming B is even
        # Note This is for only two agents.

        s_message = s_message.view(-1, 2, 1)  # Each row contains [a1, a2]

        # Swap along dimension 1
        r_message = s_message.flip(dims=[1])  # Flip [a1, a2] -> [a2, a1]

        # Reshape back to (B, 1)
        r_message = r_message.view(-1, 1)
        return r_message