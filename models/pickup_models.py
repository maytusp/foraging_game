
# Created: 28 Feb 2025
# Model for pickup_high_v1
# DecTraining DecExec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np

from models.nets import *


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOLSTMCommAgent(nn.Module):
    '''
    Agent with communication
    Observations: [image, location, message]
    '''
    def __init__(self, num_actions, grid_size=5, n_words=16, embedding_size=16, num_channels=1, image_size=3, d_model=128):
        super().__init__()
        self.grid_size = grid_size
        self.n_words = n_words
        self.embedding_size = embedding_size
        self.image_feat_dim = 16
        self.loc_dim = 4
        self.num_channels = num_channels
        self.visual_encoder = nn.Sequential(nn.Flatten(), # (1,5,5) to (25)
                                        nn.Linear(image_size * image_size * num_channels, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, self.image_feat_dim),
                                        nn.ReLU(),
                                        )

        self.message_encoder =  nn.Sequential(nn.Embedding(n_words, embedding_size), # Contains n_words tensor of size embedding_size
                                        nn.Linear(embedding_size, embedding_size), 
                                        nn.ReLU(),
                                        )

        self.location_encoder = nn.Linear(2, self.loc_dim)
        self.lstm = nn.LSTM(self.image_feat_dim+self.loc_dim+self.embedding_size, d_model)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(d_model, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(d_model, 1), std=1)
        self.message_head = layer_init(nn.Linear(d_model, n_words), std=0.01)

    # [IL] reinit only actor & critic heads (keep encoders/LSTM/message head intact)
    def reset_actor_critic(self):
        layer_init(self.actor, std=0.01)
        layer_init(self.critic, std=1)
        
    def get_states(self, input, lstm_state, done, tracks=None):
        batch_size = lstm_state[0].shape[1]
        image, location, message = input
        image_feat = self.visual_encoder(image / 255.0) # (L*B, feat_dim)
        location = location / self.grid_size # (L*B,2)
        location_feat = self.location_encoder(location)
        message_feat = self.message_encoder(message) # (L*B,1)
        message_feat = message_feat.view(-1, self.embedding_size)

        hidden = torch.cat((image_feat, location_feat, message_feat), axis=1)
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

        # For positive signalling and listening
        if message is None:
            message = message_probs.sample()

        return action, action_probs.log_prob(action), action_probs.entropy(), message, message_probs.log_prob(message), message_probs.entropy(), self.critic(hidden), lstm_state


class PPOLSTMCommAgentWithSilence(PPOLSTMCommAgent):
    def __init__(self, num_actions, grid_size=5, n_words=16, embedding_size=16, num_channels=1, image_size=3, d_model=128):
        # Initialize superclass
        super().__init__(num_actions, grid_size, n_words, embedding_size, num_channels, image_size, d_model)
        
        # Requirement 1: Override message encoder to have n_words + 1 elements
        # 0 is reserved for silence, 1 to n_words are actual words
        self.message_encoder = nn.Sequential(
            nn.Embedding(n_words + 1, embedding_size), 
            nn.Linear(embedding_size, embedding_size), 
            nn.ReLU(),
        )

    def get_action_and_value(self, input, lstm_state, done, action=None, message=None, tracks=None, pos_sig=False, pos_lis=False):
        # Unpack input (image, location, received_message)
        # Note: received_message in input is passed to get_states -> message_encoder. 
        # Since message_encoder now has size n_words+1, it can handle inputs [0, n_words] directly.
        image, location, received_message = input
        hidden, lstm_state = self.get_states((image, location, received_message), lstm_state, done, tracks)

        # --- Physical Action Logic (Same as Superclass) ---
        action_logits = self.actor(hidden)
        action_probs = Categorical(logits=action_logits)
        
        if action is None:
            action = action_probs.sample()

        # --- Communication Logic (Modified) ---
        # The message_head outputs logits for 'n_words' (size 16), representing tokens [1...n_words]
        message_logits = self.message_head(hidden)
        message_probs = Categorical(logits=message_logits)
        
        # Requirement 2: Two modes based on whether message is None
        if message is None:
            # Mode 1: Inference (producing message)
            # Sample raw logits in range [0, n_words-1]
            raw_message = message_probs.sample()
            
            # Shift [0, n_words-1] -> [1, n_words]
            # The model never produces 0 (silence) actively
            message = raw_message + 1
            
            # Calculate log probs/entropy based on the raw distribution
            msg_log_prob = message_probs.log_prob(raw_message)
            msg_entropy = message_probs.entropy()
            
        else:
            # Mode 2: Training (evaluating message)
            # 'message' input (from PPO buffer) is in range [0, n_words]
            
            # Shift [1, n_words] -> [0, n_words-1] to match logit dimensions
            # Silence (0) becomes -1, so we clamp to 0. 
            # (Silence entries are masked in the loss function, so value 0 here is a dummy placeholder)
            shifted_message = (message - 1).clamp(min=0)
            
            msg_log_prob = message_probs.log_prob(shifted_message)
            msg_entropy = message_probs.entropy()

        return (action, 
                action_probs.log_prob(action), 
                action_probs.entropy(), 
                message, 
                msg_log_prob, 
                msg_entropy, 
                self.critic(hidden), 
                lstm_state)

class TransformerBlock(nn.Module):
    """
    A tiny GPT-style transformer block:
    - Multi-head self-attention over 2 tokens (memory, current input)
    - Feedforward MLP
    """
    def __init__(self, d_model=128, n_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=False)
        self.ln2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        """
        x: [seq_len (=2), batch_size, d_model]
        """
        # Self-attention
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        # MLP
        h = self.ln2(x)
        mlp_out = self.mlp(h)
        x = x + mlp_out
        return x


class PPOTransformerCommAgent(nn.Module):
    """
    PPO agent with communication, using a tiny GPT-style transformer
    as the recurrent core instead of an LSTM.
    The architecture resmbles Recurrent Memory Transformer.

    Observations: [image, location, message]
    """
    def __init__(
        self,
        num_actions,
        grid_size=5,
        n_words=16,
        embedding_size=16,
        num_channels=1,
        image_size=3,
        d_model=128,
        n_layers=2,
        n_heads=4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_words = n_words
        self.embedding_size = embedding_size
        self.image_feat_dim = 16
        self.loc_dim = 4
        self.num_channels = num_channels
        self.d_model = d_model
        self.num_layers = n_layers

        # --- Encoders ---
        self.visual_encoder = nn.Sequential(
            nn.Flatten(),  # (C, H, W) -> (C*H*W)
            nn.Linear(image_size * image_size * num_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.image_feat_dim),
            nn.ReLU(),
        )

        self.message_encoder = nn.Sequential(
            nn.Embedding(n_words, embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
        )

        self.location_encoder = nn.Linear(2, self.loc_dim)

        # Project concatenated obs features to transformer dimension
        core_input_dim = self.image_feat_dim + self.loc_dim + self.embedding_size
        self.input_proj = nn.Linear(core_input_dim, d_model)

        # Memory token projection (so we can learn how to combine with obs)
        self.memory_proj = nn.Linear(d_model, d_model, bias=False)

        # Tiny GPT-style stack over 2 tokens: [memory_token, obs_token]
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)]
        )

        # --- Heads ---
        self.actor = layer_init(nn.Linear(d_model, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(d_model, 1), std=1.0)
        self.message_head = layer_init(nn.Linear(d_model, n_words), std=0.01)

    # [IL] reinit only actor & critic heads (keep encoders/transformer/message head intact)
    def reset_actor_critic(self):
        layer_init(self.actor, std=0.01)
        layer_init(self.critic, std=1.0)

    # --- Memory helpers ---

    def init_memory(self, batch_size, device):
        """
        Initialize memory state for a batch of environments.
        Returns tensor of shape [batch_size, d_model].
        """
        return torch.zeros(batch_size, self.d_model, device=device)

    # --- Core recurrent logic using transformer memory ---

    def _encode_obs(self, input):
        """
        input: (image, location, message)
          - image: [L*B, C, H, W] or [B, C, H, W]
          - location: [L*B, 2] or [B, 2]
          - message: [L*B] or [B]
        Returns:
          - token_feat: [L*B, core_input_dim]
        """
        image, location, message = input

        image_feat = self.visual_encoder(image / 255.0)  # [L*B, image_feat_dim]

        location = location / self.grid_size
        location_feat = self.location_encoder(location)  # [L*B, loc_dim]

        message_feat = self.message_encoder(message)  # [L*B, 1, emb]
        message_feat = message_feat.view(-1, self.embedding_size)  # [L*B, emb]

        hidden = torch.cat((image_feat, location_feat, message_feat), dim=1)
        return hidden  # [L*B, core_input_dim]

    def get_states(self, input, memory, done, tracks=None):
        """
        input: (image, location, message)
          image:   [L*B, C, H, W] or [B, C, H, W]
          location:[L*B, 2] or [B, 2]
          message: [L*B] or [B]
        memory: [B, d_model]
        done:   [L*B] or [B] (0/1)
        tracks: unused (kept for API compatibility)

        Returns:
          - hidden_flat: [L*B, d_model]
          - new_memory:  [B, d_model]
        """
        # Flatten obs to [L*B, ...]
        token_feat = self._encode_obs(input)  # [L*B, core_input_dim]
        batch_size = memory.shape[0]

        # If we only have B, treat as L=1
        if token_feat.shape[0] == batch_size:
            seq_len = 1
        else:
            assert token_feat.shape[0] % batch_size == 0, "token_feat and memory shapes mismatch"
            seq_len = token_feat.shape[0] // batch_size

        # Reshape to [L, B, *]
        token_feat = token_feat.view(seq_len, batch_size, -1)  # [L, B, core_input_dim]

        done = done.view(seq_len, batch_size)  # [L, B]

        # Project obs to transformer dimension
        token_feat = self.input_proj(token_feat)  # [L, B, d_model]

        # Recurrent over time using memory token
        mem = memory  # [B, d_model]
        all_hidden = []
        for t in range(seq_len):
            x_t = token_feat[t]  # [B, d_model]

            # 2 tokens: [memory_token, obs_token]
            mem_token = self.memory_proj(mem)  # [B, d_model]
            tokens = torch.stack([mem_token, x_t], dim=0)  # [2, B, d_model]

            # Pass through transformer blocks
            for blk in self.blocks:
                tokens = blk(tokens)  # [2, B, d_model]

            # Use last token as output / new memory
            out_t = tokens[1]  # [B, d_model]

            # Reset memory when episode done
            d_t = done[t].view(-1, 1)  # [B, 1]
            mem = (1.0 - d_t) * out_t  # [B, d_model]
            # print(f"mem {mem[:, :4]}")
            all_hidden.append(out_t)

        hidden_seq = torch.stack(all_hidden, dim=0)  # [L, B, d_model]
        hidden_flat = hidden_seq.reshape(seq_len * batch_size, self.d_model)  # [L*B, d_model]
        return hidden_flat, mem  # [L*B, d_model], [B, d_model]

    # --- PPO interface ---

    def get_value(self, x, memory, done):
        hidden, _ = self.get_states(x, memory, done)
        return self.critic(hidden)

    def get_action_and_value(
        self,
        input,
        memory,
        done,
        action=None,
        message=None,
        tracks=None,
        pos_sig=False,
        pos_lis=False,
    ):
        image, location, received_message = input

        # Main forward
        hidden, new_memory = self.get_states(
            (image, location, received_message), memory, done, tracks
        )

        action_logits = self.actor(hidden)
        action_probs = Categorical(logits=action_logits)
        action_pmf = nn.Softmax(dim=1)(action_logits)

        if action is None:
            action = action_probs.sample()

        message_logits = self.message_head(hidden)
        message_probs = Categorical(logits=message_logits)
        message_pmf = nn.Softmax(dim=1)(message_logits)

        # Counterfactual for listening
        if pos_lis:
            zero_message = torch.zeros_like(received_message).to(received_message.device)
            hidden_cf, _ = self.get_states(
                (image, location, zero_message), memory, done, tracks
            )
            action_cf_logits = self.actor(hidden_cf)
            action_cf_pmf = nn.Softmax(dim=1)(action_cf_logits)
        # Sample message if needed
        if message is None:
            message = message_probs.sample()
            
        return (
            action,
            action_probs.log_prob(action),
            action_probs.entropy(),
            message,
            message_probs.log_prob(message),
            message_probs.entropy(),
            self.critic(hidden),
            new_memory,
        )
