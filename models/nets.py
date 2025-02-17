import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class CNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels*8, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_channels*8, input_channels*16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_channels*16, input_channels*32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_channels*32, input_channels*32, kernel_size=2, stride=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return x


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


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4),
            num_layers
        )
        self.max_len = max_len

    def forward(self, x):
        # x: (batch_size, L), where L <= max_len
        batch_size, seq_len = x.size()
        assert seq_len <= self.max_len, f"Input length {seq_len} exceeds maximum length {self.max_len}"
        embedded = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        return self.encoder(embedded)  # Output shape: (batch_size, seq_len, embed_dim)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_heads, num_layers, max_len):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4),
            num_layers
        )
        self.output_head = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len
        self.embed_dim = embed_dim

    def forward(self, hidden_state, memory, start_token, eos_token):
        # hidden_state: (batch_size, hidden_dim)
        # memory: (batch_size, seq_len, embed_dim), encoded features from TransformerEncoder
        # start_token: Integer, index of the start token
        # eos_token: Integer, index of the end token

        batch_size = hidden_state.size(0)
        device = hidden_state.device

        # Initialize input and outputs
        outputs = []
        generated_token = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        positional_encodings = self.positional_encoding[:, :self.max_len, :]

        for t in range(self.max_len):
            # Embed the current token and add positional encoding
            tgt_embedded = self.embedding(generated_token) + positional_encodings[:, :t+1, :]
            # Mask future positions
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(t + 1).to(device)

            # Decode with the Transformer
            decoded = self.decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
            token_logits = self.output_head(decoded[:, -1, :])  # Get logits for the last token
            next_token = token_logits.argmax(dim=-1, keepdim=True)  # Sample the most likely token

            # Append to the output
            outputs.append(next_token)

            # Update generated token for the next step
            generated_token = torch.cat([generated_token, next_token], dim=1)

            # Stop generation if all batches have produced the EOS token
            if (next_token == eos_token).all():
                break

        # Concatenate outputs to form the full sequence
        outputs = torch.cat(outputs, dim=1)
        return outputs  # Output shape: (batch_size, seq_len)
