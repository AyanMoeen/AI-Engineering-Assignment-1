"""Positional encodings.

Sinusoidal (required):
    PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
Learned (optional experiment): an nn.Embedding over positions.
"""
import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len, d_model):
    assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)             # (max_len, 1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, x):                                   # x: (B, T, d_model)
        return x + self.pe[: x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.emb(pos)
