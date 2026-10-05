"""Decoder-style Transformer block (pre-LayerNorm):

    x = x + Attention(LayerNorm(x))     # residual around attention
    x = x + FFN(LayerNorm(x))           # residual around feed-forward

Pre-LN is used because it trains more stably for small models without long warm-up.
"""
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim=None, dropout=0.0, causal=True):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x, return_weights=False):
        if return_weights:
            a, w = self.attn(self.ln1(x), return_weights=True)
        else:
            a, w = self.attn(self.ln1(x)), None
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return (x, w) if return_weights else x
