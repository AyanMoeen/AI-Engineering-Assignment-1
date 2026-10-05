"""Self-attention and multi-head attention, written by hand.

Mask convention: a boolean mask where True = "allowed to attend".
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T, device=None):
    """(T, T) lower-triangular mask. Row t may look at columns <= t only."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """softmax(Q K^T / sqrt(d_k)) V

    q, k, v : (..., T, d)       mask : broadcastable to (..., T, T), True = keep
    returns : output (..., T, d_v), weights (..., T, T)
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)      # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))   # future -> -inf -> weight 0
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


def self_attention(x, w_q, w_k, w_v, causal=False):
    """Single-head self-attention on raw tensors.
    x: (T, d_model), w_*: (d_model, d_k)."""
    q, k, v = x @ w_q, x @ w_k, x @ w_v
    mask = causal_mask(x.size(0), x.device) if causal else None
    return scaled_dot_product_attention(q, k, v, mask)


class SelfAttention(nn.Module):
    """Single-head self-attention layer with learnable Q/K/V projections."""

    def __init__(self, d_model, d_k, causal=True):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_k, bias=False)
        self.k_proj = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_k, bias=False)
        self.causal = causal

    def forward(self, x):                                   # x: (B, T, d_model)
        mask = causal_mask(x.size(1), x.device) if self.causal else None
        return scaled_dot_product_attention(self.q_proj(x), self.k_proj(x), self.v_proj(x), mask)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with explicit split / merge of heads."""

    def __init__(self, d_model, n_heads, dropout=0.0, causal=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """(B, T, d_model) -> (B, H, T, d_k)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x):
        """(B, H, T, d_k) -> (B, T, d_model)"""
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def forward(self, x, return_weights=False):
        T = x.size(1)
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        mask = causal_mask(T, x.device) if self.causal else None   # (T, T) broadcasts over (B, H)
        out, weights = scaled_dot_product_attention(q, k, v, mask, self.attn_dropout)
        out = self.resid_dropout(self.out_proj(self.merge_heads(out)))
        return (out, weights) if return_weights else out

    @torch.no_grad()
    def trace_shapes(self, x):
        """List of (step name, tensor shape) through the whole module (for Figure 3)."""
        B, T, _ = x.shape
        q = self.q_proj(x)
        qh = self.split_heads(q)
        scores = qh @ self.split_heads(self.k_proj(x)).transpose(-2, -1)
        out, w = scaled_dot_product_attention(qh, self.split_heads(self.k_proj(x)),
                                              self.split_heads(self.v_proj(x)),
                                              causal_mask(T, x.device) if self.causal else None)
        merged = self.merge_heads(out)
        return [
            ("input x", tuple(x.shape)),
            ("Q = x W_q (before split)", tuple(q.shape)),
            ("Q after split_heads", tuple(qh.shape)),
            ("scores = Q K^T / sqrt(d_k)", tuple(scores.shape)),
            ("attention weights (softmax)", tuple(w.shape)),
            ("weights @ V (per head)", tuple(out.shape)),
            ("after merge_heads (concat)", tuple(merged.shape)),
            ("after output projection W_o", tuple(self.out_proj(merged).shape)),
        ]
