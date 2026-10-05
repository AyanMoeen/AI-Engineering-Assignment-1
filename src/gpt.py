"""MiniGPT: token embedding + position + causal Transformer blocks + LM head."""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding
from .transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: Optional[int] = None      # None -> 4 * d_model
    dropout: float = 0.1
    pos_type: str = "sinusoidal"       # sinusoidal | learned
    tie_weights: bool = False


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_type == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(cfg.d_model, cfg.context_length)
        elif cfg.pos_type == "learned":
            self.pos = LearnedPositionalEncoding(cfg.d_model, cfg.context_length)
        else:
            raise ValueError(f"unknown pos_type: {cfg.pos_type}")
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout, causal=True)
             for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, return_attn=False):
        """idx: (B, T) token ids, targets: (B, T) next-token ids (idx shifted by one)."""
        B, T = idx.shape
        assert T <= self.cfg.context_length, "sequence longer than context_length"
        x = self.tok_emb(idx) * math.sqrt(self.cfg.d_model)      # (B, T, d_model)
        x = self.drop(self.pos(x))
        attns = []
        for block in self.blocks:
            if return_attn:
                x, w = block(x, return_weights=True)
                attns.append(w)
            else:
                x = block(x)
        logits = self.lm_head(self.ln_f(x))                      # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return (logits, loss, attns) if return_attn else (logits, loss)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """Autoregressive sampling. temperature == 0 -> greedy decoding."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                            # only the last position
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        self.train(was_training)
        return idx
