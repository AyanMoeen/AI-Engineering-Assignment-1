import math

import torch

from src.attention import MultiHeadAttention
from src.gpt import GPT, GPTConfig
from src.positional_encoding import SinusoidalPositionalEncoding, sinusoidal_table
from src.transformer_block import TransformerBlock
from src.utils import count_parameters


def test_split_and_merge_heads_are_inverse():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(2, 6, 32)
    h = mha.split_heads(x)
    assert h.shape == (2, 4, 6, 8)
    assert torch.equal(mha.merge_heads(h), x)


def test_trace_shapes():
    mha = MultiHeadAttention(32, 4)
    trace = dict(mha.trace_shapes(torch.randn(2, 6, 32)))
    assert trace["Q after split_heads"] == (2, 4, 6, 8)
    assert trace["attention weights (softmax)"] == (2, 4, 6, 6)
    assert trace["after output projection W_o"] == (2, 6, 32)


def test_sinusoidal_values():
    pe = sinusoidal_table(10, 8)
    assert pe.shape == (10, 8)
    assert torch.allclose(pe[0], torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.]))   # sin(0)=0, cos(0)=1
    assert torch.allclose(pe[1, 0], torch.sin(torch.tensor(1.0)))
    x = torch.zeros(2, 5, 8)
    assert SinusoidalPositionalEncoding(8, 10)(x).shape == (2, 5, 8)


def test_block_keeps_shape():
    blk = TransformerBlock(32, 4, dropout=0.0)
    assert blk(torch.randn(2, 9, 32)).shape == (2, 9, 32)


def test_gpt_logits_shape_and_initial_loss():
    cfg = GPTConfig(vocab_size=65, context_length=32, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    model = GPT(cfg)
    idx = torch.randint(0, 65, (4, 32))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 32, 65)
    assert abs(loss.item() - math.log(65)) < 0.3          # random model ~ ln(vocab)
    assert count_parameters(model) > 0


def test_learned_positions_and_weight_tying_work():
    cfg = GPTConfig(vocab_size=20, context_length=16, d_model=32, n_heads=2, n_layers=1,
                    pos_type="learned", tie_weights=True)
    model = GPT(cfg)
    assert model.lm_head.weight is model.tok_emb.weight
    assert model(torch.randint(0, 20, (2, 16)))[0].shape == (2, 16, 20)


def test_generate_length_and_greedy_is_deterministic():
    cfg = GPTConfig(vocab_size=20, context_length=8, d_model=32, n_heads=2, n_layers=1, dropout=0.0)
    model = GPT(cfg)
    idx = torch.zeros(1, 3, dtype=torch.long)
    a = model.generate(idx, 20, temperature=0)            # longer than context: window is cropped
    b = model.generate(idx, 20, temperature=0)
    assert a.shape == (1, 23) and torch.equal(a, b)
