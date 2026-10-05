import math

import torch
import torch.nn.functional as F

from src.attention import scaled_dot_product_attention, self_attention, causal_mask, MultiHeadAttention


def test_hand_example_matches_manual_numbers():
    Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    V = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    out, w = scaled_dot_product_attention(Q, K, V)
    manual_scores = (Q @ K.T) / math.sqrt(2)
    manual_w = torch.softmax(manual_scores, dim=-1)
    assert torch.allclose(w, manual_w)
    assert torch.allclose(out, manual_w @ V)
    assert torch.allclose(w.sum(-1), torch.ones(3))          # rows sum to 1


def test_output_shapes():
    q = torch.randn(2, 4, 5, 8)
    out, w = scaled_dot_product_attention(q, q, q)
    assert out.shape == (2, 4, 5, 8) and w.shape == (2, 4, 5, 5)


def test_matches_pytorch_reference_causal():
    q, k, v = (torch.randn(2, 3, 6, 8) for _ in range(3))
    ours, _ = scaled_dot_product_attention(q, k, v, causal_mask(6))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_self_attention_function_causal_blocks_future():
    x = torch.randn(5, 6)
    w = [torch.randn(6, 4) for _ in range(3)]
    _, weights = self_attention(x, *w, causal=True)
    assert weights.shape == (5, 5)
    assert torch.all(torch.triu(weights, diagonal=1) == 0)   # nothing above the diagonal


def test_multihead_output_shape_and_weights():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(3, 10, 32)
    out, w = mha(x, return_weights=True)
    assert out.shape == (3, 10, 32) and w.shape == (3, 4, 10, 10)
