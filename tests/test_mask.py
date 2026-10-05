import torch

from src.attention import MultiHeadAttention, causal_mask
from src.gpt import GPT, GPTConfig


def test_causal_mask_is_lower_triangular():
    m = causal_mask(4)
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    assert not m[0, 1] and m[1, 0] and m[2, 2]


def test_future_attention_weights_are_zero():
    mha = MultiHeadAttention(16, 2, causal=True)
    _, w = mha(torch.randn(2, 7, 16), return_weights=True)
    assert torch.all(torch.triu(w, diagonal=1) == 0)


def test_attention_layer_ignores_future_tokens():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 2, causal=True).eval()
    x = torch.randn(1, 8, 16)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, 16)                 # change the future (positions 5..7)
    assert torch.allclose(mha(x)[:, :5], mha(x2)[:, :5], atol=1e-6)


def test_gpt_never_uses_future_tokens():
    """Causal correctness: logits at position t must not depend on tokens at positions > t."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=30, context_length=16, d_model=32, n_heads=4, n_layers=3, dropout=0.0)
    model = GPT(cfg).eval()
    idx = torch.randint(0, 30, (2, 12))
    idx2 = idx.clone()
    idx2[:, 7:] = torch.randint(0, 30, (2, 5))        # change tokens from position 7 onward
    l1, _ = model(idx)
    l2, _ = model(idx2)
    assert torch.allclose(l1[:, :7], l2[:, :7], atol=1e-5)       # before the change: identical
    assert not torch.allclose(l1[:, 7:], l2[:, 7:], atol=1e-5)   # after the change: different
