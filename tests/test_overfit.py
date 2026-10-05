import torch

from src.gpt import GPT, GPTConfig


def test_tiny_overfit_loss_drops_a_lot():
    """Debugging gate: gradients + target shifting work if the model can memorise one batch."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, context_length=16, d_model=32, n_heads=2, n_layers=2, dropout=0.0)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    seq = torch.randint(0, 20, (4, 17))
    x, y = seq[:, :-1], seq[:, 1:]                        # shifted targets
    first = None
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first
