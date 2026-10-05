"""Quantitative + qualitative evaluation of a trained checkpoint.

    python -m experiments.evaluate --checkpoint checkpoints/tiny_gpt.pt

Outputs: validation loss / perplexity, 5 fixed-prompt samples (Figure 8), greedy vs temperature comparison.
"""
import argparse
import math

import torch

from src.dataset import prepare_data
from src.generate import generate_text, load_model
from src.utils import ensure_dir, get_device, load_config

PROMPTS = ["ROMEO:", "To be, or not to be", "First Citizen:\n", "KING HENRY:\nWhat", "JULIET:\nO Romeo,"]


def distinct_ngrams(text, n=4):
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


@torch.no_grad()
def split_loss(model, data, split, iters, batch_size, device):
    losses = []
    for _ in range(iters):
        x, y = data.get_batch(split, batch_size, model.cfg.context_length, device)
        losses.append(model(x, y)[1].item())
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=300)
    a = p.parse_args()

    ensure_dir("outputs/samples"); ensure_dir("outputs/logs")
    device = get_device()
    model, tok = load_model(a.checkpoint, device)
    cfg = load_config(a.config)
    _, data = prepare_data(cfg["data"], 0.1)
    torch.manual_seed(1234)
    tr, va = (split_loss(model, data, s, a.iters, 32, device) for s in ("train", "val"))
    lines = [f"checkpoint: {a.checkpoint}", f"train loss: {tr:.4f}", f"val loss:   {va:.4f}",
             f"val perplexity: {math.exp(va):.2f}", ""]

    lines.append(f"=== 5 samples (temperature={a.temperature}, top_k=40) ===")
    for i, prompt in enumerate(PROMPTS):
        torch.manual_seed(100 + i)                         # fixed seed -> reproducible samples
        text = generate_text(model, tok, prompt, a.max_new_tokens, a.temperature, 40, device)
        lines += [f"--- prompt {i + 1}: {prompt!r} ---", text, ""]

    lines.append("=== greedy vs temperature sampling (prompt 'ROMEO:') ===")
    for name, temp in [("greedy (T=0)", 0.0), ("T=0.8", 0.8), ("T=1.2", 1.2)]:
        torch.manual_seed(7)
        text = generate_text(model, tok, "ROMEO:", a.max_new_tokens, temp, None if temp == 0 else 40, device)
        lines += [f"[{name}] distinct 4-grams = {distinct_ngrams(text):.2f}", text, ""]
    out = "\n".join(lines)
    print(out)
    open("outputs/samples/final_samples.txt", "w").write(out)
    open("outputs/logs/evaluation.txt", "w").write(out.split("=== 5 samples")[0])


if __name__ == "__main__":
    main()
