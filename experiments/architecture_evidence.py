"""Figures 3, 4, 5: multi-head shape trace, positional-encoding plot, model summary.

    python -m experiments.architecture_evidence [--config configs/tiny_gpt.yaml] [--checkpoint checkpoints/tiny_gpt.pt]
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch

from src.attention import MultiHeadAttention
from src.gpt import GPT, GPTConfig
from src.positional_encoding import sinusoidal_table
from src.utils import count_parameters, ensure_dir, load_config
from experiments._common import FIG_DIR, text_figure


def fig3_shapes():
    B, T, D, H = 2, 6, 32, 4
    mha = MultiHeadAttention(D, H)
    lines = [f"MultiHeadAttention(d_model={D}, n_heads={H})  ->  d_k = d_model / n_heads = {D // H}",
             f"batch B={B}, sequence length T={T}", ""]
    for name, shape in mha.trace_shapes(torch.randn(B, T, D)):
        lines.append(f"{name:<34s} {shape}")
    text = "\n".join(lines)
    print(text)
    text_figure(text, f"{FIG_DIR}/fig3_multihead_shapes.png", "Figure 3 - multi-head tensor-shape trace")
    open("outputs/logs/multihead_shapes.txt", "w").write(text)


def fig4_positional():
    pe = sinusoidal_table(128, 64).numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    im = ax[0].imshow(pe, aspect="auto", cmap="RdBu")
    ax[0].set_xlabel("embedding dimension"); ax[0].set_ylabel("position"); ax[0].set_title("PE matrix (128 positions x 64 dims)")
    fig.colorbar(im, ax=ax[0])
    for d in (0, 2, 8, 20):
        ax[1].plot(pe[:, d], label=f"dim {d}")
    ax[1].set_xlabel("position"); ax[1].set_title("low dims oscillate fast, high dims slowly"); ax[1].legend()
    fig.suptitle("Figure 4 - Sinusoidal positional encoding")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig4_positional_encoding.png", dpi=150); plt.close(fig)
    print("saved fig4_positional_encoding.png")


def fig5_summary(cfg_path, ckpt_path):
    cfg = load_config(cfg_path)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        gcfg = GPTConfig(**ck["model_config"]); source = f"checkpoint {ckpt_path}"
    else:
        gcfg = GPTConfig(vocab_size=65, **cfg["model"]); source = f"{cfg_path} (vocab_size=65 for Tiny Shakespeare)"
    model = GPT(gcfg)
    lines = [f"MiniGPT summary - from {source}", f"config: {gcfg}", "", f"{'module':<28s}{'parameters':>12s}"]
    for name, child in model.named_children():
        lines.append(f"{name:<28s}{count_parameters(child):>12,}")
    lines += ["", f"{'TOTAL':<28s}{count_parameters(model):>12,}  ({count_parameters(model) / 1e6:.2f}M)", "", str(model)]
    text = "\n".join(lines)
    print(text)
    text_figure(text, f"{FIG_DIR}/fig5_model_summary.png", "Figure 5 - model summary / parameter count")
    open("outputs/logs/model_summary.txt", "w").write(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    a = p.parse_args()
    ensure_dir(FIG_DIR); ensure_dir("outputs/logs")
    fig3_shapes(); fig4_positional(); fig5_summary(a.config, a.checkpoint)


if __name__ == "__main__":
    main()
