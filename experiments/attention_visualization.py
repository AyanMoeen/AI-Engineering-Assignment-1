"""Figure 7: attention heatmaps from a TRAINED model.

    python -m experiments.attention_visualization --checkpoint checkpoints/tiny_gpt.pt --layer -1
"""
import argparse

import matplotlib.pyplot as plt
import torch

from src.generate import load_model
from src.utils import ensure_dir, get_device
from experiments._common import FIG_DIR


def pretty(c):
    return {"\n": "\\n", " ": "_"}.get(c, c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    p.add_argument("--prompt", default="ROMEO:\nO, she doth teach the torches")
    p.add_argument("--layer", type=int, default=-1, help="which Transformer block (-1 = last)")
    a = p.parse_args()

    ensure_dir(FIG_DIR)
    device = get_device()
    model, tok = load_model(a.checkpoint, device)
    ids = tok.encode(a.prompt)[: model.cfg.context_length]
    idx = torch.tensor([ids], device=device)
    with torch.no_grad():
        _, _, attns = model(idx, return_attn=True)        # list of (1, H, T, T)
    w = attns[a.layer][0].cpu()
    labels = [pretty(c) for c in tok.decode(ids)]
    H = w.size(0)
    cols = min(H, 4); rows = (H + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for h in range(rows * cols):
        ax = axes[h // cols][h % cols]
        if h >= H:
            ax.axis("off"); continue
        ax.imshow(w[h], cmap="viridis")
        ax.set_title(f"head {h}", fontsize=10)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("key (attended to)", fontsize=8); ax.set_ylabel("query", fontsize=8)
    layer_no = a.layer % len(attns)
    fig.suptitle(f"Figure 7 - trained model attention, layer {layer_no} (bright = high weight; upper triangle = 0)")
    fig.tight_layout()
    out = f"{FIG_DIR}/fig7_attention_layer{layer_no}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
