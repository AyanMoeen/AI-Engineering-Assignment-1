"""Lesson 06: BERT-style (bidirectional) masking vs GPT-style (causal) masking.

    python -m experiments.mlm_vs_clm
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.attention import MultiHeadAttention, causal_mask
from src.utils import ensure_dir
from experiments._common import FIG_DIR, annotated_heatmap


def main():
    ensure_dir(FIG_DIR); ensure_dir("outputs/logs")
    toks = ["The", "cat", "[M]", "on", "mat"]
    T = len(toks)
    bi = np.ones((T, T), dtype=int)
    cl = causal_mask(T).int().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    annotated_heatmap(ax[0], bi, "BERT / MLM: every token sees all tokens", toks, toks, "Greens", "{:.0f}")
    annotated_heatmap(ax[1], cl, "GPT / CLM: token sees only itself + the past", toks, toks, "Blues", "{:.0f}")
    for a in ax:
        a.set_xlabel("key (attended to)"); a.set_ylabel("query")
    fig.suptitle("Bidirectional (MLM) vs causal (CLM) attention masks")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/mlm_vs_clm_masks.png", dpi=150); plt.close(fig)

    # tiny experiment: same weights, change only the LAST token; does position 0 notice?
    torch.manual_seed(0)
    bi_attn = MultiHeadAttention(16, 2, causal=False).eval()
    cl_attn = MultiHeadAttention(16, 2, causal=True).eval()
    cl_attn.load_state_dict(bi_attn.state_dict())
    x = torch.randn(1, 6, 16)
    x2 = x.clone(); x2[:, -1] = torch.randn(16)
    d_bi = (bi_attn(x)[:, 0] - bi_attn(x2)[:, 0]).abs().max().item()
    d_cl = (cl_attn(x)[:, 0] - cl_attn(x2)[:, 0]).abs().max().item()
    text = (
        "Comparison experiment: change ONLY the last token, look at the output of position 0\n"
        f"  bidirectional (BERT-style) max change at position 0: {d_bi:.6f}   <- position 0 'sees the future'\n"
        f"  causal        (GPT-style)  max change at position 0: {d_cl:.6f}   <- future is blocked\n\n"
        "Why it matters: BERT predicts a masked word using BOTH sides, so it needs full attention.\n"
        "GPT is trained to predict the NEXT token; if it could see the future, it would just copy\n"
        "the answer, so the causal mask is required for training and for generation."
    )
    print(text)
    open("outputs/logs/mlm_vs_clm.txt", "w").write(text)


if __name__ == "__main__":
    main()
