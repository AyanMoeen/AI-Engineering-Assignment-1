"""Figure 1 + Figure 2: hand-calculated attention for a 3-token sequence (d_k = d_v = 2).

    python -m experiments.hand_calculation

Copy the printed numbers into your report (they can be checked with a calculator).
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.attention import scaled_dot_product_attention, causal_mask
from src.utils import ensure_dir
from experiments._common import FIG_DIR, annotated_heatmap

np.set_printoptions(precision=4, suppress=True)


def softmax_rows(m):
    e = np.exp(m - m.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    ensure_dir(FIG_DIR)
    Q = np.array([[1., 0.], [0., 1.], [1., 1.]])
    K = np.array([[1., 0.], [0., 1.], [1., 1.]])
    V = np.array([[1., 2.], [3., 4.], [5., 6.]])
    d_k = 2

    print("Q =\n", Q, "\nK =\n", K, "\nV =\n", V)
    scores = Q @ K.T
    print("\nStep 2: Q K^T =\n", scores)
    scaled = scores / math.sqrt(d_k)
    print(f"\nStep 3: divide by sqrt(d_k) = sqrt({d_k}) = {math.sqrt(d_k):.4f}\n", scaled)
    W = softmax_rows(scaled)
    print("\nStep 4: row-wise softmax (attention weights; each row sums to 1)\n", W, "\nrow sums:", W.sum(1))
    out = W @ V
    print("\nStep 5: output = weights @ V\n", out)

    mask = np.tril(np.ones((3, 3), dtype=bool))
    masked = np.where(mask, scaled, -np.inf)
    Wc = softmax_rows(masked)
    outc = Wc @ V
    print("\nStep 6: causal mask (True = allowed)\n", mask.astype(int))
    print("masked scores (future = -inf):\n", masked)
    print("causal attention weights:\n", Wc)
    print("causal output:\n", outc)
    print("\nStep 7 (in words):")
    print("  token 1 can only see itself -> its output is exactly V1")
    print("  token 2 sees tokens 1-2; token 3 sees all three tokens (nothing changes for it)")

    # check the manual numbers against the real implementation
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    o, w = scaled_dot_product_attention(t(Q), t(K), t(V))
    oc, wc = scaled_dot_product_attention(t(Q), t(K), t(V), causal_mask(3))
    assert np.allclose(o.numpy(), out, atol=1e-5) and np.allclose(wc.numpy(), Wc, atol=1e-5)
    print("\nOK: manual numbers match src/attention.py")

    toks = ["tok1", "tok2", "tok3"]
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    annotated_heatmap(ax[0, 0], Q, "Q (3x2)", ylabels=toks, xlabels=["d1", "d2"], cmap="Greens")
    annotated_heatmap(ax[0, 1], K, "K (3x2)", ylabels=toks, xlabels=["d1", "d2"], cmap="Greens")
    annotated_heatmap(ax[0, 2], V, "V (3x2)", ylabels=toks, xlabels=["d1", "d2"], cmap="Greens")
    annotated_heatmap(ax[0, 3], scores, "Q K^T", xlabels=toks, ylabels=toks, cmap="Oranges")
    annotated_heatmap(ax[1, 0], scaled, "scores / sqrt(d_k)", xlabels=toks, ylabels=toks, cmap="Oranges")
    annotated_heatmap(ax[1, 1], W, "softmax -> weights", xlabels=toks, ylabels=toks, cmap="Blues")
    annotated_heatmap(ax[1, 2], out, "output = weights @ V", xlabels=["d1", "d2"], ylabels=toks, cmap="Purples")
    ax[1, 3].axis("off")
    ax[1, 3].text(0, 0.5, "Steps:\n1. Q, K, V given\n2. QK^T\n3. / sqrt(d_k)\n4. softmax per row\n5. weights @ V",
                  fontsize=11, va="center")
    fig.suptitle("Figure 1 - Hand-calculated scaled dot-product attention")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig1_hand_attention.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    annotated_heatmap(ax[0], mask.astype(int), "causal mask (1 = allowed)", toks, toks, cmap="Greys", fmt="{:.0f}")
    annotated_heatmap(ax[1], masked, "masked scores (future = -inf)", toks, toks, cmap="Oranges")
    annotated_heatmap(ax[2], Wc, "causal attention weights", toks, toks, cmap="Blues")
    for a in ax:
        a.set_xlabel("key (attended to)"); a.set_ylabel("query (current token)")
    fig.suptitle("Figure 2 - Causal mask: future tokens get weight 0")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig2_causal_mask.png", dpi=150); plt.close(fig)
    print(f"saved {FIG_DIR}/fig1_hand_attention.png and {FIG_DIR}/fig2_causal_mask.png")


if __name__ == "__main__":
    main()
