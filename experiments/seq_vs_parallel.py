"""Lesson 01: sequential (RNN/LSTM) vs parallel (attention) computation - a measured comparison.

    python -m experiments.seq_vs_parallel
"""
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.utils import ensure_dir, get_device
from experiments._common import FIG_DIR


def timeit(fn, device, reps=5):
    fn()                                                   # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    ensure_dir(FIG_DIR); ensure_dir("outputs/logs")
    device = get_device()
    B, D, H = 16, 128, 4
    lstm = nn.LSTM(D, D, batch_first=True).to(device)
    mha = MultiHeadAttention(D, H).to(device)
    rows = ["T    | LSTM sec | Attention sec | LSTM sequential steps | Attention score memory (MB)"]
    Ts, t_l, t_a = [64, 128, 256, 512], [], []
    for T in Ts:
        x = torch.randn(B, T, D, device=device)
        def run_lstm():
            lstm(x)[0].sum().backward()
        def run_attn():
            mha(x).sum().backward()
        tl, ta = timeit(run_lstm, device), timeit(run_attn, device)
        t_l.append(tl); t_a.append(ta)
        mem = B * H * T * T * 4 / 2**20
        rows.append(f"{T:<4d} | {tl:8.4f} | {ta:13.4f} | {T:21d} | {mem:10.1f}")
    text = f"device: {device}  (batch={B}, d_model={D}, forward+backward)\n" + "\n".join(rows) + (
        "\n\nSequential depth: LSTM needs T steps one after another (O(T)); attention needs O(1) sequential\n"
        "steps because all positions are computed at once. Trade-off: attention stores a T x T score\n"
        "matrix per head, so memory grows as O(T^2).")
    print(text)
    open("outputs/logs/seq_vs_parallel.txt", "w").write(text)
    plt.figure(figsize=(6, 4))
    plt.plot(Ts, t_l, "o-", label="LSTM (sequential)"); plt.plot(Ts, t_a, "o-", label="Self-attention (parallel)")
    plt.xlabel("sequence length T"); plt.ylabel("seconds (fwd+bwd)"); plt.legend(); plt.grid(alpha=0.3)
    plt.title("Sequential vs parallel computation"); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/seq_vs_parallel.png", dpi=150)


if __name__ == "__main__":
    main()
