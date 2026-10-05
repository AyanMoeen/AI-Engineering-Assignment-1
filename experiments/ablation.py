"""Figure 9: controlled ablation experiments (change ONE factor, keep the rest fixed).

    python -m experiments.ablation --experiments depth heads context --steps 2000

Experiments: depth (2/4/6 layers), heads (2/4/8), context (64/128/256), position (sinusoidal/learned).
Same seed and same number of steps for every run. Results: outputs/logs/ablation_results.csv + ablation_table.md
"""
import argparse
import copy
import csv

import matplotlib.pyplot as plt

from src.generate import generate_text, load_model
from src.train import train
from src.utils import ensure_dir, load_config
from experiments._common import FIG_DIR

GRID = {
    "depth": ("model", "n_layers", [2, 4, 6]),
    "heads": ("model", "n_heads", [2, 4, 8]),
    "context": ("model", "context_length", [64, 128, 256]),
    "position": ("model", "pos_type", ["sinusoidal", "learned"]),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--experiments", nargs="+", default=["depth", "heads", "context"], choices=list(GRID))
    p.add_argument("--steps", type=int, default=2000)
    a = p.parse_args()

    base = load_config(a.config)
    base["train"].update(max_steps=a.steps, eval_interval=max(a.steps // 8, 1), sample_interval=0, eval_iters=50)
    ensure_dir("outputs/logs"); ensure_dir(FIG_DIR); ensure_dir("checkpoints/ablation")
    rows, samples = [], []
    for exp in a.experiments:
        section, key, values = GRID[exp]
        for v in values:
            cfg = copy.deepcopy(base)
            cfg[section][key] = v
            name = f"abl_{exp}_{v}"
            ck = f"checkpoints/ablation/{name}.pt"
            print(f"\n>>> {exp} = {v}")
            r = train(cfg, run_name=name, ckpt_path=ck, verbose=False, make_plots=False)
            model, tok = load_model(ck)
            samples.append(f"=== {exp} = {v} ===\n" + generate_text(model, tok, "ROMEO:", 250, 0.8, 40) + "\n")
            rows.append({"experiment": exp, "value": v, "params": r["params"], "train_loss": round(r["train_loss"], 4),
                         "val_loss": round(r["val_loss"], 4), "perplexity": round(r["perplexity"], 2),
                         "train_time_s": round(r["train_time_s"], 1), "tokens_per_s": round(r["tokens_per_s"]),
                         "peak_mem_mb": None if r["peak_mem_mb"] is None else round(r["peak_mem_mb"])})
            print({k: v for k, v in rows[-1].items()})

    with open("outputs/logs/ablation_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["| experiment | value | params | train loss | val loss | perplexity | time (s) | tokens/s | peak mem (MB) |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['experiment']} | {r['value']} | {r['params']:,} | {r['train_loss']} | {r['val_loss']} | "
           f"{r['perplexity']} | {r['train_time_s']} | {r['tokens_per_s']} | {r['peak_mem_mb']} |" for r in rows]
    open("outputs/logs/ablation_table.md", "w").write("\n".join(md))
    open("outputs/samples/ablation_samples.txt", "w").write("\n".join(samples))
    print("\n" + "\n".join(md))

    fig, axes = plt.subplots(1, len(a.experiments), figsize=(4.5 * len(a.experiments), 4), squeeze=False)
    for ax, exp in zip(axes[0], a.experiments):
        rr = [r for r in rows if r["experiment"] == exp]
        ax.bar([str(r["value"]) for r in rr], [r["val_loss"] for r in rr], color="steelblue")
        for i, r in enumerate(rr):
            ax.text(i, r["val_loss"], f"{r['val_loss']:.3f}", ha="center", va="bottom", fontsize=9)
        lo = min(r["val_loss"] for r in rr); hi = max(r["val_loss"] for r in rr)
        ax.set_ylim(lo - 0.3 * (hi - lo + 0.05), hi + 0.3 * (hi - lo + 0.05))
        ax.set_title(f"{exp}"); ax.set_ylabel("validation loss")
    fig.suptitle(f"Figure 9 - ablations ({a.steps} steps each)")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig9_ablation.png", dpi=150)
    print(f"saved {FIG_DIR}/fig9_ablation.png")


if __name__ == "__main__":
    main()
