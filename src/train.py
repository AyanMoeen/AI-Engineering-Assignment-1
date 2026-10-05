"""Training script.

    python -m src.train --config configs/tiny_gpt.yaml          # full training
    python -m src.train --overfit                                # tiny-batch debugging gate
    python -m src.train --n-layers 2 --run-name depth2           # override a config value
"""
import argparse
import copy
import csv
import json
import math
import time
from dataclasses import asdict

import torch

from .dataset import prepare_data
from .generate import generate_text
from .gpt import GPT, GPTConfig
from .utils import count_parameters, ensure_dir, get_device, load_config, set_seed


def get_lr(step, t):
    """Linear warm-up, then cosine decay to min_lr."""
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    if step >= t["max_steps"]:
        return t["min_lr"]
    ratio = (step - t["warmup_steps"]) / max(1, t["max_steps"] - t["warmup_steps"])
    return t["min_lr"] + 0.5 * (1 + math.cos(math.pi * ratio)) * (t["lr"] - t["min_lr"])


@torch.no_grad()
def estimate_loss(model, data, block, t, device):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(t["eval_iters"])
        for i in range(t["eval_iters"]):
            x, y = data.get_batch(split, t["batch_size"], block, device)
            losses[i] = model(x, y)[1].item()
        out[split] = losses.mean().item()
    model.train()
    return out


def make_optimizer(model, t):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": t["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t["lr"])


def save_plots(history, run_name, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [h["step"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, [h["train_loss"] for h in history], label="train loss")
    plt.plot(steps, [h["val_loss"] for h in history], label="validation loss")
    plt.xlabel("step"); plt.ylabel("cross-entropy loss"); plt.title(f"Loss curve ({run_name})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{fig_dir}/loss_{run_name}.png", dpi=150)
    plt.close()


def train(cfg, run_name="tiny_gpt", overfit=False, ckpt_path=None, verbose=True,
          log_dir="outputs/logs", fig_dir="outputs/figures", sample_dir="outputs/samples", make_plots=True):
    """Train a model. Returns a dict of metrics (used by experiments/ablation.py)."""
    t, m = copy.deepcopy(cfg["train"]), copy.deepcopy(cfg["model"])
    if overfit:
        m["dropout"] = 0.0
        t["max_steps"] = t.get("overfit_steps", 300)
        t["warmup_steps"] = 10
        t["sample_interval"] = 0
    set_seed(t["seed"])
    device = get_device()
    tok, data = prepare_data(cfg["data"], t["val_fraction"])
    gcfg = GPTConfig(vocab_size=tok.vocab_size, **m)
    model = GPT(gcfg).to(device)
    n_params = count_parameters(model)
    opt = make_optimizer(model, t)
    block, bs = gcfg.context_length, t["batch_size"]

    if verbose:
        print(f"run: {run_name} | device: {device} | params: {n_params:,} | vocab: {tok.vocab_size}")
        print(f"model config: {asdict(gcfg)}")

    fixed = data.get_batch("train", 8, block, device) if overfit else None   # one tiny repeated batch
    history, samples = [], []
    ensure_dir(log_dir); ensure_dir(fig_dir); ensure_dir(sample_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    first_loss = last_loss = None

    model.train()
    for step in range(t["max_steps"]):
        lr = get_lr(step, t)
        for g in opt.param_groups:
            g["lr"] = lr

        if not overfit and (step % t["eval_interval"] == 0 or step == t["max_steps"] - 1):
            losses = estimate_loss(model, data, block, t, device)
            history.append({"step": step, "train_loss": losses["train"], "val_loss": losses["val"],
                            "lr": lr, "elapsed_s": time.time() - start})
            if verbose:
                print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e}")
        if t.get("sample_interval") and step % t["sample_interval"] == 0:
            s = generate_text(model, tok, "\n", 200, 0.8, 40, device)
            model.train()
            samples.append((step, s))
            if verbose:
                print(f"--- sample @ step {step} ---\n{s}\n---------------------------")

        x, y = fixed if overfit else data.get_batch("train", bs, block, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss
        if overfit and verbose and (step % 25 == 0 or step == t["max_steps"] - 1):
            print(f"[overfit] step {step:4d} | loss {last_loss:.4f}")

    elapsed = time.time() - start
    result = {"run": run_name, "params": n_params, "device": str(device), "train_time_s": elapsed,
              "tokens_per_s": t["max_steps"] * bs * block / elapsed,
              "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None}
    if overfit:
        result.update(initial_loss=first_loss, final_loss=last_loss)
        if verbose:
            print(f"overfit test: loss {first_loss:.4f} -> {last_loss:.4f}")
        return result

    final = estimate_loss(model, data, block, t, device)
    result.update(train_loss=final["train"], val_loss=final["val"], perplexity=math.exp(final["val"]),
                  history=history)
    if verbose:
        print(f"FINAL | train {final['train']:.4f} | val {final['val']:.4f} | ppl {result['perplexity']:.2f} "
              f"| time {elapsed:.0f}s")

    with open(f"{log_dir}/{run_name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr", "elapsed_s"])
        w.writeheader(); w.writerows(history)
    with open(f"{sample_dir}/{run_name}_training_samples.txt", "w") as f:
        for step, s in samples:
            f.write(f"=== step {step} ===\n{s}\n\n")
    with open(f"{log_dir}/{run_name}_summary.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"}, f, indent=2)
    if make_plots:
        save_plots(history, run_name, fig_dir)
    if ckpt_path:
        ensure_dir(str(ckpt_path).rsplit("/", 1)[0] if "/" in str(ckpt_path) else ".")
        torch.save({"model_state": model.state_dict(), "model_config": asdict(gcfg),
                    "chars": tok.chars, "train_config": t,
                    "metrics": {k: v for k, v in result.items() if k != "history"}}, ckpt_path)
        if verbose:
            print(f"checkpoint saved -> {ckpt_path}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--run-name", default="tiny_gpt")
    p.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    p.add_argument("--overfit", action="store_true", help="tiny repeated batch: debugging gate")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--n-layers", type=int)
    p.add_argument("--n-heads", type=int)
    p.add_argument("--d-model", type=int)
    p.add_argument("--context-length", type=int)
    p.add_argument("--pos-type", choices=["sinusoidal", "learned"])
    args = p.parse_args()

    cfg = load_config(args.config)
    for arg, section, key in [("max_steps", "train", "max_steps"), ("batch_size", "train", "batch_size"),
                              ("lr", "train", "lr"), ("n_layers", "model", "n_layers"),
                              ("n_heads", "model", "n_heads"), ("d_model", "model", "d_model"),
                              ("context_length", "model", "context_length"), ("pos_type", "model", "pos_type")]:
        if getattr(args, arg) is not None:
            cfg[section][key] = getattr(args, arg)
    train(cfg, run_name=args.run_name, overfit=args.overfit,
          ckpt_path=None if args.overfit else args.checkpoint)


if __name__ == "__main__":
    main()
