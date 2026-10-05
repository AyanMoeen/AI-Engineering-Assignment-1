"""Inference CLI.

    python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
"""
import argparse

import torch

from .gpt import GPT, GPTConfig
from .tokenizer import CharTokenizer
from .utils import count_parameters, download_file, get_device


def load_model(path, device=None):
    """Load a checkpoint saved by src.train -> (model, tokenizer)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharTokenizer(ckpt["chars"])


def generate_text(model, tok, prompt, max_new_tokens=200, temperature=0.8, top_k=40, device=None):
    device = device or next(model.parameters()).device
    ids = tok.encode(prompt) or tok.encode("\n")         # empty prompt -> newline
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    p.add_argument("--checkpoint-url", default=None, help="download the checkpoint from this URL if missing")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.checkpoint_url:
        download_file(args.checkpoint_url, args.checkpoint)
    device = get_device()
    model, tok = load_model(args.checkpoint, device)
    text = generate_text(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k, device)

    print(f"Prompt: {args.prompt}")
    print(f"Generated: {text}")
    print(f"model parameters: {count_parameters(model) / 1e6:.2f}M")
    print(f"context length: {model.cfg.context_length}")
    print(f"temperature: {args.temperature}")


if __name__ == "__main__":
    main()
