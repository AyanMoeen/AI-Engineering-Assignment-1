"""Small helpers: config, seed, device, parameter count, download."""
import os
import random
import urllib.request
from pathlib import Path

import torch
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def download_file(url, dest):
    """Download url -> dest (skips if the file already exists)."""
    dest = Path(dest)
    if dest.exists():
        return dest
    ensure_dir(dest.parent)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest
