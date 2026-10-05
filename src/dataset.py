"""Data pipeline: download text, tokenize, split, and build (x, y) batches.

Next-token setup: for a window of tokens t[i : i+T]
    x = t[i     : i+T]      (input)
    y = t[i + 1 : i+T+1]    (target = input shifted left by one)
so the target at position k is the token that follows x[k].
"""
import torch

from .tokenizer import CharTokenizer
from .utils import download_file


class LMData:
    def __init__(self, train_ids, val_ids):
        self.train = train_ids
        self.val = val_ids

    def get_batch(self, split, batch_size, block_size, device):
        data = self.train if split == "train" else self.val
        # i + block_size + 1 <= len(data)  ->  i < len(data) - block_size
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


def split_ids(ids, val_fraction=0.1):
    """Contiguous split: first part = train, last part = validation."""
    n = int(len(ids) * (1 - val_fraction))
    return ids[:n], ids[n:]


def prepare_data(data_cfg, val_fraction=0.1):
    path = download_file(data_cfg["url"], data_cfg["path"])
    text = open(path, "r", encoding="utf-8").read()
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, val_ids = split_ids(ids, val_fraction)
    return tok, LMData(train_ids, val_ids)
