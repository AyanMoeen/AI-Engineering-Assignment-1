import torch

from src.dataset import LMData, split_ids
from src.tokenizer import CharTokenizer


def test_tokenizer_roundtrip_and_determinism():
    text = "hello world"
    a, b = CharTokenizer.from_text(text), CharTokenizer.from_text(text)
    assert a.chars == b.chars == sorted(set(text))
    assert a.decode(a.encode(text)) == text
    assert a.encode("hzx") == a.encode("h")               # unknown chars are dropped


def test_targets_are_inputs_shifted_by_one():
    ids = torch.arange(100)
    data = LMData(ids, ids)
    x, y = data.get_batch("train", batch_size=16, block_size=10, device="cpu")
    assert x.shape == y.shape == (16, 10)
    assert torch.equal(y[:, :-1], x[:, 1:])               # y[t] is the token after x[t]
    assert torch.all(y == x + 1)                          # for ids = arange


def test_split_is_contiguous():
    tr, va = split_ids(torch.arange(100), 0.1)
    assert len(tr) == 90 and len(va) == 10 and va[0] == 90
