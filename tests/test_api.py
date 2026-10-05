import os
from dataclasses import asdict

import torch
from fastapi.testclient import TestClient

from src.gpt import GPT, GPTConfig


def test_checkpoint_reload_and_api(tmp_path, monkeypatch):
    """A fresh process can load saved weights, and the API can generate text."""
    chars = list("abcdefgh \n")
    cfg = GPTConfig(vocab_size=len(chars), context_length=16, d_model=32, n_heads=2, n_layers=1, dropout=0.0)
    model = GPT(cfg)
    path = tmp_path / "tiny.pt"
    torch.save({"model_state": model.state_dict(), "model_config": asdict(cfg), "chars": chars}, path)

    monkeypatch.setenv("CHECKPOINT_PATH", str(path))
    from src.api import app
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/generate", json={"prompt": "abc", "max_new_tokens": 10})
        assert r.status_code == 200 and r.json()["generated"].startswith("abc")
        assert len(r.json()["generated"]) == 13
