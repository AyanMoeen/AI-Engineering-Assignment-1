"""FastAPI service.

    uvicorn src.api:app --host 0.0.0.0 --port 8000
    curl -X POST localhost:8000/generate -H "Content-Type: application/json" \
         -d '{"prompt": "ROMEO:", "max_new_tokens": 100}'

Environment variables:
    CHECKPOINT_PATH  (default checkpoints/tiny_gpt.pt)
    CHECKPOINT_URL   (optional) downloaded at startup if the file is missing
"""
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generate import generate_text, load_model
from .utils import count_parameters, download_file

STATE = {}


@asynccontextmanager
async def lifespan(app):
    path = os.environ.get("CHECKPOINT_PATH", "checkpoints/tiny_gpt.pt")
    url = os.environ.get("CHECKPOINT_URL")
    try:
        if url and not os.path.exists(path):
            download_file(url, path)
        STATE["model"], STATE["tok"] = load_model(path, torch.device("cpu"))
        print(f"model loaded from {path}")
    except Exception as e:  # keep the server up so /health explains the problem
        STATE["error"] = str(e)
        print(f"could not load model: {e}")
    yield
    STATE.clear()


app = FastAPI(title="MiniGPT API", version="1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field("ROMEO:", max_length=1000)
    max_new_tokens: int = Field(200, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=1)


@app.get("/")
def root():
    return {"name": "MiniGPT API", "endpoints": ["/health", "/generate (POST)", "/docs"]}


@app.get("/health")
def health():
    if "model" not in STATE:
        return {"status": "model not loaded", "error": STATE.get("error")}
    m = STATE["model"]
    return {"status": "ok", "parameters": count_parameters(m), "context_length": m.cfg.context_length}


@app.post("/generate")
def generate(req: GenerateRequest):
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE.get('error')}")
    text = generate_text(STATE["model"], STATE["tok"], req.prompt, req.max_new_tokens,
                         req.temperature, req.top_k, torch.device("cpu"))
    return {"prompt": req.prompt, "generated": text, "temperature": req.temperature,
            "parameters": count_parameters(STATE["model"])}
