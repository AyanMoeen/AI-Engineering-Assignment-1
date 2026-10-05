# MiniGPT: a small GPT built from scratch

A GPT-style causal language model (character-level, trained on Tiny Shakespeare) with every Transformer
component written by hand in PyTorch. **No pretrained models and no `transformers` library.**

## Where things live

| Part | File |
|---|---|
| Tokenizer (char-level, 65 tokens, no special tokens) | `src/tokenizer.py` |
| Data pipeline (download, split, shifted targets) | `src/dataset.py` |
| Self-attention, multi-head attention, causal mask | `src/attention.py` |
| Sinusoidal (and learned) positional encoding | `src/positional_encoding.py` |
| Transformer block (LayerNorm, attention, FFN, residuals) | `src/transformer_block.py` |
| MiniGPT model + text generation | `src/gpt.py` |
| Training loop / inference CLI / REST API | `src/train.py`, `src/generate.py`, `src/api.py` |
| Tests | `tests/` |
| Experiments and figures | `experiments/` |
| Config | `configs/tiny_gpt.yaml` |
| Colab notebook | `notebooks/colab_demo.ipynb` |
| Report template | `report/REPORT_TEMPLATE.md` |

## Setup

```bash
git clone <YOUR_REPO_URL> && cd transformer-gpt-assignment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Commands

```bash
# 1. unit tests (attention, causal mask, shapes, overfit, API)
python -m pytest -q

# 2. evidence figures 1-5 + BERT-vs-GPT + sequential-vs-parallel
python -m experiments.hand_calculation           # Fig 1, 2
python -m experiments.architecture_evidence      # Fig 3, 4, 5
python -m experiments.mlm_vs_clm
python -m experiments.seq_vs_parallel

# 3. tiny overfit gate (loss must fall close to 0)
python -m src.train --overfit

# 4. full training (GPU recommended; data is downloaded automatically)
python -m src.train --config configs/tiny_gpt.yaml

# 5. inference
python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
python -m src.generate --checkpoint-url <CHECKPOINT_LINK> --prompt "ROMEO:"    # download checkpoint first

# 6. evaluation, attention plot, ablations
python -m experiments.evaluate
python -m experiments.attention_visualization --layer -1
python -m experiments.ablation --experiments depth heads context --steps 2000

# 7. REST API
uvicorn src.api:app --port 8000
curl -X POST localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt":"ROMEO:","max_new_tokens":100}'
```

Outputs go to `outputs/figures`, `outputs/logs`, `outputs/samples`. Checkpoints go to `checkpoints/`.

## Expected output (numbers will differ)

```
Prompt: ROMEO:
Generated: ROMEO: ...
model parameters: 3.19M
context length: 128
temperature: 0.8
```

## Default model

`d_model=256, n_heads=4, n_layers=4, ffn_dim=4*d_model, context=128, dropout=0.1`, AdamW (lr 3e-4, warm-up + cosine decay),
batch 64, 5000 steps. About 3.19M parameters. Change anything in `configs/tiny_gpt.yaml` or with CLI flags
(`--n-layers`, `--n-heads`, `--context-length`, `--d-model`, `--pos-type`, `--max-steps`).

## Deploy (Render)

1. Upload `checkpoints/tiny_gpt.pt` as a GitHub Release asset (or a public link) and copy its direct download URL.
2. On Render: New + -> Blueprint -> pick this repo (`render.yaml` is included).
3. Set the env var `CHECKPOINT_URL` to that link. Open `/docs` to try the API, `/health` to check the model loaded.

The free tier sleeps when idle, so the first request can take about 1 minute.

## Known limitations

- Character-level model: it learns spelling and play format, but not long-range meaning. Text looks like Shakespeare but is not coherent.
- Only about 1 MB of training text, so the model overfits if trained too long or made too large.
- Context is 128 characters, so it forgets anything earlier.
- Generated text is an experiment, not factual.
- Speed tests on CPU can show attention slower than an LSTM at long sequences (T^2 cost); the parallel advantage shows on GPU.

## Sources

Tiny Shakespeare dataset (Karpathy, char-rnn). Architecture from "Attention Is All You Need" (Vaswani et al., 2017) and GPT-2 style
pre-LayerNorm blocks. Add any tutorials or code you consulted here.
