# MiniGPT: a small GPT built from scratch

AI Engineering | Transformer Deep Dive assignment

## Team

| Name | Roll number |
|---|---|
| Ayan Bin Moeen | 23I-0124 |
| Mir Ahmed | 23I-0142 |

## Links

- **Live API:** https://ai-engineering-assignment-1-production.up.railway.app/docs
- **Report (Medium):** https://medium.com/@i230124/building-a-small-gpt-from-scratch-1807dab5e3d8
- **Slides:** [presentations and demo/MiniGPT_presentation.pdf](presentation/MiniGPT_presentation.pdf)
- **Demo video:** ADD-YOUR-VIDEO-LINK-HERE
- **Colab demo:** [Open in Colab](https://colab.research.google.com/github/AyanMoeen/AI-Engineering-Assignment-1/blob/main/notebooks/colab_demo.ipynb)
- **Checkpoint:** https://github.com/AyanMoeen/AI-Engineering-Assignment-1/releases/download/v1.0/tiny_gpt.pt

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
| API deployment (Railway) | `Dockerfile`, `requirements-api.txt` |
| Report (published copy used for Medium) | `docs/` |
| Presentation | `presentation/` |

## Setup

```bash
git clone https://github.com/AyanMoeen/AI-Engineering-Assignment-1.git && cd AI-Engineering-Assignment-1
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Commands

```bash
# 1. unit tests (attention, causal mask, shapes, overfit, pipeline, API)
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
python -m src.generate --checkpoint-url https://github.com/AyanMoeen/AI-Engineering-Assignment-1/releases/download/v1.0/tiny_gpt.pt --prompt "ROMEO:"

# 6. evaluation, attention plot, ablations
python -m experiments.evaluate
python -m experiments.attention_visualization --layer -1
python -m experiments.ablation --experiments depth heads context --steps 2000

# 7. REST API (local)
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
batch 64, 5000 steps. 3,192,832 parameters. Final result: train loss 1.2818, validation loss 1.5052, perplexity 4.51
(about 7.6 minutes on a Colab T4). Change anything in `configs/tiny_gpt.yaml` or with CLI flags
(`--n-layers`, `--n-heads`, `--context-length`, `--d-model`, `--pos-type`, `--max-steps`).

## Deployment (Railway)

The API is live at https://ai-engineering-assignment-1-production.up.railway.app/docs (health check: `/health`).

It is built from the `Dockerfile` in this repo, which installs CPU PyTorch and the API dependencies from
`requirements-api.txt`, then starts `uvicorn src.api:app`. The checkpoint is not stored in the repository: the
`CHECKPOINT_URL` environment variable (set in the Dockerfile, and overridable in Railway's Variables tab) points to the
GitHub release asset, and the API downloads it on startup.

To deploy your own copy:

1. On railway.com: New Project -> Deploy from GitHub repo -> select this repo. Railway detects the `Dockerfile`.
2. Open the service -> Settings -> Networking -> Generate Domain.
3. Open `/health` to confirm the model loaded (`"status": "ok"`), then `/docs` to try `POST /generate`.

Railway's free trial runs on limited credit, and idle services may be slow on the first request. `render.yaml` is also
included as an alternative for Render, but Render now asks for a payment card, even for free services.

## Known limitations

- Character-level model: it learns spelling and play format, but not long-range meaning. Text looks like Shakespeare but is not coherent.
- Only about 1 MB of training text, so the model overfits if trained too long or made too large (train 1.28 vs validation 1.51).
- Context is 128 characters, so it forgets anything earlier.
- Greedy decoding falls into repetition loops; temperature sampling with top-k avoids this.
- Generated text is an experiment, not factual.
- In our speed test, attention and an LSTM ran at similar speed at these small sizes; the O(1) sequential-depth advantage of attention is theoretical here, while its O(T^2) memory cost was measured.
- Ablations used one seed and 2,000 steps each, so small differences (for example between head counts) may be noise.

## Sources

- Tiny Shakespeare dataset: Andrej Karpathy, char-rnn (https://github.com/karpathy/char-rnn).
- "Attention Is All You Need" (Vaswani et al., 2017); GPT-2 style pre-LayerNorm blocks (Radford et al., 2019); BERT (Devlin et al., 2018) for the masked-language-model comparison.
- AI assistance was used for coding and writing support, as permitted by the assignment. Both authors can explain every component.
- Add any tutorials or code you consulted here.
