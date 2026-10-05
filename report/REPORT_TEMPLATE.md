# Medium article template: Building a small GPT from scratch

> Replace every `[...]`. Put the figure images in the article. Keep "observation" (what you measured) and "interpretation" (why) separate.

## 1. Problem and scenario
[2-3 sentences: can a small Transformer trained from scratch learn next-token structure of a domain corpus?]

## 2. Transformer foundations
- Why Transformers: RNN = T sequential steps, attention = O(1) sequential depth but O(T^2) memory. Add result of `experiments.seq_vs_parallel` (outputs/logs/seq_vs_parallel.txt).
- Equation: `Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V`
- **Hand calculation** (paste output of `python -m experiments.hand_calculation`) + **Figure 1**, **Figure 2** (causal mask). Explain in words what each token can see.
- Multi-head attention: **Figure 3** (shape trace). Positional encoding: **Figure 4** + explanation.
- Transformer block diagram (draw: LN -> MHA -> +residual -> LN -> FFN -> +residual).
- BERT vs GPT: `outputs/figures/mlm_vs_clm_masks.png` + `outputs/logs/mlm_vs_clm.txt`.

## 3. Implementation
[Tokenizer (char-level, vocab 65, no special tokens), attention, masking, training loop. Mention file names.]

## 4. Dataset and preprocessing
[Tiny Shakespeare, about 1.1 MB, 90/10 contiguous split, vocab size, context length.]

## 5. Training setup
[Hardware (Colab GPU name), hyperparameters, optimizer, batch size, steps, wall-clock time, checkpoint.]

## 6. Results
- **Figure 5** model summary / parameter count: [..M]
- **Figure 6** training vs validation loss curve: final train [..], final val [..], perplexity [..]
- **Figure 7** attention visualization + what you observe (e.g. heads looking at the previous character, at spaces, at the start of a word)
- **Figure 8** at least 5 generated samples (`outputs/samples/final_samples.txt`)

## 7. Ablation study (**Figure 9**)
[Paste `outputs/logs/ablation_table.md`.]
| Observation | Interpretation |
|---|---|
| Val loss went from X to Y | [why, supported by evidence] |

## 8. Failure analysis (at least 2)
1. [e.g. invented words / repetition loops with greedy decoding] Cause: [...]
2. [e.g. forgets speaker/context beyond 128 chars] Cause: [...]

## 9. Engineering packaging
[CLI command, API endpoint, Render link, how someone else runs inference (Figure 10: screenshot of CLI or API call).]

## 10. Conclusion
[What the experiments show about Transformer-based causal language modeling.]

## References
[Every source, tutorial, repo you used.]
