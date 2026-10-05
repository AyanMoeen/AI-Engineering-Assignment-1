# Dataset

**Tiny Shakespeare** (about 1.1 MB, about 40,000 lines of Shakespeare plays).

- Hugging Face: https://huggingface.co/datasets/karpathy/tiny_shakespeare
- Direct file: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

The training script downloads it automatically to `data/input.txt`. To download by hand:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/input.txt
```

Split: first 90% of the characters = train, last 10% = validation (contiguous, no shuffling).
The text file is not committed to Git (see `.gitignore`).
