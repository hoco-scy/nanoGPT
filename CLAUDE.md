# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanoGPT is a minimalist GPT-2 training/finetuning framework by Andrej Karpathy. The core is just two files: `model.py` (model definition) and `train.py` (training loop). It reproduces GPT-2 (124M) on OpenWebText in ~4 days on 8xA100. **Deprecated** in favor of "nanochat" (Nov 2025).

## Common Commands

### Install dependencies
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### Prepare data
```bash
python data/shakespeare_char/prepare.py   # character-level Shakespeare (fast, good for debugging)
python data/shakespeare/prepare.py        # BPE Shakespeare
python data/openwebtext/prepare.py        # full OpenWebText (~17GB, requires HuggingFace download)
```

### Train
```bash
# Quick debug run (character-level Shakespeare, ~3 min on GPU)
python train.py config/train_shakespeare_char.py

# Full GPT-2 reproduction (8 GPUs, ~4 days)
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# Override config via CLI
python train.py config/train_shakespeare_char.py --batch_size=32 --compile=False --wandb=False
```

### Sample
```bash
python sample.py --out_dir=out-shakespeare
python sample.py --out_dir=out-shakespeare --start="ROMEO:" --temperature=0.8 --top_k=200
python sample.py --out_dir=out-shakespeare --start="FILE:prompt.txt"
```

### Evaluate
```bash
python train.py config/eval_gpt2.py            # GPT-2 base (val loss ~3.12)
python train.py config/eval_gpt2_medium.py      # GPT-2 medium (val loss ~2.84)
```

### Benchmark
```bash
python bench.py
```

## Architecture

### Two-file core

- **`model.py`** — GPT-2 transformer implementation. `GPTConfig` dataclass holds hyperparameters. `GPT` nn.Module builds token+positional embeddings, N transformer blocks (pre-norm residual), final LayerNorm, and tied `lm_head`. Uses Flash Attention via `scaled_dot_product_attention` when available. Includes `from_pretrained()` (loads HuggingFace GPT-2 weights, transposing Conv1D→Linear), `configure_optimizers()` (decay/no-decay param groups, fused AdamW on CUDA), and `estimate_mfu()` (PaLM formula, A100 baseline).

- **`train.py`** — Training loop with DDP, gradient accumulation, mixed precision (bf16/fp16), cosine LR schedule with linear warmup, and checkpointing. Data loaded via `np.memmap` on uint16 binary files. Auto-detects distributed env via `RANK`. Three init modes: `'scratch'`, `'resume'` (from checkpoint), `'gpt2*'` (from OpenAI weights).

### Configuration system (`configurator.py`)

Uses "Poor Man's Configurator" — each script defines globals with defaults, then `exec(open('configurator.py').read())` processes CLI args:
- Bare filenames (e.g., `config/train_gpt2.py`) are `exec()`'d to override globals
- `--key=value` args override individual settings with type checking

Config files in `config/` are plain Python scripts that assign to global variables.

### Data pipeline (`data/`)

All datasets follow the pattern: download → tokenize → write `train.bin`/`val.bin` as uint16 numpy arrays. Character-level encoding also saves `meta.pkl` with `stoi`/`itos`/`vocab_size`. Tokenized data is memory-mapped at training time.

## Windows Notes

- Use `--compile=False` if `torch.compile` causes issues (common on Windows/older PyTorch)
- The `pyproject.toml` includes CUDA 12.8 PyTorch wheel source for RTX 50-series GPUs (via uv)
