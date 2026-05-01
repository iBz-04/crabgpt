# RustGPT

A character-level GPT in Rust using Candle with Metal acceleration on Apple Silicon.

## Requirements
- Rust 1.70+
- macOS for Metal (CPU fallback is automatic)

## Quick Start
1. Put your training text in input.txt (TinyShakespeare works: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).
2. Run:
   cargo run --release

## Defaults (src/main.rs)
- batch_size: 32
- block_size: 256
- max_iters: 8000
- eval_interval: 1000
- eval_iters: 100
- learning_rate: 3e-4
- n_embd: 192
- n_head: 6
- n_layer: 4
- dropout: 0.1
- max_new_tokens: 500
- temperature: 0.9
- top_k: 40

## Output
- Prints train/val loss during training
- Prints a generated sample
- Writes model.safetensors and model.safetensors.meta.json

## Notes
- Uses Metal if available, otherwise CPU.
- Generation is seeded from a random slice of the training data for better coherence.

