# RustGPT - A Transformer Language Model in Rust

A character-level GPT (Generative Pre-trained Transformer) implementation in Rust using the HuggingFace Candle framework with Metal acceleration for Apple Silicon.

## What We're Building

We're implementing a **language model** - a neural network that learns to predict the next character in a sequence of text. By training on Shakespeare's works, the model learns patterns and generates Shakespeare-like text.

Think of it like autocomplete on your phone, but trained to write in Shakespearean style.

## The Problem We're Solving

Traditional language models (like predicting one word from the previous few) have a **memory problem**: they can't look back very far to understand context. If you're reading a novel, you need to remember details from earlier chapters!

The original transformer model (2017) solved this with a clever mechanism called **Attention**, which lets the model look at *any* part of the previous text, no matter how far back.

## How It Works: Attention Explained Simply

Imagine you're reading a sentence:
> "The king sat on the throne. **He** was wise."

When predicting the next word after "He", your brain instantly remembers that "He" refers to "the king" - even though they're separated by other words.

**Attention works the same way:**

1. **Query**: "What am I trying to predict right now?" (the next character)
2. **Key & Value**: "What information do I have from the past?"
3. **Matching**: The model scores how relevant each past position is
4. **Weighted Sum**: It combines all past positions weighted by relevance

In math form:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**In plain English**: 
- Multiply current position (Q) against all past positions (K) to get similarity scores
- Scale by $\sqrt{d_k}$ so gradients stay reasonable
- Use softmax to turn scores into percentages (they sum to 1)
- Take a weighted average of all past information (V) using those percentages

**Multi-Head Attention**: Do this process 6 times in parallel with different "lenses", then combine. Like looking at a document through 6 different colored glasses and merging what you see.

## Architecture Overview

```
Input Text
    ↓
Token Embedding (map characters to vectors)
    ↓
Position Embedding (add location information)
    ↓
4 Transformer Blocks, each containing:
  ├─ Multi-Head Attention (6 attention heads)
  │  └─ What parts of history are relevant?
  └─ Feed Forward Network (MLP)
     └─ Process and mix the information
    ↓
Output Layer Norm
    ↓
Predict Next Character (vocab_size logits)
```

### Key Components

**Causal Masking**: During training, we prevent the model from "cheating" by looking at future characters. We mask them out with a huge negative number (makes softmax ~0).

**Embeddings**: 
- **Token**: Each character gets a unique 192-dimensional vector
- **Position**: Each position (1st char, 2nd char, etc) gets its own vector added to token embedding

**Layer Normalization**: Stabilizes training by keeping values in a reasonable range.

**Dropout**: During training, randomly "turns off" 20% of neurons to prevent overfitting.

## Why Rust?

- **Performance**: Close to C/C++ speed, no garbage collector
- **Memory Safety**: Catches bugs at compile time
- **Metal Support**: Direct access to Apple Silicon GPU acceleration
- **Deploy Anywhere**: Single compiled binary, no runtime needed

## Configuration

The model is configured with (see `src/main.rs`):
- **vocab_size**: 65 unique characters in Shakespeare
- **n_embd**: 192 dimensions for embeddings
- **n_head**: 6 attention heads
- **n_layer**: 4 transformer blocks
- **block_size**: 128 characters of context
- **batch_size**: 32 sequences per training step
- **Learning Rate**: 3e-4 (Adam optimizer)

This gives us ~1.8M parameters to learn.

## Training Data

Uses the [TinyShakespeare dataset](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) (~1MB of Shakespeare's complete works).

## Building & Running

### Requirements
- Rust 1.70+
- macOS (for Metal acceleration, or CPU fallback)

### Train the Model
```bash
cargo run --release
```

This will:
1. Load and tokenize Shakespeare text
2. Train for 3000 iterations
3. Print loss every 500 steps
4. Generate sample text
5. Save weights to `model.safetensors`

**Training time**: ~20-40 minutes on Apple Silicon with Metal

### Output Files
- `model.safetensors` - Binary weight file
- `model.safetensors.meta.json` - Tensor metadata (shapes/layout)

## Generated Text Example

After training, the model generates text like:
```
ROSALIND: Why, cousin, I do but fain let me to the Duke,
That he may know of thy love to me,
For I am sure thou lovest me...
```

It learns Shakespeare's vocabulary, character names, and verse structure!

## Mathematical Background

### Attention Formula
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = Query matrix (what we're looking for)
- $K$ = Key matrix (labels for memory)
- $V$ = Value matrix (actual memory content)
- $d_k$ = Dimension of keys (192 / 6 heads = 32)

### Multi-Head
```
head_output_i = Attention(Q_i, K_i, V_i)  for i = 1..6
concat_heads = [head_output_1 || head_output_2 || ... || head_output_6]
output = W_o @ concat_heads
```

### Feed Forward (per Transformer Block)
```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```
- First layer: 192 → 768 (expand)
- ReLU activation (zero out negatives)
- Second layer: 768 → 192 (project back)

### Loss Function
```
Loss = Cross Entropy(predicted_logits, actual_next_character)
```
Measures how wrong our predictions are. Optimizer (AdamW) adjusts weights to minimize this.

## References

- **Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.10677) (Vaswani et al., 2017)
- **Framework**: [Candle by HuggingFace](https://github.com/huggingface/candle)
- **Dataset**: [TinyShakespeare](https://github.com/karpathy/char-rnn)
- **Inspiration**: [Andrej Karpathy's makemore](https://github.com/karpathy/makemore) series

## Implementation Notes

- **Device**: Uses Metal GPU (M-series Macs) with CPU fallback
- **Precision**: F32 (32-bit floats) for stability
- **Batch Processing**: Processes 32 sequences in parallel
- **Evaluation**: Runs 200 evaluation iterations on validation set every 500 training steps

## Future Improvements

- [ ] Load/resume training from checkpoint
- [ ] Inference-only mode
- [ ] Web interface for generation
- [ ] Support for larger models (multi-GPU)
- [ ] Mixed precision training
- [ ] Export to ONNX format

## License

MIT
