# Complete Transformer
A complete transformer built from scratch and hand-coded from the ground up in PyTorch, inspired by the paper *Attention Is All You Need*.

## Goal

To understand the transformer by building every part myself instead of importing it. The aim is a complete, working transformer with its own training loop and testing loop, where I can explain each piece of the code in my own words.

## How this is being built

I learn each concept first and then write the code by hand. I use an AI assistant (Claude) as a tutor to explain ideas in plain language and to review my output. I write the code myself and run it to check that it works.

## Roadmap

| Part | Status |
|---|---|
| Scaled dot-product attention (by hand on paper, checked in PyTorch) | Done |
| `attention(Q, K, V)` function with √d scaling | Done |
| Q, K and V built from an input `X` using weight matrices | Done |
| Multi-head attention | In progress |
| Positional encoding | Upcoming |
| Feed-forward layer, residual connections, layer normalization | Upcoming |
| Encoder block | Upcoming |
| Decoder block with masked attention | Upcoming |
| Full transformer model | Upcoming |
| Training loop | Upcoming |
| Testing loop | Upcoming |

## Repository structure

```
transformer.ipynb    The Kaggle notebook with the code
README.md            This file
```

## Learning log

**Attention basics.** Calculated a full attention step (QKᵀ, softmax, multiply by V) by hand on paper, then reproduced it in PyTorch to confirm the numbers matched.

**Scaling.** Added the √d scaling and compared behavior with small and large feature sizes, including a 512-feature experiment, to see how softmax responds.

**Q, K, V from input.** Produced Q, K and V from a single input `X` using weight matrices and checked the shape of every tensor.

## Honest notes

- This is a learning project, not a production implementation or a benchmark.
- The code is written for understanding and clarity, not for speed.

## Tools

Python, PyTorch, Kaggle notebooks, other related stuff.
