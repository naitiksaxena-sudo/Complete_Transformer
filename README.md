# Complete Transformer
A complete transformer built from scratch and hand-coded from the ground up in PyTorch, inspired by the paper *Attention Is All You Need*.

## Goal

To understand the transformer architecture by building every part from scratch. 
The aim is a complete, working transformer with its own training loop and testing loop, and further experimentation with own techniques.


## Roadmap

| Part | Status |
|---|---|
| Scaled dot-product attention (by hand on paper, checked in PyTorch) | Done |
| `attention(Q, K, V)` function with √d scaling | Done |
| Q, K and V built from an input `X` using weight matrices | Done |
| Multi-head attention | Done |
| Positional encoding | Done |
| Feed-forward layer | Done
|residual connections, layer normalization | Upcoming |
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

## Progress log

**Attention basics.** Calculated a full attention step (QKᵀ, softmax, multiply by V) by hand on paper, then reproduced it in PyTorch to confirm the numbers matched.

**Scaling.** Added the √d scaling and compared behavior with small and large feature sizes, including a 512-feature experiment, to see how softmax responds.

**Q, K, V from input.** Produced Q, K and V from a single input `X` using weight matrices and checked the shape of every tensor.

**Multi-head attention.** Worked out complete multi head attention with 8 heads while verifying the results.

**Positional encoding.** Build end to end pipeline of positional encoding by applying sine , cosine wave frequency framework.

**Feed Forward Layer.** Reproduced the feed Forward concept by transforming embedding from 512 -> 2048, then reverse, verified the results.

## Honest note

This is a learning project, not a production implementation or a benchmark.

## Tools

Python, PyTorch, Kaggle notebooks, other related stuff.
