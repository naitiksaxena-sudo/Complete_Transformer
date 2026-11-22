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
| Token embedding (scaled by √d_model) | Done |
| Byte-pair-encoding tokenizer | In progress |
|residual connections, layer normalization | Done |
| Encoder block | Done |
| Decoder block with masked attention | Done |
| Padding mask | Done |
| Output layer (linear + softmax) | Done |
| Full transformer model | Done |
| Loss function (label smoothing) | Done |
| Optimizer and learning-rate warm-up schedule | Done |
| Training loop | In progress |
| Greedy decoding | Done |
| Copy-task sanity test | Done |
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

**Residual connections and layer normalization.** Built layer normalization by hand and wrapped every sub-layer as LayerNorm(x + Sublayer(x)), with dropout applied to the sub-layer output before the addition. Checked that each token comes out with mean near 0 and standard deviation near 1.

**Token embedding.** Built the learned embedding table scaled by √d_model. Initialised the table with standard deviation d_model^-0.5, so that after scaling its values are about the same size as the positional encoding. Without this, the copy test below could not learn, because the position signal was swamped.

**Encoder.** Joined multi-head self-attention and the feed-forward layer, each with its residual connection and layer normalization, into an encoder layer, and stacked six independent copies. Checked the output shape and normalization.

**Decoder.** Built a decoder layer with three sub-layers: masked self-attention, encoder-decoder attention, and the feed-forward layer. Stacked six independent copies. Checked that changing later target tokens does not change the outputs at earlier positions.

**Padding mask.** Built a source padding mask from the token IDs and passed it through encoder self-attention and encoder-decoder attention. Checked that the same sentence padded to two different lengths gives identical outputs for its real tokens, and that it does not without the mask.

**Output layer and full model.** Added the linear output layer and assembled the full transformer: source and target embeddings with positional encoding, encoder, decoder and output layer. The parameter count (45,843,632 with test vocabularies of 1000 and 1200 tokens) matched a hand calculation.

**Loss and optimizer.** Used cross-entropy with label smoothing 0.1 that skips padding positions. Used Adam with β1 = 0.9, β2 = 0.98, ε = 10⁻⁹ and the warm-up learning-rate schedule from Section 5.3 of the paper, checked at selected steps.

**Training step and copy-task test.** Wrote one training step (forward pass, loss, backward pass, optimizer step, scheduler step) with the decoder input shifted by one position. Tested it on a toy task where the model must copy a random sequence. The first attempt did not learn (fixed by the embedding initialisation above), the second collapsed at a high learning rate (fixed with a lower peak), and the third settled near the loss floor set by label smoothing, with accuracy between 0.98 and 1.0. This shows the model can learn; it is not a translation result.

**Greedy decoding.** Wrote decoding that runs the encoder once and then generates one token at a time from the model's own predictions. On the trained toy model, the generated sequences matched the source.

**Tokenizer.** Wrote byte-pair-encoding training. Encoding and decoding are not finished yet.

## Honest note

This is a learning project, not a production implementation or a benchmark.

## Tools

Python, PyTorch, Kaggle notebooks, other related stuff.
