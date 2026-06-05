# Scaling the Quantum Channel LM to ~1B on a DGX Spark

This document translates the CPU proof-of-concept into a concrete, verified
configuration for a single **NVIDIA DGX Spark (GB10 Grace Blackwell)**.

## 1. Hardware envelope (verified)

| Resource | DGX Spark (GB10) |
|---|---|
| Unified memory | **128 GB** LPDDR5x (CPU+GPU coherent) |
| Memory bandwidth | **~273 GB/s** (the main bottleneck) |
| Tensor compute | **~1 PFLOP FP4 sparse** (~500 dense FP4 TFLOP, ~125 dense BF16 TFLOP) |
| CPU | 20-core Arm (10× Cortex-X925 + 10× Cortex-A725) |
| Practical limits | inference ≤200B (FP4), fine-tune ≤~70B |

Implication: a ~1B-parameter QCLM is **far** inside the memory budget; the
binding practical constraints are (a) keeping the Hilbert dimension `n` small
enough that the `O(n³)` channel update is cheap, and (b) complex-arithmetic
kernel efficiency.

## 2. How the QCLM scales

Parameter count (real numbers; complex = 2 reals), **verified to match the code
exactly**:

```
params ≈ 2 · V · W · n²        (+2n for the initial state)
```

- `V` = vocabulary size, `n` = Hilbert-space dimension, `W` = Kraus operators/token.
- Parameters live in the per-token Kraus operators — effectively a large,
  structured input/output operator table (the QCLM analogue of embedding +
  unembedding).

**Key design decision: switch from characters to a BPE subword vocabulary.**
Char-level (V=65) would need `n ≈ 1387` to reach 1B params, and dense `O(n³)`
operations make that impractical. A subword vocab puts the parameters where they
scale cheaply (in `V`) while keeping `n` small and the per-step linear algebra
fast.

## 3. Recommended configurations

(from `qlm/scaling.py`; "train(GB)" includes fp32 master + Adam + grads + acts at
batch 64, T 512)

| Preset | V | n | W | Params | Weights | Train mem | GF/token |
|---|---|---|---|---|---|---|---|
| cpu-poc (this repo) | 65 | 48 | 4 | 1.2 M | 0.0 GB | 0.1 GB | — |
| spark-100M (char) | 65 | 320 | 4 | 53 M | 0.2 GB | 14 GB | 1.10 |
| spark-300M (bpe-8k) | 8192 | 64 | 4 | 0.27 B | 1.1 GB | 5 GB | 0.28 |
| **spark-1B (bpe-8k)** | **8192** | **128** | **4** | **1.07 B** | **4.3 GB** | **21.5 GB** | **1.14** |
| spark-1B (bpe-16k) | 16384 | 96 | 4 | 1.21 B | 4.8 GB | 22 GB | 1.24 |
| spark-1.2B (bpe-32k, W2) | 32000 | 96 | 2 | 1.18 B | 4.7 GB | 21 GB | 2.37 |
| spark-2B (bpe-16k) | 16384 | 128 | 4 | 2.15 B | 8.6 GB | 38 GB | 2.21 |

**Recommended target: `spark-1B (bpe-8k)`** — 1.07 B params, ~21.5 GB training
footprint, leaving ample headroom on 128 GB for larger batches / longer context.

### Suggested run config (starting point)
```
vocab:        BPE 8192  (train on a larger corpus than tiny-shakespeare)
dim n:        128
kraus W:      4
block T:      512–1024
batch:        64–128   (memory allows scaling this up)
dtype:        complex stored bf16 + fp32 master weights (mixed precision)
optimizer:    Adam, lr 1e-3 cosine, grad clip 1.0, warmup ~1–2k steps
```

## 4. Code changes needed for the scale-up

The current `qlm/model.py` already runs unchanged on GPU (`model.to('cuda')`); the
isometry projection (`eigh`), `einsum`, and complex ops are all CUDA-capable.
For an efficient serious 1B run, add:

1. **BPE tokenizer** (e.g. `tokenizers`/`sentencepiece`) + a larger training
   corpus. Swap `CharTokenizer` for the subword tokenizer; nothing else changes.
2. **Mixed precision / complex-as-real kernels.** Blackwell tensor cores don't do
   complex natively; implement complex matmuls as real 2×2 block ops in bf16 to
   hit the tensor cores (~4× the real FLOPs but full hardware utilization). Keep
   an fp32 master copy of weights.
3. **Chunked scan over T.** The channel composition `E_{x_t}∘…∘E_{x_1}` is
   **associative**, so the sequence likelihood admits a parallel
   (Blelloch-style) prefix scan over transfer maps — or, more memory-friendly, a
   chunked sequential scan (parallel within a chunk, sequential across chunks),
   exactly as done for linear SSMs. This removes the per-token Python loop.
4. **Optional rank-capped `ρ`.** If memory/compute on activations becomes tight,
   represent `ρ` in low-rank eigenform (`Σ_{k≤r} λ_k |φ_k⟩⟨φ_k|`) for `O(r·n²)`
   updates instead of `O(n³)`.

## 5. What we expect from scale-up

The CPU PoC is *capacity-bound*: validation bits/char improves monotonically with
`n` (n=24→48 took 3.01→2.81), and the quantum-over-classical advantage **grows**
with dimension (Δ 0.200→0.239). A subword 1B model with n=128 and a larger corpus
should move well past the n-gram regime into genuine word- and phrase-level
coherence, while preserving the architecture's defining feature — generation by
quantum measurement and interference.

> Tell me which preset you want and I'll wire up the BPE tokenizer + mixed-precision
> path and hand you a single `train.py` invocation for the Spark.
