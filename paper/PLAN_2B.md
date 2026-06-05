# Plan: training a ~2B Quantum Channel LM on a DGX Spark

This is the concrete scale-up plan for the QCLM proof-of-concept, with the
tokenizer and corpus decisions and exact commands. All code referenced here is in
the repo and was validated end-to-end on CPU with tiny data; only the corpus and
device differ on the Spark.

---

## 1. Tokenizer decision — byte-level BPE, vocab 16,384

**Recommendation: GPT-2-style byte-level BPE, `V = 16,384`.** (`qlm/tokenizer.py`)

Why byte-level BPE, and why this size, for *this* model specifically:

- **Never OOV on web text.** Byte-level BPE works on raw UTF-8 bytes, so FineWeb's
  messy unicode/markup never breaks tokenization.
- **More text per quantum step.** Subwords (~4–4.5× fewer tokens than characters)
  mean the fixed-size density matrix `ρ` summarizes more context per step, which
  directly helps the linear-state model's effective memory and coherence. The
  model also stops spending capacity on spelling.
- **Vocabulary is nearly "free" for training compute.** A property we exploit
  (see §3): QCLM *training* cost is independent of `V`. The only costs that grow
  with `V` are parameter memory and generation. So `V=16,384` is a balanced
  choice; `V=32,768` is also fine on 128 GB if you prefer denser tokens.

Alternatives considered: SentencePiece-Unigram (fine, marginal differences);
word/char-level (rejected — char wastes capacity and can't reach 2B at a tractable
Hilbert dimension; word-level has OOV). Byte-level BPE is the robust default.

## 2. Corpus decision — FineWeb-Edu, `sample-10BT`

**Recommendation: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, streamed.**
(`qlm/data_fineweb.py`, `streaming=True` — no full download.)

- FineWeb-Edu is the educational-quality-filtered slice of FineWeb; higher signal
  per token, which matters on a single-box token budget aimed at coherence rather
  than raw scale.
- `sample-10BT` (~10B tokens) is far more than a first 2B run needs (§4) and
  streams without filling disk. Move to `sample-100BT` only if scaling the budget.
- Train the BPE on the first ~50k streamed documents, then train the model on the
  stream. Documents are joined with `<|endoftext|>`.

## 3. Model configuration — 2.15B parameters

| field | value | note |
|---|---|---|
| vocab `V` | 16,384 | byte-level BPE |
| Hilbert dim `n` | 128 | dense density matrix `ρ ∈ ℂ^{128×128}` |
| Kraus `W` | 4 | operators per token |
| **parameters** | **2.15 B** | `= 2·V·W·n²` |
| weights (complex64) | 8.6 GB | bf16 master adds 4.3 GB |
| training footprint | ~38 GB | weights + Adam + grads + activations |
| block `T` | 512 (→1024 later) | sequence length |
| effective batch | ~0.5–1 M tokens | via `--batch` × `--grad_accum` |
| TBPTT | 128 | bounds activation memory |
| optimizer | AdamW, lr 1e-3 cosine, warmup 2k, clip 1.0 | |

Everything fits comfortably in the Spark's 128 GB unified memory with room to
grow the batch.

### The efficiency property that makes this cheap
Because the POVM resolves the identity (`Σ_x M_x = I`), the next-token denominator
is **exactly 1** — there is no softmax sum over the vocabulary. The Born
probability is `p(x_t) = Tr(E_{x_t}(ρ))`, which already falls out of the state
update. Therefore **training compute is `O(W·n³)` per token, independent of `V`**.
The model is compute-light for its parameter count (its 2B params are accessed
sparsely, like a large embedding table). At 2B the practical limiter is **memory
bandwidth** — the Adam step over all parameters (~110 ms at 273 GB/s) and the
Kraus-operator gathers — not FLOPs. (`qlm/model.py` implements the `Tr(E)`
shortcut; `qlm/scaling.py` reports the numbers.)

## 4. Token budget & expected wall-clock

Throughput must be measured on-device (the unknown is complex-kernel efficiency),
but the compute envelope implies an effective **~8–20k tokens/s** once the
optimizer and gathers are included. Then:

| tokens | @20k tok/s | @8k tok/s |
|---|---|---|
| 1 B (first milestone) | ~14 h | ~35 h |
| 3 B | ~1.7 d | ~4.3 d |

Plan: **target 1 B tokens first**, checkpoint and inspect samples; extend to 3–5 B
only while validation bits/token is still falling. (QCLM data-optimal ratios are
unknown — this is a research run, so watch the curve rather than assuming
Chinchilla.)

## 5. Staged execution

**Stage A — GPU port validation (hours).** Train a small model on a few hundred M
tokens to confirm CUDA correctness and that subword-level samples clearly beat the
char PoC:
```bash
python qlm/train_scale.py --device auto \
  --source fineweb --fineweb_dataset HuggingFaceFW/fineweb-edu --fineweb_name sample-10BT \
  --vocab 16384 --dim 64 --kraus 4 \
  --block 512 --batch 16 --grad_accum 4 --tbptt 128 \
  --steps 3000 --lr 1e-3 --warmup 500 --eval_every 500 --tag qclm_portcheck
```

**Stage B — the 2B run.**
```bash
python qlm/train_scale.py --device auto \
  --source fineweb --fineweb_dataset HuggingFaceFW/fineweb-edu --fineweb_name sample-10BT \
  --vocab 16384 --dim 128 --kraus 4 \
  --block 512 --batch 16 --grad_accum 32 --tbptt 128 \
  --steps 60000 --lr 1e-3 --warmup 2000 --clip 1.0 \
  --eval_every 1000 --ckpt_every 2000 --tag qclm_2b
```
(`--batch 16 × --grad_accum 32 × --block 512 ≈ 262k tokens/optimizer-step`; raise
`--grad_accum` for a larger effective batch, memory permitting.)

## 6. The one performance lever worth implementing first

`qlm/model.py` runs **as-is** on CUDA in complex64. For peak Blackwell tensor-core
utilization, replace the complex `einsum`s in the state update with **real 2×2
block matmuls in bf16** (a complex matmul = 4 real matmuls; keep an fp32 master
copy of weights). This is a localized kernel change — the architecture, math, and
training script are unchanged — and is the single biggest throughput win. The
sequential scan over `T` can additionally be turned into a chunked/associative
scan (channel composition is associative), but TBPTT already bounds memory, so do
the complex-as-real kernels first.

## 7. What success looks like

A subword 2B QCLM trained on FineWeb-Edu should move decisively past the n-gram
regime of the CPU PoC into genuine word- and sentence-level coherence on
open-domain English, while preserving the defining mechanism — **generation by
quantum measurement and interference**, with the same controlled decoherence
ablation available to re-confirm that the quantum coherences are still doing work
at scale.
