# Quantum Channel Language Model (QCLM)

A proof-of-concept language model built from **quantum mathematics** — Hilbert
spaces, complex amplitudes, the Born rule, and quantum channels — that is
**neither a transformer (no attention) nor an RNN (no gated recurrence).**

Context is carried as a **quantum density matrix** `ρ`. Reading a token is a
**measurement**; the next-token distribution is the **Born rule** `p(x)=Tr(ρ Mₓ)`;
observing a token **collapses and evolves** the state through a completely-positive
quantum channel. The only nonlinearity in the entire model is measurement.

> One-line identity: the QCLM is a *complex-valued, trace-preserving linear
> state-space model with a Born-rule readout* — a quantum cousin of modern SSMs,
> distinguished by genuine amplitude **interference**.

## Headline result (tiny-shakespeare, char-level)

At matched Hilbert-space dimension, the quantum model beats its **decohered twin**
(off-diagonal coherences zeroed each step → an exact classical Hidden Markov Model
of identical size). The quantum advantage *grows* with dimension.

| dim n | quantum (bits/char) | classical / decohered | advantage |
|---|---|---|---|
| 24 | **3.008** | 3.208 | 0.200 |
| 48 | **2.806** | 3.045 | 0.239 |

Reference baselines: uniform 6.022 · unigram 4.829 · bigram 3.583 · trigram 2.952
· 4-gram 2.574. The quantum model also keeps a genuinely coherent state
(mean purity ≈ 0.94, mean ℓ1-coherence ≈ 37; a classical model has coherence 0).

## Install / run

```bash
pip install torch numpy scipy scikit-learn matplotlib   # CPU is fine
# (tiny-shakespeare is already in data/)

# 0. verify the model is a valid quantum instrument
python experiments/test_sanity.py

# 1. classical n-gram reference points
python qlm/baselines.py

# 2. train the quantum model (~15 min on 4 CPU cores)
python qlm/train.py --dim 48 --kraus 4 --block 64 --steps 1800 --tag q_d48

# 2b. train the decohered (classical-HMM) ablation
python qlm/train.py --dim 48 --kraus 4 --block 64 --steps 1800 --tag c_d48 --decohere

# 3. diagnostics + figures + samples
python qlm/analysis.py --ckpt artifacts/q_d48_best.pt

# 4. prompted generation ("expresses itself")
python qlm/demo.py --ckpt artifacts/q_d48_best.pt --temperature 0.6 --top_k 8

# 5. aggregate all runs into the headline table + figures
python qlm/aggregate.py

# 6. DGX Spark scaling calculator (param/memory/compute table)
python qlm/scaling.py
```

## Scale-up tooling (DGX Spark, ~2B)

```bash
pip install tokenizers datasets        # for BPE + streaming FineWeb

# train a byte-level BPE (smoke test on local data)
python qlm/tokenizer.py
# streaming FineWeb -> packed token blocks (offline smoke test on local data)
python qlm/data_fineweb.py
# GPU/scale training (streams FineWeb-Edu on the Spark; runs on CPU with tiny flags)
python qlm/train_scale.py --device auto --vocab 16384 --dim 128 --kraus 4 \
    --source fineweb --fineweb_name sample-10BT --tag qclm_2b
```

The 2B configuration, corpus/tokenizer rationale, token budget, and exact commands
are in **`paper/PLAN_2B.md`**. Note a useful property: because the POVM resolves
the identity, the next-token denominator is exactly 1 — so **training compute is
independent of vocabulary size** (`O(W·n³)` per token), making the model
compute-light for its parameter count.

## Repository layout

```
qlm/model.py        QuantumChannelLM: Kraus operators, POVM, Born-rule forward,
                    quantum-channel state update, decoherence ablation, generation
qlm/train.py        autoregressive NLL training (the quantum forward algorithm)
qlm/data.py         char tokenizer + dataset
qlm/baselines.py    add-k smoothed n-gram baselines
qlm/analysis.py     purity / ℓ1-coherence trajectories, Hilbert-space char map, samples
qlm/demo.py         prompted generation
qlm/aggregate.py    headline table + figures (fig_learning_curves.png, fig_bits_bar.png)
qlm/scaling.py      DGX Spark parameter/memory/compute calculator
qlm/tokenizer.py    byte-level BPE tokenizer (scale-up)
qlm/data_fineweb.py streaming FineWeb/FineWeb-Edu -> packed token blocks (scale-up)
qlm/train_scale.py  GPU/scale training (BPE + streaming + grad-accum + TBPTT)
experiments/        sanity checks, sweep/orchestration helpers
paper/paper.md      the write-up;       paper/NOTES.md   research log
paper/SCALING.md    scaling overview;   paper/PLAN_2B.md  the 2B run plan
artifacts/          checkpoints, logs, figures
```

## Scaling to ~1B

See `paper/SCALING.md`. Short version: switch to a BPE subword vocabulary (so
params scale cheaply in `V` while the Hilbert dimension `n` stays small) — e.g.
`V=8192, n=128, W=4 → 1.07B params, ~21.5 GB training memory`, comfortably inside
a single DGX Spark's 128 GB.

## The math, briefly

```
state:     ρ ∈ ℂ^{n×n}, Hermitian, PSD, Tr ρ = 1          (a density matrix)
per token: Kraus {K_{x,w}};  channel E_x(ρ)=Σ K ρ K†;  POVM M_x = Σ K† K
constraint Σ_{x,w} K†K = I   ⇒   Σ_x M_x = I   (a valid quantum measurement)
predict:   p(x_t | x_<t) = Tr(ρ_{t-1} M_{x_t})            (Born rule)
update:    ρ_t = E_{x_t}(ρ_{t-1}) / p(x_t)                (collapse + renormalize)
train:     minimize Σ_t −log p(x_t | x_<t)               (quantum forward algorithm)
```
The decoherence ablation zeros the off-diagonals of `ρ` each step, collapsing the
model to a classical HMM and isolating the value of quantum interference.
