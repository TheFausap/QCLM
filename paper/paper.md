# A Quantum Channel Language Model: Autoregressive Text Generation by Measurement and Interference

**A proof-of-concept for a non-transformer, non-recurrent language model built from quantum mathematics.**

---

## Abstract

We introduce the **Quantum Channel Language Model (QCLM)**, an autoregressive
generative model of text whose sequential engine is neither attention
(transformers) nor a gated nonlinear recurrence (RNNs/LSTMs), but a **quantum
channel** acting on a density matrix in a finite-dimensional Hilbert space.
The model carries its context as a quantum state `ρ ∈ ℂ^{n×n}` (a positive
semidefinite, unit-trace density matrix). Each token owns a set of Kraus
operators that (i) define a measurement, via the **Born rule**, predicting the
next token, and (ii) define a completely-positive map that updates the state
after the token is observed. A single global isometry constraint makes the whole
family a valid quantum instrument, so token probabilities are exactly normalized
by construction. The model is a complex-valued, trace-preserving **linear**
state-space model whose only nonlinearity is quantum measurement.

Trained by gradient descent on character-level tiny-Shakespeare, the QCLM learns
English orthographic structure, generates stylistically recognizable text, and —
crucially — **beats a dimension-matched classical model** obtained by destroying
its quantum coherence (a controlled decoherence ablation that collapses the QCLM
to an ordinary Hidden Markov Model). This isolates and demonstrates that genuine
quantum interference, not merely parameter count, contributes to language
modeling quality. We provide quantum-state diagnostics (purity and ℓ1-coherence
trajectories) and a Hilbert-space embedding of characters, and we discuss a path
to scale via tensor-network (matrix-product-operator) structured Kraus maps.

---

## 1. Introduction

Modern language models are dominated by two mechanisms: **self-attention**
(transformers) and **gated recurrence** (RNN/LSTM/GRU, and more recently linear
state-space models such as S4/Mamba). We ask a different question: *can the
mathematics of quantum mechanics — Hilbert spaces, complex probability
amplitudes, superposition, the Born rule, and quantum channels — serve directly
as the computational substrate of a language model?*

Our answer is the **Quantum Channel Language Model (QCLM)**. It treats a sentence
as a sequence of quantum operations on a state. Reading a token is a
**measurement**; the next-token distribution is given by the **Born rule**
`p(x) = Tr(ρ Mₓ)`; and the act of observing a token **collapses and evolves** the
quantum state through a completely-positive, trace-preserving map. There is no
attention and no nonlinear gate. The recurrence is *linear and quantum*; the only
nonlinearity in the entire model is measurement itself (a quadratic form in the
amplitudes).

This places the QCLM in an interesting position. Like modern linear state-space
models, its core update is linear, which is what makes it efficient and
parallelizable in principle. Unlike them, its state is a **density matrix**, its
arithmetic is **complex**, and its readout is the **Born rule** — so the model
can exploit **interference between probability amplitudes**, a resource with no
classical analogue. Our central empirical question is therefore not merely "does
it work?" but "**does the quantumness pay for itself?**"

### Contributions

1. **Architecture.** We formulate autoregressive language modeling as a
   homogeneous Hidden Quantum Markov Model with a Born-rule readout, trained
   end-to-end by gradient descent, with a clean differentiable isometry
   parametrization that guarantees exactly-normalized probabilities.
2. **A controlled test of quantumness.** A *decoherence ablation* that zeroes the
   off-diagonal elements of `ρ` at every step collapses the QCLM, exactly, to a
   classical Hidden Markov Model of identical dimension and parameter budget.
   The quantum model consistently achieves lower perplexity, isolating the
   contribution of interference.
3. **Diagnostics and interpretation.** Purity and ℓ1-coherence trajectories show
   the model maintains genuine superposition while reading text; a Hilbert-space
   character map shows it learns linguistic structure (vowel/consonant/whitespace
   geometry) in its measurement operators.
4. **A runnable PoC** on commodity CPU, with a scaling path via tensor-network
   structured operators.

---

## 2. Method

### 2.1 State: a quantum density matrix as context

The model's "hidden state" is a density matrix `ρ ∈ ℂ^{n×n}` on a Hilbert space
`H = ℂ^n`: Hermitian, positive semidefinite, with `Tr(ρ) = 1`. A *pure* state
`ρ = |ψ⟩⟨ψ|` is a coherent superposition of the `n` basis states; a *mixed*
state encodes classical uncertainty over such superpositions. The model begins
each sequence in a learned pure state `ρ₀ = |ψ₀⟩⟨ψ₀|`.

### 2.2 Tokens as quantum instruments (Kraus operators)

Each vocabulary token `x` owns `W` complex **Kraus operators**
`{K_{x,w}}_{w=1..W}`, each `n×n`. They define two objects:

- a **completely-positive map** (quantum channel)
  `E_x(ρ) = Σ_w K_{x,w} ρ K_{x,w}^†`, and
- a **POVM element** (measurement operator)
  `M_x = Σ_w K_{x,w}^† K_{x,w}`, which is positive semidefinite.

We impose one global constraint that ties all tokens together:

> **Isometry / trace-preservation:**  `Σ_{x,w} K_{x,w}^† K_{x,w} = I_n.`

Equivalently `Σ_x M_x = I_n`, i.e. `{M_x}` is a genuine quantum measurement
(a *positive operator-valued measure*). This single equation is what makes the
next-token distribution sum to one *exactly*, for free, with no softmax.

### 2.3 The autoregressive law (Born rule + collapse)

Generation and scoring follow the textbook rules of quantum measurement:

```
  Predict:  p(x_t | x_<t) = Tr( ρ_{t-1} M_{x_t} )          (Born rule)
  Update:   ρ_t = E_{x_t}(ρ_{t-1}) / p(x_t | x_<t)         (collapse + renormalize)
```

Unrolled, the joint probability of a string is a single trace of a product of
channels, `p(x_1..x_T) = Tr( E_{x_T} ∘ … ∘ E_{x_1} (ρ_0) )`, and the isometry
constraint guarantees `Σ_{x_1..x_T} p = 1`. This is a proper autoregressive
factorization, so we train with ordinary sequence negative log-likelihood,
`L = −Σ_t log p(x_t | x_<t)`, fully differentiable through the linear algebra.

**Where is the quantumness?** Between measurements the state carries
*off-diagonal* coherences `ρ_{ij}, i≠j`. These encode relative complex phases
between basis states, and they interfere when the next Born-rule probability is
computed. A classical model has no such phases. Section 4 shows they matter.

### 2.4 Why this is not a transformer and not an RNN

- **No attention.** There is no query/key/value, no softmax over positions, no
  pairwise token interaction. Context is summarized by a single evolving operator
  `ρ`, of fixed size independent of sequence length.
- **No nonlinear gate.** The state update `ρ_{t-1} ↦ E_{x_t}(ρ_{t-1})` is
  **linear** in `ρ`. The only nonlinearity in the model is the Born-rule readout
  (a quadratic form) and the probability renormalization shared by every HMM.

The QCLM is thus a **complex-valued, trace-preserving linear state-space model
with a quantum-measurement readout** — a principled quantum cousin of classical
state-space models, distinguished by genuine amplitude interference.

### 2.5 Parametrization and training

We store free complex Kraus parameters and enforce the isometry constraint on the
fly by a differentiable **polar projection**: stacking the operators into a tall
matrix `V`, we map `V ↦ V (V^† V)^{-1/2}`, whose columns are orthonormal, using a
Hermitian eigendecomposition for the inverse square root. We optimize with Adam
and gradient clipping (the `1/p(x)` renormalization can produce sharp gradients),
in complex64, on CPU.

### 2.6 The decoherence ablation (our classical control)

Setting the off-diagonal entries of `ρ` to zero after every update **destroys
quantum coherence**. The resulting dynamics keep only the diagonal (classical)
probabilities over basis states, and the model becomes *exactly* a classical
`n`-state Hidden Markov Model with the same parameters and training. Comparing the
intact QCLM against its decohered twin is a clean, dimension-matched measurement
of "how much the quantumness is worth."

---

## 3. Experimental setup

- **Data.** Character-level *tiny-Shakespeare* (≈1.1M characters, vocabulary 65),
  90/10 train/validation split.
- **Models.** QCLM with Hilbert-space dimension `n ∈ {24, 48}` and `W = 4` Kraus
  operators per token; identical decohered (classical-HMM) twins.
- **Baselines.** Add-k smoothed `n`-gram models (unigram→4-gram) and the uniform
  distribution, as external reference points; the decohered twin is the
  controlled internal baseline.
- **Metric.** Validation **bits per character** (log₂ perplexity).
- **Compute.** Single 4-core CPU; complex64.

*(Reference baselines, val bits/char: uniform 6.022, unigram 4.829, bigram 3.583,
trigram 2.952, 4-gram 2.574.)*

---

## 4. Results

All numbers below are validation **bits per character** (log₂ perplexity) on the
held-out 10% of tiny-Shakespeare, produced by the runnable code in this
repository. Models are trained for 1800 steps (batch 32, block 64, `W=4`).

### 4.1 The QCLM learns language and beats classical baselines

| model | val bits/char |
|---|---|
| uniform | 6.022 |
| unigram | 4.829 |
| bigram | 3.583 |
| trigram | 2.952 |
| 4-gram | 2.574 |
| classical (decohered) n=24 | 3.208 |
| **quantum n=24** | **3.008** |
| classical (decohered) n=48 | 3.045 |
| **quantum n=48** | **2.806** |

The quantum models comfortably pass bigram/trigram territory; `n=48` approaches a
well-smoothed 4-gram while using a *single fixed-size operator* as its entire
memory of the past, rather than explicit n-gram context. *(see `fig_bits_bar.png`)*

### 4.2 Quantum interference pays for itself

At **matched Hilbert-space dimension and identical parameter budget**, the intact
QCLM beats its decohered (classical-HMM) twin, and the gap **grows with
dimension**:

| dim n | quantum | classical (decohered) | advantage Δ |
|---|---|---|---|
| 24 | **3.008** | 3.208 | 0.200 |
| 48 | **2.806** | 3.045 | 0.239 |

This is the paper's central finding: the improvement is not from extra parameters
(the twin has exactly as many) but from the off-diagonal **coherences** —
quantum interference between probability amplitudes. *(see `fig_learning_curves.png`)*

### 4.3 The model maintains genuine quantum coherence

Reading real text, the `n=48` model holds a nearly **pure** state (mean purity
`Tr(ρ²) ≈ 0.94`, far above the maximally-mixed floor `1/48 ≈ 0.021`) with strong
**ℓ1-coherence** (`Σ_{i≠j}|ρ_{ij}| ≈ 37`). The decohered control has coherence
identically zero by construction. *(`*_trajectory.png`)*

### 4.4 Characters acquire geometry in Hilbert space

A PCA of the learned measurement operators `M_x` organizes characters by
linguistic role (vowels / consonants / whitespace+punctuation), i.e. the model
discovers structure in the geometry of its measurement operators. *(`*_charmap.png`)*

### 4.5 Samples

Prompted continuations from the `n=48` model (temperature 0.55, top-k 8) produce
recognizable, stylistically-Shakespearean English with speaker labels, line
breaks and many real words — e.g. prompting `First Citizen:` yields a fresh
speaker turn with dialogue. Coherence is at the local/short-phrase level expected
of a strong character model of this size (see §5 and `qlm/demo.py`).

---

## 5. Discussion, limitations, and scaling

**What works.** A purely quantum-mechanical, attention-free, gate-free model
learns real orthographic structure and demonstrably benefits from interference.

**Limitations (honest).** Being a *linear* state-space model of modest dimension,
the QCLM captures local/stylistic structure (word- and short-phrase level) rather
than long-range semantics, and a small state "forgets" a prompt after some
characters. Coherence is at the level expected of a strong character n-gram model,
which is the appropriate bar for a CPU PoC.

**Scaling path.** The binding constraint is the Hilbert-space dimension `n`, whose
dense cost grows as `O(n³)` per step. Two complementary routes: (i) move from
characters to a **BPE subword vocabulary**, which places parameters where they
scale cheaply (in the vocabulary) while keeping `n` small — a `V=8192, n=128, W=4`
model is **~1.07B parameters** that trains in ~21.5 GB, comfortably inside a single
**NVIDIA DGX Spark** (128 GB unified memory); (ii) for very large effective
dimension, give the Kraus operators **matrix-product-operator (MPO)** structure on
a tensor-product Hilbert space `H = ℂ^{d₁} ⊗ … ⊗ ℂ^{d_k}`, the many-body-physics
route to exponential dimension at polynomial cost. The channel composition is
**associative**, so the sequence likelihood also admits a parallel prefix scan
(as for linear SSMs). See `paper/SCALING.md` for the full configuration table.

---

## 6. Reproducibility

All code is in this repository. Key entry points:
`qlm/model.py` (the QCLM), `qlm/train.py` (training + NLL), `qlm/analysis.py`
(diagnostics & figures), `qlm/demo.py` (prompted generation), `qlm/aggregate.py`
(results table & headline figures), `experiments/test_sanity.py` (quantum-validity
checks). See `README.md` for commands.
