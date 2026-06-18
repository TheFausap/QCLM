# When Does Quantum Coherence Help Sequence Modeling? A Controlled Study

**A task--resource boundary, characterized at three levels: task performance, generation quality, and the quantum-information content of the model's state.**

---

## Abstract

We ask whether the resources of quantum mechanics --- complex probability
amplitudes, superposition, the Born rule, entanglement, and lossless unitary
evolution --- provide a *scaling* advantage when used as the computational
substrate of an autoregressive sequence model, and if so, *when*. Building on a
quantum-channel language model (QCLM) whose sequential engine is a quantum
channel rather than attention or gated recurrence, we run a controlled
experimental ladder in which each rung isolates a single quantum resource and
each rung carries a *decoherence ablation* that collapses the quantum mechanism
to a dimension-matched classical control. We find a sharp, reproducible boundary.
On a synthetic long-range recall task, lossless unitary phase memory yields an
advantage that **grows with both recall distance and state size**, and is
destroyed by any mid-sequence measurement --- the advantage is precisely the
coherent phase. On natural language, the *same* mechanisms provide advantages
that are real but **bounded (~0.1--0.16 bits) and, for lossless phase memory,
net-negative** --- confirmed across two token-embedding schemes, two metric
families (perplexity and long-range generation coherence), and a quantum-
information diagnostic. That diagnostic, adapted from the out-of-time-order-
correlator / mutual-information framework of Singh et al. (2026), shows the model
develops **~2.9 nats of cut-invariant mutual information** in its memory state
while reading language, **~1 nat of it phase-attributable** --- substantial,
genuinely quantum correlation that the language task nonetheless does not convert
into predictive advantage. The boundary is the result: quantum correlation helps
when the task structurally requires lossless long-range structured memory, and is
*built but unrewarded* when the task rewards real-valued statistical association,
which attention already captures.

---

## 1. Introduction

Modern language models are built from self-attention (transformers) or gated /
linear recurrence (RNNs, S4/Mamba). A different question motivates this work: can
quantum mathematics --- Hilbert spaces, complex amplitudes, the Born rule, quantum
channels, and unitary evolution --- serve *directly* as the engine of an
autoregressive model, and does the "quantumness" pay for itself as the model
scales?

The honest answer we reach is conditional, and the conditionality is the
contribution. Prior "quantum machine learning" results frequently report a
quantum model matching or beating a classical baseline on some task, leaving open
*why*, and whether the advantage would survive scaling or a fair ablation. We
instead fix a single architecture family, vary one quantum resource at a time,
and attach to every comparison a decoherence ablation that yields an exactly
dimension- and parameter-matched classical control. This lets us ask not "does it
work?" but "**does the quantum resource scale, and on which tasks?**" --- and to
answer at three levels of analysis.

### Contributions

1. **A controlled experimental ladder** isolating coherence, quantum routing,
   entanglement, and lossless unitary memory, each with a decoherence ablation,
   tested on both synthetic recall and natural language.
2. **A positive scaling result:** lossless unitary phase memory yields a recall
   advantage that grows with recall distance and with state size, and is
   destroyed by mid-sequence measurement --- isolating coherent phase as the
   mechanism.
3. **A robust negative result for language:** the same resources are bounded
   (~0.1--0.16 bits) or net-negative, confirmed across embeddings, perplexity, and
   long-range generation-coherence metrics.
4. **A quantum-information diagnostic** (adapting the OTOC / mutual-information
   framework of Singh et al., 2026, to a non-RBM architecture) showing the model
   develops substantial, cut-invariant quantum correlation while modeling
   language --- establishing that the bounded advantage is *not* a failure to build
   quantum structure but a property of the task--resource match.

---

## 2. The architecture family

The base object is a Quantum Channel Language Model: the context is carried as a
quantum state, each token defines a quantum operation, and the next-token
distribution is read out by the Born rule. We summarize the variants used as
rungs; full definitions and the global-POVM constraint are in the appendix.

- **Pure QCLM.** State $\rho \in \mathbb{C}^{n\times n}$ (density matrix); each
  token owns Kraus operators defining a channel $E_x(\rho)=\sum_w K\rho K^\dagger$
  and a POVM element $M_x=\sum_w K^\dagger K$, with a single global completeness
  constraint $\sum_x M_x = I$. The autoregressive law is

$$p(x_t \mid x_{<t}) = \operatorname{Tr}(\rho\, M_{x_t}), \qquad \rho_t = E_{x_t}(\rho)/p(x_t).$$

  A complex-valued, trace-preserving *linear* state-space model whose only
  nonlinearity is the Born readout. Decoherence ablation: zero the off-diagonals
  of $\rho$ each step $\rightarrow$ exactly a classical Hidden Markov Model.

- **Hybrid (quantum features / routing + attention).** The pure QCLM is a linear
  state-space model; its correlations decay and it cannot do long-range or
  conversational structure. The hybrid uses a quantum sub-layer as a local
  mixer alongside causal self-attention for nonlinear all-to-all routing.

- **Lossless unitary memory.** A token-conditioned unitary recurrence
  $\psi_t = U(x_t)\,\psi_{t-1}$, with $U(x)$ formed by the **Cayley transform** of a
  learned Hermitian generator (exactly unitary, differentiable, norm-preserving
  over hundreds of steps; chosen over `matrix_exp`, which was non-unitary on
  complex input, and over `eigh`, which was unstable). The state is **never
  measured mid-sequence**; nonlinearity comes from the readout and the attention
  layers. Decoherence ablation: zero the phase in the readout (and, for the
  diagnostic, in the state), collapsing interference.

All quantum sub-layers reuse a validated isometry/unitary parametrization. The
decoherence ablation is the spine of every comparison: it removes exactly the
quantum resource while holding dimension and parameter count fixed.

---

## 3. The positive result: lossless unitary memory scales on recall

We built a synthetic copy task: the model sees a symbol, then must reproduce it
after a controlled *delay* of intervening symbols. This task structurally
requires lossless storage of information across distance --- the property unitary
evolution provides natively. We compare, at matched parameter count (~20K--330K):
the lossless unitary model, its decohered (phase-zeroed) twin, and a GRU of equal
state size.

### 3.1 The advantage grows with recall distance

Final recall accuracy (chance = 0.125), state size n = 32:

| delay | unitary (lossless) | unitary decohered | GRU |
|---|---|---|---|
| 10 | 0.999 | 0.740 | 0.968 |
| 20 | 0.880 | 0.558 | 0.730 |
| 40 | 0.659 | 0.391 | 0.452 |
| 80 | 0.562 | 0.294 | 0.323 |

The unitary advantage over the GRU widens monotonically with delay
(0.03 $\rightarrow$ 0.15 $\rightarrow$ 0.21 $\rightarrow$ 0.24). At short range everyone succeeds; as the recall
distance grows, the GRU decays toward chance while the lossless unitary model
degrades gracefully. The decohered twin is *worst* at every delay --- removing
phase drops the unitary model below even the classical GRU, so the advantage is
unambiguously the coherent phase.

### 3.2 Mid-sequence measurement destroys the advantage

At fixed delay 40, sweeping how often the state is measured mid-sequence
(`measure_every`):

| measure_every | unitary accuracy |
|---|---|
| 0 (lossless) | 0.659 |
| 2 | 0.421 |
| 4 | 0.396 |
| 8 | 0.514 |

Any mid-sequence collapse degrades performance toward the classical floor
(~0.52). There is no coherence-vs-nonlinearity "sweet spot" for recall: the task
wants zero mid-sequence measurement. The nonlinearity must come from elsewhere
(the readout, or --- in a language model --- the attention layers).

### 3.3 The advantage grows with state size

At fixed delay 40, sweeping state size n:

| n | unitary (lossless) | decohered | GRU |
|---|---|---|---|
| 32 | 0.659 | 0.391 | 0.558 |
| 64 | 0.894 | 0.622 | 0.480 |
| 128 | 0.869 | 0.778 | 0.509 |

The unitary model converts additional state capacity into recall --- leaping
0.66 $\rightarrow$ 0.89 from n=32 to n=64 --- while the GRU stays flat (~0.48--0.51) at any
size. A unitary model at n=64 (83K params) beats a GRU at n=128 (300K params).
The advantage grows on a second axis (state size), confirming \S 3.1.

**Conclusion (positive):** lossless unitary phase memory is a quantum resource
whose advantage *grows* with the relevant axes (recall range and state size) and
is destroyed by measurement --- categorically different from the bounded resources
below. On a task structurally built to need lossless structured memory, the
quantum mechanism is the right tool and it scales.

---

## 4. The negative result: language does not reward the same resources

We then tested whether this transfers to natural language (character-level
tiny-Shakespeare; the hybrid integrates the lossless memory track with attention
+ MLP blocks, reading the complex memory state non-destructively into the
residual stream). All comparisons use best-validation checkpoints with dropout
and weight decay (the lossless memory is a high-capacity memorizer and overfits a
small corpus without regularization --- itself indirect evidence the mechanism is
powerful).

### 4.1 Perplexity: the decohered (classical) regime wins

Best validation bits/char (lower is better), 13.5M params, 3000 steps:

| | real embedding | complex (amplitude+phase) embedding |
|---|---|---|
| intact (phase on) | 2.250 | 2.232 |
| **decohered (phase off)** | **2.136** | **2.156** |

The decohered model wins under both embeddings. A learned complex
amplitude+phase embedding --- making phase a first-class property of the *data*,
testing whether a classical lookup interface was starving the mechanism ---
improved the intact model only marginally (2.250 $\rightarrow$ 2.232) and did not flip the
sign. The interface was not the bottleneck. Across the earlier rungs the same
pattern held: quantum coherence as features or as Born-rule routing contributed a
*bounded* ~0.16 bits; entanglement on a tensor-product Hilbert space contributed
a bounded ~0.10 bits that did not grow with the number of entangled subsystems.

### 4.2 Generation coherence: the decohered regime also wins

Perplexity scores local prediction and could miss long-range coherence, so we
measured generation-level coherence over 40 samples, calibrated to the real
corpus (speaker recurrence 0.972, real cast 172 characters):

| config | val bits/char | speaker recurrence | real-cast fraction |
|---|---|---|---|
| complex intact | 2.232 | 0.066 | 0.860 |
| **complex decohere** | 2.156 | **0.352** | **0.951** |
| real intact | 2.250 | 0.079 | 0.740 |
| **real decohere** | 2.136 | **0.250** | **0.915** |

The decohered model is more coherent on every axis --- 4--5$\times$ higher speaker
recurrence (returning to an established cast rather than inventing new names) and
higher real-cast fraction. An initial qualitative impression that intact samples
looked more coherent did not survive quantitative measurement; the coherence
metrics point the *same* way as perplexity. Phase is neutral-to-harmful for
language on both metric families.

---

## 5. The quantum-information level: the resource is built but unrewarded

To establish *why* the advantage is bounded, we adapt the OTOC /
mutual-information diagnostic of Singh et al. (2026, \S 3.2) --- which for RBMs
relates the imaginary part of an out-of-time-order correlator to a subsystem
covariance and bounds it against quantum mutual information --- to our non-RBM
architecture, estimating the quantities numerically from the model's memory
state. We treat the unitary memory state $\psi \in \mathbb{C}^{128}$ as a pure state on a
bipartite Hilbert space and, as the model reads real text, compute the mutual
information $I(A:B)$ and entanglement across the cut.

Mutual information (nats) across two independent bipartitions:

| | 16$\times$8 cut | 8$\times$16 cut |
|---|---|---|
| complex intact | 2.94 | 2.94 |
| complex decohered | 1.93 | 1.93 |
| real intact | 2.88 | 2.88 |
| real decohered | 2.04 | 2.02 |

Two facts matter. First, **MI is invariant to the cut** (stable to two decimals),
so it is a genuine property of the state, not a bipartition artifact --- the
quantity is publishable. Second, the intact state develops **~2.9 nats of mutual
information at ~0.5--0.7 entanglement saturation** --- substantial, far from
product-like --- of which **~1 nat is phase-attributable** (intact minus decohered,
$\approx$1.0 and $\approx$0.86 for the two models).

**The interpretation is the crux.** The model is *not* correlation-limited: it
builds about a nat of phase-driven quantum correlation while reading language. Yet
that correlation yields at most ~0.1 bit of predictive advantage, net-negative
after generalization. The resource is present (measured, ~1 nat), it is used
(half-saturation entanglement), and the language task does not convert it into
predictive value --- in direct contrast to the recall task, where the same kind of
correlation scales with and drives performance.

*(Methodological note: we report MI and entanglement, which are basis-invariant
for a pure-state bipartition. An $\ell_1$-coherence-of-$\rho_A$ measure and a
computational-basis covariance $\eta$ were also computed but did not cleanly isolate
phase --- a real $\rho_A$ can carry large off-diagonals --- and are not used in the
claims. That the data exposed this is the diagnostic working as intended.)*

---

## 6. The boundary, and what it means

Across three levels of analysis the same boundary appears:

- **Task performance:** quantum advantage *scales* on synthetic long-range recall
  (grows with delay and state size); is *bounded-to-negative* on language.
- **Generation quality:** the decohered regime is more coherent on language, not
  less --- confirming the perplexity result on an independent metric family.
- **Quantum information:** the model develops substantial, cut-invariant,
  phase-attributable quantum correlation while modeling language --- so the bounded
  advantage is a property of the *task--resource match*, not a failure to build
  quantum structure.

**The boundary:** quantum correlation helps when the task structurally requires
*lossless long-range structured memory* (exact recall across distance), and is
*built but unrewarded* when the task rewards *real-valued statistical
association*, which attention already captures and which carries its useful
information in similarity rather than phase.

This directly addresses the open problem stated in the QML-for-complex-systems
literature --- "a deeper theoretical understanding of when and why quantum
resources improve learning performance." The successful applications surveyed
there are uniformly on quantum-structured data (many-body wavefunctions,
molecular Hamiltonians, strongly-correlated sampling); natural language is not
such a distribution, and our three-level characterization shows precisely how the
mismatch manifests: the mechanism develops quantum correlation, the task does not
pay for it.

### Implications

For a **quantum-core language model**, the data are unambiguous: scale the
decohered/classical regime (attention and depth do the conversational work; the
quantum core is not the engine for language). For the **quantum machinery built
here** --- a differentiable, hardware-validated, token-conditioned unitary
evolution engine with a clean decoherence ablation --- the natural home is the
domain where the surveyed literature finds genuine advantage: quantum-structured
data (neural-network quantum states, molecular ground states, strongly-correlated
sampling), where the recall-style scaling we demonstrated is expected to apply to
the task it was built for.

---

## 7. Limitations and honest scope

- Language experiments are at character level on a ~1M-character corpus
  (tiny-Shakespeare). A large web corpus (FineWeb) with genuine long-range
  structure and unmemorizable scale *could* alter the language verdict; we judge
  this unlikely given the consistency across embeddings, metrics, and the MI
  diagnostic, but we did not run it, and a falsification test with the decohere
  twin as control remains the honest way to close the question definitively.
- The recall task is synthetic; it demonstrates the scaling mechanism cleanly but
  is not itself a language capability.
- The MI diagnostic is estimated numerically on a pure-state bipartition of the
  memory; it does not use the closed-form OTOC available only for RBMs, and the
  basis-dependent $\eta$ / coherence measures are reported but not relied upon.
- Throughput: the per-token unitary scan is sequential (~1600 tok/s eager at
  block 256; `torch.compile` was unstable on the target GPU). A parallel
  associative prefix-scan (unitary composition is associative) is the known fix
  required before any scale-up.

---

## 8. Reproducibility

Code: `qlm/unitary_hybrid.py` (lossless-memory + attention hybrid, complex-
embedding option), `qlm/train_unitary_hybrid.py` (training with dropout / weight
decay / best-VAL checkpointing), `qlm/recall_probe.py` (synthetic recall sweeps),
`qlm/coherence_metric.py` (generation-coherence metrics), `qlm/otoc_diagnostic.py`
(mutual-information diagnostic), and the base QCLM in `qlm/model.py`. Every
quantum comparison has a `--decohere` control.

---

## Appendix: the decoherence ablation as a controlled instrument

The single methodological commitment that makes this study interpretable is that
each quantum resource is tested against a control that removes *only* that
resource. For the density-matrix QCLM, zeroing off-diagonals collapses $\rho$ to a
classical HMM of identical dimension and parameter count. For the unitary memory,
zeroing the readout/state phase removes interference while preserving amplitudes
and parameters. Because the control is dimension- and parameter-matched, any
performance difference is attributable to the quantum resource itself, not to
capacity --- and any *absence* of difference (as on language) is therefore
meaningful rather than confounded.
