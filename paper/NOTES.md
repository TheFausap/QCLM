# Research Notes — Quantum Channel Language Model (QCLM)

Running log of decisions, experiments, and findings. Newest insights appended.

## 1. Motivation & design goal
Build a genuinely novel LLM concept grounded in quantum mathematics (Hilbert
spaces, complex amplitudes, superposition, the Born rule, quantum channels),
**not** a transformer (no attention) and **not** an RNN (no gated nonlinear
recurrence). Must be runnable on CPU and produce locally coherent text as a PoC.

## 2. The architecture chosen: a Hidden Quantum Markov Model as a generative LM
State = density matrix `rho` (n x n complex, PSD, trace 1) in a Hilbert space C^n.
Each token x has Kraus operators {K_{x,w}}; channel E_x(rho)=sum_w K rho K^dag,
POVM M_x=sum_w K^dag K. Global isometry constraint sum_{x,w}K^dag K = I makes the
family trace-preserving so {M_x} is a valid measurement (sum_x M_x = I).

Autoregressive Born-rule law:
  p(x_t | x_<t) = Tr(rho_{t-1} M_{x_t})        # measurement
  rho_t = E_{x_t}(rho_{t-1}) / p(x_t)          # post-measurement collapse + renorm

Why this is *not* a transformer/RNN, in one line: it is a complex-valued,
trace-preserving **linear** state-space model whose ONLY nonlinearity is the
Born rule (a quadratic form |amplitude|^2). Spiritually the quantum cousin of
modern SSMs (S4/Mamba cores are also linear recurrences), but with genuine
quantum interference between probability amplitudes.

Novel contributions vs. prior HQMM work (Monras'10, Srinivasan'18 learned HQMMs
for *low-order sequences*): (a) we use it as a full autoregressive **language
model** at scale (vocab x large Hilbert space) trained end-to-end by gradient
descent with a clean isometry parametrization (differentiable polar projection);
(b) a controlled **decoherence ablation** that collapses the model to a classical
HMM of identical dimension, isolating the value of quantum interference;
(c) quantum-state diagnostics (purity, l1-coherence trajectories) and a
Hilbert-space character map showing learned linguistic structure.

## 3. Implementation notes
- Kraus stored as free real/imag params (V,W,n,n); projected to satisfy the
  isometry constraint via V_iso = V (V^dag V)^{-1/2}, inv-sqrt by eigh.
- forward() runs the quantum forward algorithm, predicting every position.
- Gradient clipping needed (division by p(x) in the state update can spike grads).
- complex64, n in {24,48,64}, W=4 Kraus operators. CPU, 4 threads.

## 4. Sanity checks (PASSED)
POVM completeness ||sum M_x - I|| ~ 7e-7; each M_x Hermitian PSD; rho always a
valid density matrix (trace 1, PSD); per-step probs sum to 1; gradients flow.

## 5. Baselines on tiny-shakespeare (char-level, vocab 65), val bits/char
uniform 6.022 | unigram 4.829 | bigram 3.583 | trigram 2.952 | 4-gram 2.574.
(The most important baseline is the *decohered* model = classical HMM of same n.)

## 6. Results so far
- Smoke (n=48, 300 steps, ~3 min): 4.67 -> 2.886 val bits/char. Already beats
  bigram, near trigram. Samples show real words, speaker labels (CARDTK:, POLIO:),
  line structure.
- Quantum diagnostics on smoke model: mean purity ~0.85 (genuinely near-pure,
  coherent state; maximally mixed would be 1/48=0.021), mean l1-coherence ~35.8
  (strongly nonzero => the model exploits superposition; a classical model is 0).
- n=24 quantum plateaus ~3.04 val bits/char (small state dim limits capacity).
  => running matched quantum-vs-decohered sweep at n in {24,48}.

## 7. Observations / open questions
- Linear-state model => repetition ("that that the the") at low n; larger n and
  more training reduce it. Coherence is "local LM" quality (word/short-phrase),
  appropriate for a PoC.

## 8. FINAL PoC RESULTS (complete matrix)
val bits/char on tiny-shakespeare (1800 steps, W=4, block 64, batch 32):
            quantum   classical(decohered)   advantage
  n=24       3.008          3.208              0.200
  n=48       2.806          3.039              0.234
External refs: bigram 3.583, trigram 2.952, 4-gram 2.574.
=> Quantum beats matched-dimension classical HMM; advantage GROWS with n. The
   only difference between the two is the off-diagonal coherences => interference
   is doing real work, not parameter count.
Diagnostics (n=48): mean purity 0.94 (near-pure, coherent), mean l1-coherence ~37
   (classical = 0). Char Hilbert-map separates vowels/consonants/whitespace.
Figures: fig_learning_curves.png, fig_bits_bar.png, q_d48_trajectory.png,
   q_d48_charmap.png. Samples: artifacts/samples.txt.

## 9. DGX Spark scale-up (user has the hardware)
Verified GB10 envelope: 128GB unified LPDDR5x @ 273 GB/s, ~1 PFLOP FP4 sparse.
QCLM params = 2*V*W*n^2 (verified == code). Char-level can't reach 1B (n~1387,
infeasible). Recommended: BPE vocab. spark-1B = V8192/n128/W4 = 1.07B params,
~21.5GB train mem (fits easily). Full plan + config table in paper/SCALING.md,
calculator in qlm/scaling.py.

## 10. STATUS: PoC COMPLETE.
All code runnable on CPU; sanity checks pass; figures + paper + README + scaling
plan delivered in /mnt/session/outputs/quantum-lm. Next step is the user's call:
wire BPE + mixed precision and run a chosen preset on the DGX Spark.
