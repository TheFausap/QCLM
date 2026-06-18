"""OTOC / mutual-information diagnostic for the unitary-memory model.

Motivation (Singh et al., "Quantum machine learning for complex systems",
Academia Quantum 2026, Sec. 3.2): the imaginary part of the out-of-time-order
correlator is proportional to a subsystem covariance eta, and (eta, I) -- with I
the quantum mutual information between subsystems -- are constrained to a convex
region with analytic bounds. Trained quantum learners trace trajectories in this
(I, eta) plane; how much quantum correlation a model develops, and whether it
saturates a bound, is a PRINCIPLED measure of how much the quantum structure is
doing -- the quantitative version of our decohere ablation (which only asks "does
VAL change"). For non-RBM models the paper notes these must be estimated
numerically; that is what this script does directly from the model's quantum
state.

We treat the unitary memory state psi in C^n as a pure state on a bipartite
Hilbert space C^{nA} x C^{nB} (n = nA*nB) and, as the model reads real text,
measure along the sequence:

  - ENTANGLEMENT ENTROPY  S(rho_A) across the cut  (0 = product, ln(nA) = maximal)
  - MUTUAL INFORMATION    I(A:B) = S(rho_A)+S(rho_B)-S(rho_AB)  (rho_AB pure -> S=0,
                          so I = 2 S(rho_A) for a pure global state)
  - ETA (correlation)     |Cov(Z_A, Z_B)| from computational-basis Z observables on
                          each subsystem -- the OTOC-accessible pairwise correlation
  - PHASE COHERENCE       l1 off-diagonal coherence of rho_A (the resource decohere
                          destroys)

Run intact vs the phase-zeroed control on the SAME trained model: the gap
quantifies the quantum correlation the phase carries. If the intact state builds
substantial entanglement/MI/coherence while the decohered control sits near zero,
the model is genuinely using quantum structure -- and the SIZE of that quantity,
versus the bounded VAL advantage, explains WHY the advantage was bounded.

  python qlm/otoc_diagnostic.py --ckpt artifacts/uce_int_best.pt --nA 16
"""
from __future__ import annotations
import os, sys, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from qlm.data import CharTokenizer, load_text
from qlm.unitary_hybrid import UnitaryMemoryHybridLM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def von_neumann(rho, eps=1e-12):
    rho = 0.5 * (rho + rho.conj().mH)
    ev = torch.linalg.eigvalsh(rho).clamp_min(eps)
    ev = ev / ev.sum(-1, keepdim=True)
    return -(ev * torch.log(ev)).sum(-1)            # nats


def bipartite_diagnostics(psi, nA, nB):
    """psi: (..., n) complex pure state, n=nA*nB. Returns dict of mean diagnostics."""
    psi = psi / (psi.norm(dim=-1, keepdim=True) + 1e-12)
    shp = psi.shape[:-1]
    M = psi.reshape(*shp, nA, nB)                   # Schmidt matrix
    rhoA = M @ M.conj().mH                          # (...,nA,nA) reduced on A
    rhoB = M.conj().mH @ M                          # (...,nB,nB) reduced on B
    SA = von_neumann(rhoA)                          # entanglement entropy
    SB = von_neumann(rhoB)
    # global state is pure -> S(rho_AB)=0, so I(A:B) = SA + SB
    I = SA + SB
    # eta: |Cov(Z_A, Z_B)| with Z = diag(+1,-1,+1,...) computational-basis parity
    zA = torch.tensor([1.0 if (i % 2 == 0) else -1.0 for i in range(nA)],
                      device=psi.device)
    zB = torch.tensor([1.0 if (i % 2 == 0) else -1.0 for i in range(nB)],
                      device=psi.device)
    pA = rhoA.diagonal(dim1=-2, dim2=-1).real       # marginal on A
    pB = rhoB.diagonal(dim1=-2, dim2=-1).real
    # joint distribution over (a,b) = |M_ab|^2
    pjoint = M.abs().pow(2)                         # (...,nA,nB)
    EzA = (pA * zA).sum(-1)
    EzB = (pB * zB).sum(-1)
    EzAzB = (pjoint * zA[:, None] * zB[None, :]).sum((-2, -1))
    eta = (EzAzB - EzA * EzB).abs()
    # l1 phase coherence of rhoA (off-diagonal magnitude)
    offmask = ~torch.eye(nA, dtype=torch.bool, device=psi.device)
    coh = (rhoA.abs() * offmask).sum((-2, -1))
    return dict(S_ent=SA.mean().item(), MI=I.mean().item(),
                eta=eta.mean().item(), coherence=coh.mean().item(),
                S_max=math.log(nA))


def load(ckpt, vocab, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    a = ck["args"]
    m = UnitaryMemoryHybridLM(vocab, n_layers=a["n_layers"], mem_dim=a["mem_dim"],
                              d_model=a["d_model"], n_heads=a["n_heads"],
                              block_size=a["block"], dropout=0.0,
                              complex_embed=a.get("complex_embed", False)).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, a["mem_dim"], ck.get("val_bits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "tinyshakespeare.txt"))
    ap.add_argument("--nA", type=int, default=0,
                    help="size of subsystem A (n=nA*nB). Default: factor n near sqrt.")
    ap.add_argument("--n_seq", type=int, default=16, help="number of text windows")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else (args.device if args.device != "auto" else "cpu"))
    corpus = load_text(args.data)
    tok = CharTokenizer(corpus)
    model, n, val_bits = load(args.ckpt, tok.vocab_size, dev)

    # choose bipartition
    nA = args.nA
    if nA <= 0:
        nA = int(round(math.sqrt(n)))
        while n % nA != 0:
            nA -= 1
    assert n % nA == 0, f"nA={nA} must divide mem_dim n={n}"
    nB = n // nA
    print(f"ckpt {os.path.basename(args.ckpt)} | mem_dim n={n} | bipartition {nA}x{nB} "
          f"| val_bits {val_bits}")

    # sample real-text windows
    ids = torch.tensor(tok.encode(corpus[:200000]), device=dev)
    starts = torch.linspace(0, len(ids) - args.seq_len - 1, args.n_seq).long()
    batch = torch.stack([ids[s:s + args.seq_len] for s in starts])  # (n_seq, seq_len)

    def run(decohere):
        traj = model.memory.state_trajectory(batch, decohere=decohere)  # (B,L,n)
        # diagnostics on the SECOND HALF of each sequence (after context built up)
        psi = traj[:, args.seq_len // 2:, :].reshape(-1, n)
        return bipartite_diagnostics(psi, nA, nB)

    intact = run(False)
    dec = run(True)
    print(f"\n=== quantum-correlation diagnostics (mean over real-text windows) ===")
    print(f"{'quantity':>14} | {'intact':>9} | {'decohere':>9} | {'max':>7}")
    print(f"{'-'*14}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}")
    print(f"{'entanglement S':>14} | {intact['S_ent']:9.4f} | {dec['S_ent']:9.4f} | {intact['S_max']:7.4f}")
    print(f"{'mutual info I':>14} | {intact['MI']:9.4f} | {dec['MI']:9.4f} | {2*intact['S_max']:7.4f}")
    print(f"{'eta (corr)':>14} | {intact['eta']:9.4f} | {dec['eta']:9.4f} |")
    print(f"{'l1 coherence':>14} | {intact['coherence']:9.4f} | {dec['coherence']:9.4f} |")
    sat = intact['S_ent'] / intact['S_max'] if intact['S_max'] > 0 else 0.0
    print(f"\nentanglement saturation S/S_max = {sat:.3f}  "
          f"(near 1 => maximally entangled across the cut; near 0 => product-like)")
    print(f"\nInterpretation:")
    print(f"  - MI / entanglement S measure the CORRELATION STRUCTURE the state")
    print(f"    develops across the cut (the paper's I in the (I,eta) plane). A real")
    print(f"    (decohered) state can also be entangled, so the intact-vs-decohere MI")
    print(f"    difference is NOT purely a phase effect -- read MI as 'how much")
    print(f"    correlation the model uses', and compare its magnitude to the bounded")
    print(f"    VAL advantage to see whether the model is correlation-limited.")
    print(f"  - l1 COHERENCE is the PHASE-specific resource (off-diagonal magnitude")
    print(f"    of rho_A); decohere zeros phase, so coherence_intact - coherence_dec")
    print(f"    isolates the genuinely quantum (interference) part.")
    print(f"  intact MI = {intact['MI']:.4f} (of max {2*intact['S_max']:.4f}, "
          f"saturation {intact['MI']/(2*intact['S_max']):.2f})")
    print(f"  phase coherence: intact {intact['coherence']:.4f} vs decohere "
          f"{dec['coherence']:.4f}")


if __name__ == "__main__":
    main()
