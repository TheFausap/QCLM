"""Sanity checks: verify the model is a valid quantum channel / probability model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from qlm.model import QuantumChannelLM

torch.manual_seed(0)
V, n, W = 7, 6, 3
m = QuantumChannelLM(vocab_size=V, dim=n, kraus=W)

K = m.kraus_operators()
M = m.povm(K)

# 1. POVM completeness: sum_x M_x = I
Msum = M.sum(0)
I = torch.eye(n, dtype=Msum.dtype)
print("[1] ||sum_x M_x - I|| =", (Msum - I).abs().max().item())

# 2. Each M_x Hermitian PSD
herm_err = max((Mx - Mx.conj().T).abs().max().item() for Mx in M)
evmin = min(torch.linalg.eigvalsh(0.5*(Mx+Mx.conj().T)).min().item() for Mx in M)
print("[2] max Hermiticity err =", herm_err, " min eigenvalue of M_x =", evmin)

# 3. rho0 valid density matrix
rho0 = m.initial_rho(1)[0]
print("[3] Tr(rho0) =", torch.trace(rho0).real.item(),
      " min eig =", torch.linalg.eigvalsh(0.5*(rho0+rho0.conj().T)).min().item())

# 4. Probabilities over vocab sum to 1 at step 0
pall = torch.einsum('ij,xji->x', rho0, M).real
print("[4] sum_x p(x) at step0 =", pall.sum().item(), " (should be 1)")

# 5. After an update rho stays trace-1 and PSD
tokens = torch.randint(0, V, (2, 10))
out = m.forward(tokens, return_probs=True)
probs = out['probs']  # (B, L, V)
print("[5] per-step prob sums (should ~1):", probs.sum(-1).mean().item())
print("    loss (nats):", out['loss'].item(), " => bits/char:", out['loss'].item()/0.6931)

# 6. Gradient flow
out['loss'].backward()
gnorm = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
print("[6] total grad L1 =", gnorm, " (should be > 0)")

# 7. Decohered forward runs and gives valid probs
outd = m.forward(tokens, decohere=True, return_probs=True)
print("[7] decohered loss:", outd['loss'].item(),
      " prob sums:", outd['probs'].sum(-1).mean().item())

# 8. Generation works
ids = m.generate(prompt_ids=[1,2,3], n_new=15, seed=0)
print("[8] generated ids len:", len(ids), "->", ids)
print("ALL SANITY CHECKS RAN")
