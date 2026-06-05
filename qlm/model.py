"""Quantum Channel Language Model (QCLM).

A language model whose sequential engine is a *quantum channel*, not attention
(transformer) and not a gated nonlinear recurrence (RNN/LSTM/GRU).

Mathematical object
-------------------
The model carries a quantum state as a density matrix rho in C^{n x n}
(Hermitian, positive semidefinite, unit trace) living in a Hilbert space H = C^n.

Each vocabulary token x owns a set of Kraus operators {K_{x,w}}_{w=1..W}, complex
n x n matrices, defining:

    quantum channel : E_x(rho) = sum_w K_{x,w} rho K_{x,w}^dagger
    POVM element    : M_x      = sum_w K_{x,w}^dagger K_{x,w}   (PSD)

A single global isometry constraint  sum_{x,w} K_{x,w}^dag K_{x,w} = I  makes the
collection trace-preserving, so the POVM {M_x} resolves the identity: sum_x M_x = I.

Autoregressive law (Born rule)
------------------------------
    p(x_t | x_{<t}) = Tr(rho_{t-1} M_{x_t})        (Born-rule measurement)
    rho_t           = E_{x_t}(rho_{t-1}) / p(x_t)  (post-measurement update)

This is a homogeneous Hidden Quantum Markov Model used generatively. It is a
complex-valued, trace-preserving LINEAR state-space model with a QUADRATIC
(Born-rule) readout -- the only nonlinearity in the whole model is measurement.

Decoherence ablation
--------------------
Zeroing the off-diagonal entries of rho after every step destroys quantum
coherence and collapses the model to a classical n-state Hidden Markov Model.
Comparing the two isolates the contribution of genuine quantum interference.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


def _inv_sqrt_hermitian(G: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """G^{-1/2} for a Hermitian PSD matrix via eigendecomposition (differentiable)."""
    # symmetrize for numerical Hermiticity
    G = 0.5 * (G + G.conj().transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(G)
    evals = torch.clamp(evals.real, min=eps)
    inv_sqrt = evecs @ torch.diag_embed((evals ** -0.5).to(evecs.dtype)) @ evecs.conj().transpose(-1, -2)
    return inv_sqrt


class QuantumChannelLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 48, kraus: int = 4,
                 cdtype: torch.dtype = torch.complex64, init_scale: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n = dim
        self.W = kraus
        self.cdtype = cdtype
        rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64
        self.rdtype = rdtype

        V = vocab_size
        n = dim
        W = kraus
        # Free (unconstrained) Kraus parameters, stored as real/imag pairs.
        # Shape (V, W, n, n). Projected to satisfy the isometry constraint on the fly.
        scale = init_scale / math.sqrt(V * W * n)
        self.Kr = nn.Parameter(torch.randn(V, W, n, n, dtype=rdtype) * scale)
        self.Ki = nn.Parameter(torch.randn(V, W, n, n, dtype=rdtype) * scale)

        # Learned initial pure state |psi0>.
        self.psi0_r = nn.Parameter(torch.randn(n, dtype=rdtype) / math.sqrt(n))
        self.psi0_i = nn.Parameter(torch.randn(n, dtype=rdtype) / math.sqrt(n))

    # ----- constrained operators -------------------------------------------------
    def kraus_operators(self) -> torch.Tensor:
        """Return isometry-projected Kraus operators K of shape (V, W, n, n) complex.

        Enforces sum_{x,w} K_{x,w}^dag K_{x,w} = I via differentiable polar projection.
        """
        K = torch.complex(self.Kr, self.Ki)  # (V, W, n, n)
        V, W, n, _ = K.shape
        # Stack all (x,w,row) into the tall axis: Vmat (V*W*n, n)
        Vmat = K.reshape(V * W * n, n)
        G = Vmat.conj().transpose(-1, -2) @ Vmat            # (n, n) Hermitian PSD = sum K^dag K
        G_inv_sqrt = _inv_sqrt_hermitian(G)
        Vmat_iso = Vmat @ G_inv_sqrt                        # columns orthonormal => isometry
        K_iso = Vmat_iso.reshape(V, W, n, n)
        return K_iso

    def povm(self, K: torch.Tensor) -> torch.Tensor:
        """POVM elements M_x = sum_w K_{x,w}^dag K_{x,w}; shape (V, n, n) complex."""
        # M[x,a,b] = sum_{w,i} conj(K[x,w,i,a]) K[x,w,i,b]
        return torch.einsum('xwia,xwib->xab', K.conj(), K)

    def initial_rho(self, batch: int, decohere: bool = False) -> torch.Tensor:
        psi = torch.complex(self.psi0_r, self.psi0_i)
        psi = psi / (psi.norm() + 1e-12)
        rho = torch.outer(psi, psi.conj())                  # (n, n)
        if decohere:
            rho = torch.diag(torch.diagonal(rho).real).to(rho.dtype)
            rho = rho / rho.diagonal().real.sum().clamp_min(1e-12)
        return rho.unsqueeze(0).expand(batch, -1, -1).contiguous()

    # ----- core sequence likelihood ---------------------------------------------
    def forward(self, tokens: torch.Tensor, decohere: bool = False,
                return_probs: bool = False, tbptt: int = 0):
        """Compute autoregressive NLL over a batch of token sequences.

        tokens: (B, L) int64. Predicts EVERY position from the preceding quantum
        state (position 0 is predicted from the prior rho0 = no context).

        tbptt: if > 0, detach the quantum state every `tbptt` steps (truncated
        backprop through time) to bound activation memory on long sequences -- the
        forward recurrence is unchanged, only the gradient path is truncated.

        Returns dict with 'loss' (mean NLL, nats), 'nll_sum', 'n_tokens',
        and optionally per-step probability distributions.
        """
        B, L = tokens.shape
        n = self.n
        K = self.kraus_operators()        # (V, W, n, n)
        # The full POVM (all vocab) is ONLY needed to monitor full distributions;
        # training does not need it (see below), which makes cost independent of V.
        M = self.povm(K) if return_probs else None

        rho = self.initial_rho(B, decohere=decohere)  # (B, n, n)

        eps = 1e-12
        nll_sum = tokens.new_zeros((), dtype=self.rdtype)
        n_tokens = B * L
        prob_log = [] if return_probs else None

        for t in range(L):
            tgt = tokens[:, t]            # (B,)
            if return_probs:
                pall = torch.einsum('bij,xji->bx', rho, M).real    # (B, V)
                pall = pall.clamp_min(eps)
                prob_log.append(pall.detach() / pall.sum(-1, keepdim=True))

            # Post-measurement (unnormalized) state with the observed token.
            # Key identity: p(x_t | x_<t) = Tr(rho M_{x_t}) = Tr(E_{x_t}(rho)) = Tr(E).
            # So the next-token probability falls out of the state update for FREE,
            # and we never form the full-vocab readout during training. Cost per
            # step is O(B * W * n^3), INDEPENDENT of vocabulary size V.
            Kx = K[tgt]                    # (B, W, n, n)
            Krho = torch.einsum('bwij,bjk->bwik', Kx, rho)         # (B, W, n, n)
            E = torch.einsum('bwik,bwlk->bil', Krho, Kx.conj())    # (B, n, n) = sum_w K rho K^dag
            p_tgt = torch.einsum('bii->b', E).real.clamp_min(eps)  # (B,) = Tr(E) = Born prob
            nll_sum = nll_sum - torch.log(p_tgt).sum()
            rho = E / p_tgt[:, None, None]
            if decohere:
                diag = torch.diagonal(rho, dim1=-2, dim2=-1).real  # (B, n)
                rho = torch.diag_embed(diag.to(rho.dtype))
                rho = rho / diag.sum(-1).clamp_min(eps)[:, None, None]
            if tbptt and (t + 1) % tbptt == 0:
                rho = rho.detach()

        loss = nll_sum / n_tokens
        out = {"loss": loss, "nll_sum": nll_sum.detach(), "n_tokens": n_tokens}
        if return_probs:
            out["probs"] = torch.stack(prob_log, dim=1)            # (B, L, V)
        return out

    # ----- generation ------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt_ids: list[int] | None, n_new: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 decohere: bool = False, seed: int | None = None) -> list[int]:
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        K = self.kraus_operators()
        M = self.povm(K)
        rho = self.initial_rho(1, decohere=decohere)               # (1, n, n)
        eps = 1e-12
        out_ids: list[int] = []

        def step_update(tok_id: int):
            nonlocal rho
            Kx = K[tok_id].unsqueeze(0)                            # (1, W, n, n)
            Krho = torch.einsum('bwij,bjk->bwik', Kx, rho)
            E = torch.einsum('bwik,bwlk->bil', Krho, Kx.conj())
            p = torch.einsum('bii->b', E).real.clamp_min(eps)
            rho = E / p[:, None, None]
            if decohere:
                diag = torch.diagonal(rho, dim1=-2, dim2=-1).real
                rho = torch.diag_embed(diag.to(rho.dtype))
                rho = rho / diag.sum(-1).clamp_min(eps)[:, None, None]

        # consume the prompt (teacher forcing)
        if prompt_ids:
            for tid in prompt_ids:
                out_ids.append(tid)
                step_update(tid)

        for _ in range(n_new):
            pall = torch.einsum('bij,xji->bx', rho, M).real.squeeze(0)  # (V,)
            pall = pall.clamp_min(eps)
            logits = torch.log(pall) / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.numel()))
                logits[logits < v[-1]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, 1, generator=g).item()
            out_ids.append(tok)
            step_update(tok)
        return out_ids

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
