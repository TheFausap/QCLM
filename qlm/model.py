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

Fast-kernel mode (fast_kernels=True)
-------------------------------------
On CUDA, complex64 einsum is not natively accelerated by tensor cores. The
per-token state update is replaced with real 2x2 block matmuls in bf16:

    A_block = [[Ar, -Ai], [Ai,  Ar]]   (2n x 2n)
    B_block = [Br; Bi]                  (2n x  n)
    C_block = A_block @ B_block         (2n x  n)  -> Cr = C_block[:n], Ci = C_block[n:]

A complex (n x n) @ (n x n) becomes one (2n x 2n) @ (2n x n) bf16 matmul, which
fully utilises Blackwell/Hopper tensor cores. fp32 master weights are preserved in
Kr/Ki; the optimizer step is unaffected. The isometry projection stays in fp32/
complex64 for numerical stability (called once per forward, not per token).

Additionally, forward() pre-gathers all Kraus operators for the sequence in one
batched index (Kr_seq = Kr_iso[tokens], shape B x L x W x n x n) and delegates
the per-token scan to a torch.compile'd chunk function (_SCAN_CHUNK). Dynamo
unrolls the inner loop for the static chunk size and fuses the block matmuls,
trace, log, and divide across all iterations in the chunk. The Python loop reduces
from L to L/tbptt iterations. The first CUDA call incurs a one-time Triton
compilation cost (~1-5 min for chunk_size=128); subsequent calls are fast.
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


def _cplx_mm_block(
    Ar: torch.Tensor, Ai: torch.Tensor,
    Br: torch.Tensor, Bi: torch.Tensor,
    cdtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C = (Ar+i·Ai) @ (Br+i·Bi) via a single real 2×2-block matmul in cdtype.

    Packs the four real sub-products into one larger matmul, giving tensor cores
    a bigger tile and a single kernel launch instead of four:

        A_block = [[Ar, −Ai], [Ai, Ar]]   shape (..., 2m, 2k)
        B_block = [Br; Bi]                 shape (..., 2k,  n)
        C_block = A_block @ B_block        shape (..., 2m,  n)
          → Cr = C_block[..., :m, :]
            Ci = C_block[..., m:, :]
    """
    m = Ar.shape[-2]
    A_block = torch.cat([torch.cat([Ar, -Ai], dim=-1),
                         torch.cat([Ai,  Ar], dim=-1)], dim=-2).to(cdtype)
    B_block = torch.cat([Br, Bi], dim=-2).to(cdtype)
    C = (A_block @ B_block).to(Ar.dtype)
    return C[..., :m, :], C[..., m:, :]


def _cplx_mm_dag_block(
    Ar: torch.Tensor, Ai: torch.Tensor,
    Br: torch.Tensor, Bi: torch.Tensor,
    cdtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C = (Ar+i·Ai) @ (Br+i·Bi)†  via real 2×2-block matmul in cdtype.

    B† = (Br − i·Bi).mT  ⟹  B†_r = Br.mT,  B†_i = −Bi.mT
    """
    return _cplx_mm_block(Ar, Ai, Br.mT, -Bi.mT, cdtype)


def _scan_chunk_impl(
    Kr_chunk: torch.Tensor,   # (B, C, W, n, n) float32, contiguous
    Ki_chunk: torch.Tensor,   # (B, C, W, n, n) float32, contiguous
    rho_r: torch.Tensor,      # (B, n, n) float32
    rho_i: torch.Tensor,      # (B, n, n) float32
    eps: float,
    decohere: bool,
    cdtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequential scan over one TBPTT chunk. Returns (nll_chunk, rho_r_out, rho_i_out).

    No .item() calls — all tensor arithmetic so torch.compile can fuse the
    entire unrolled loop into one or a few Triton kernels per chunk.
    """
    C = Kr_chunk.shape[1]
    nll_chunk = torch.zeros((), dtype=rho_r.dtype, device=rho_r.device)
    for t in range(C):
        Kr_x = Kr_chunk[:, t]   # (B, W, n, n) — slice, no gather kernel
        Ki_x = Ki_chunk[:, t]
        Krho_r, Krho_i = _cplx_mm_block(
            Kr_x, Ki_x, rho_r.unsqueeze(1), rho_i.unsqueeze(1), cdtype)
        E_r, E_i = _cplx_mm_dag_block(Krho_r, Krho_i, Kr_x, Ki_x, cdtype)
        E_r = E_r.sum(1)   # (B, n, n)
        E_i = E_i.sum(1)
        p_t = E_r.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(eps)  # (B,)
        nll_chunk = nll_chunk - torch.log(p_t).sum()
        rho_r = E_r / p_t[:, None, None]
        rho_i = E_i / p_t[:, None, None]
        if decohere:
            diag = rho_r.diagonal(dim1=-2, dim2=-1)
            rho_r = torch.diag_embed(diag)
            rho_i = torch.zeros_like(rho_r)
            rho_r = rho_r / diag.sum(-1).clamp_min(eps)[:, None, None]
    return nll_chunk, rho_r, rho_i


# Compiled once at module import (lazy: Triton compilation happens on first CUDA call).
# dynamic=False: Dynamo guards on chunk_size and unrolls the inner loop into a single
# monolithic graph per unique chunk_size — typically one for the main chunk (= tbptt)
# and one for the optional partial last chunk. Specialises on bool decohere and dtype
# cdtype automatically, producing at most a small set of cached compiled graphs.
_SCAN_CHUNK = torch.compile(_scan_chunk_impl, fullgraph=True, dynamic=False)


class QuantumChannelLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 48, kraus: int = 4,
                 cdtype: torch.dtype = torch.complex64, init_scale: float = 1.0,
                 fast_kernels: bool = False,
                 compute_dtype: torch.dtype = torch.bfloat16):
        """
        fast_kernels: on CUDA, replaces per-token complex64 einsums with real
                      2×2-block bf16 matmuls AND activates the compiled chunked
                      scan (pre-gathered Kraus ops + torch.compile'd inner loop).
                      No-op on CPU; architecture and numerics are unchanged.
        compute_dtype: dtype used for the block matmuls (default bfloat16).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n = dim
        self.W = kraus
        self.cdtype = cdtype
        rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64
        self.rdtype = rdtype
        self.fast_kernels = fast_kernels
        self.compute_dtype = compute_dtype

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
        K = self.kraus_operators()        # (V, W, n, n) complex64
        # The full POVM (all vocab) is ONLY needed to monitor full distributions;
        # training does not need it (see below), which makes cost independent of V.
        M = self.povm(K) if return_probs else None

        rho = self.initial_rho(B, decohere=decohere)  # (B, n, n)

        eps = 1e-12
        nll_sum = tokens.new_zeros((), dtype=self.rdtype)
        n_tokens = B * L
        prob_log = [] if return_probs else None

        use_fast = self.fast_kernels and tokens.device.type == 'cuda'

        if use_fast:
            # Split projected Kraus into real/imag float32 views (no copy).
            # Gradients flow correctly: .real/.imag are differentiable on complex tensors.
            Kr_iso = K.real   # (V, W, n, n) float32
            Ki_iso = K.imag
            # Initial state split; clone because .real/.imag are strided views.
            rho_r = rho.real.clone()   # (B, n, n) float32
            rho_i = rho.imag.clone()
            cdtype = self.compute_dtype

            # Pre-gather all Kraus operators for the whole sequence at once.
            # One batched index op instead of L per-token gather kernels.
            Kr_seq = Kr_iso[tokens]   # (B, L, W, n, n)
            Ki_seq = Ki_iso[tokens]

            if return_probs:
                # Diagnostic path: needs per-step full-vocab Born readout.
                # Use pre-gathered slices (no per-step gather) but skip compiled scan.
                for t in range(L):
                    pall = torch.einsum('bij,xji->bx',
                                        torch.complex(rho_r, rho_i), M).real
                    pall = pall.clamp_min(eps)
                    prob_log.append(pall.detach() / pall.sum(-1, keepdim=True))
                    Kr_x = Kr_seq[:, t]   # (B, W, n, n) — slice, no gather
                    Ki_x = Ki_seq[:, t]
                    Krho_r, Krho_i = _cplx_mm_block(
                        Kr_x, Ki_x,
                        rho_r.unsqueeze(1), rho_i.unsqueeze(1), cdtype)
                    E_r, E_i = _cplx_mm_dag_block(Krho_r, Krho_i, Kr_x, Ki_x, cdtype)
                    E_r = E_r.sum(1); E_i = E_i.sum(1)
                    p_tgt = E_r.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(eps)
                    nll_sum = nll_sum - torch.log(p_tgt).sum()
                    rho_r = E_r / p_tgt[:, None, None]
                    rho_i = E_i / p_tgt[:, None, None]
                    if decohere:
                        diag = rho_r.diagonal(dim1=-2, dim2=-1)
                        rho_r = torch.diag_embed(diag)
                        rho_i = torch.zeros_like(rho_r)
                        rho_r = rho_r / diag.sum(-1).clamp_min(eps)[:, None, None]
                    if tbptt and (t + 1) % tbptt == 0:
                        rho_r = rho_r.detach()
                        rho_i = rho_i.detach()
            else:
                # Compiled chunk scan path.
                # chunk_size aligns with tbptt so detach falls at chunk boundaries.
                chunk_size = tbptt if tbptt > 0 else L
                for start in range(0, L, chunk_size):
                    end = min(start + chunk_size, L)
                    # .contiguous() required: Kr_seq[:, start:end] is a non-contiguous
                    # view (slicing dim 1 of a 5-D tensor); block matmuls need contiguous.
                    Kr_chunk = Kr_seq[:, start:end].contiguous()
                    Ki_chunk = Ki_seq[:, start:end].contiguous()
                    nll_chunk, rho_r, rho_i = _SCAN_CHUNK(
                        Kr_chunk, Ki_chunk, rho_r, rho_i, eps, decohere, cdtype)
                    # nll_sum + nll_chunk keeps the NLL gradient graph alive;
                    # the state detach below only truncates the state gradient (TBPTT).
                    nll_sum = nll_sum + nll_chunk
                    if tbptt:
                        rho_r = rho_r.detach()
                        rho_i = rho_i.detach()

        else:
            # Original complex64 einsum path (CPU / debugging / no fast_kernels).
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
                E = torch.einsum('bwik,bwlk->bil', Krho, Kx.conj())    # (B, n, n)
                p_tgt = torch.einsum('bii->b', E).real.clamp_min(eps)  # (B,)
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

        use_fast = self.fast_kernels and rho.device.type == 'cuda'
        # Always define rho_r/rho_i so the closure can declare them nonlocal
        rho_r = rho.real.clone() if use_fast else None
        rho_i = rho.imag.clone() if use_fast else None
        cdtype = self.compute_dtype

        def step_update(tok_id: int):
            nonlocal rho, rho_r, rho_i
            if use_fast:
                Kx_r = K.real[tok_id].unsqueeze(0)   # (1, W, n, n)
                Kx_i = K.imag[tok_id].unsqueeze(0)
                Krho_r, Krho_i = _cplx_mm_block(
                    Kx_r, Kx_i,
                    rho_r.unsqueeze(1), rho_i.unsqueeze(1), cdtype)
                E_r, E_i = _cplx_mm_dag_block(Krho_r, Krho_i, Kx_r, Kx_i, cdtype)
                E_r = E_r.sum(1); E_i = E_i.sum(1)
                p = E_r.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(eps)
                rho_r = E_r / p[:, None, None]
                rho_i = E_i / p[:, None, None]
                if decohere:
                    diag = rho_r.diagonal(dim1=-2, dim2=-1)
                    rho_r = torch.diag_embed(diag)
                    rho_i = torch.zeros_like(rho_r)
                    rho_r = rho_r / diag.sum(-1).clamp_min(eps)[:, None, None]
            else:
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
            # Full-vocab readout always uses the complex path (generation bottleneck
            # is the O(V·n²) pall einsum, not the state update).
            rho_cplx = torch.complex(rho_r, rho_i) if use_fast else rho
            pall = torch.einsum('bij,xji->bx', rho_cplx, M).real.squeeze(0)  # (V,)
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
