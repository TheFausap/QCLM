"""Rung 2 of the architecture ladder: a SERIES HYBRID of quantum-channel
sublayers and self-attention, the smallest design that could plausibly hold a
prompt across a sentence (the thing the pure HQMM provably cannot).

Why this shape
--------------
A pure HQMM is a Markov system: its correlations decay exponentially with
distance, so a fixed-size density matrix forgets the prompt after a few tokens.
Attention has no such decay -- any position can read any earlier position
directly. So we keep the quantum channel as the *local mixer* (interference-rich
per-token feature extraction) and add attention as the *long-range router*. Each
block is:

    x -> [quantum-channel sublayer] -> +residual -> [self-attention] -> +residual -> [MLP] -> +residual

The quantum sublayer carries a density matrix rho in C^{n x n} and, at each
position, (i) updates rho through the per-token Kraus channel and (ii) reads out
a feature vector from rho (real+imag parts -> includes off-diagonal COHERENCES).
The readout is what attention consumes. This is the interface where quantum-ness
either pays or doesn't -- and the decoherence ablation (zero rho's off-diagonals)
turns the readout classical, giving a clean matched-capacity control at EVERY
layer, exactly as in the pure-HQMM paper.

Two persistence variants (--persist):
  - "thread": ONE rho evolves across the whole sequence within a layer, seeded
    from the previous layer's final rho. Closest to "quantum recurrence with
    attention as a side channel" (the original vision).
  - "fresh": each layer seeds rho from its own learned |psi0> and evolves it over
    that layer's (attention-mixed) input. Closer to "transformer whose mixer is a
    quantum channel". Usually easier to optimise.

This file is deliberately self-contained and SMALL -- it is a probe, not the 2B
run. Train it on char-level tiny-Shakespeare in minutes and watch whether it can
continue a prompt coherently. Only the winning configuration gets scaled.

Kraus operators reuse the validated per-token... NO -- reuse the GLOBAL-POVM
projection from qlm.model (_KrausProjection): each quantum sublayer owns a small
vocabulary-indexed... actually here tokens index Kraus ops exactly as in the base
model, so we reuse _KrausProjection unchanged (global completeness sum_x M_x = I).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from qlm.model import _KrausProjection, _cplx_mm_block, _cplx_mm_dag_block


# ───────────────────────── quantum-channel sublayer ─────────────────────────
def _qfeat_scan_impl(Kr_seq, Ki_seq, rho_r, rho_i, eps, decohere):
    """Per-token Kraus scan emitting the rho readout features at EVERY position.

    Kr_seq/Ki_seq: (B, L, W, n, n) float32 (pre-gathered per-token Kraus ops).
    Returns feats (B, L, 2*n*n): [Re(rho).flatten(), Im(rho).flatten()] per step,
    plus the final (rho_r, rho_i). No .item()/Python-scalar ops, so torch.compile
    can fuse the unrolled loop into a few Triton kernels (same trick as the base
    model's _SCAN_CHUNK, but emitting features instead of NLL).
    """
    B, L = Kr_seq.shape[0], Kr_seq.shape[1]
    n = rho_r.shape[-1]
    feats = []
    for t in range(L):
        Kr_x = Kr_seq[:, t]; Ki_x = Ki_seq[:, t]            # (B,W,n,n)
        Krho_r, Krho_i = _cplx_mm_block(Kr_x, Ki_x,
                                        rho_r.unsqueeze(1), rho_i.unsqueeze(1),
                                        torch.float32)
        E_r, E_i = _cplx_mm_dag_block(Krho_r, Krho_i, Kr_x, Ki_x, torch.float32)
        E_r = E_r.sum(1); E_i = E_i.sum(1)                  # (B,n,n)
        p = E_r.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(eps)
        rho_r = E_r / p[:, None, None]
        rho_i = E_i / p[:, None, None]
        if decohere:
            d = rho_r.diagonal(dim1=-2, dim2=-1)
            rho_r = torch.diag_embed(d)
            rho_i = torch.zeros_like(rho_r)
            rho_r = rho_r / d.sum(-1).clamp_min(eps)[:, None, None]
        feats.append(torch.cat([rho_r.reshape(B, -1), rho_i.reshape(B, -1)], -1))
    return torch.stack(feats, 1), rho_r, rho_i   # (B,L,2*n*n)


_QFEAT_SCAN = torch.compile(_qfeat_scan_impl, fullgraph=True, dynamic=False)


class QuantumChannelSublayer(nn.Module):
    """Per-token Kraus channel producing a per-position readout of rho.

    Kraus ops are indexed by the input token id (like the base model) so the
    parameter count is 2*V*W*n^2 -- but here n is SMALL (depth carries capacity).
    Readout = [Re(rho).flatten(), Im(rho).flatten()] -> Linear -> d_model.
    With decohere=True the off-diagonals are zeroed every step, so the readout
    is purely classical (diagonal) -> matched-capacity ablation.
    """

    def __init__(self, vocab_size: int, n: int, W: int, d_model: int,
                 rdtype=torch.float32, compile_scan: bool = True):
        super().__init__()
        self.V, self.n, self.W = vocab_size, n, W
        self.rdtype = rdtype
        self.compile_scan = compile_scan
        scale = 1.0 / math.sqrt(vocab_size * W * n)
        K_raw = torch.complex(torch.randn(vocab_size, W, n, n) * scale,
                              torch.randn(vocab_size, W, n, n) * scale)
        with torch.no_grad():
            Vm = K_raw.reshape(vocab_size * W * n, n)
            # initialise on the global isometry manifold (sum_x M_x = I)
            from qlm.model import _inv_sqrt_hermitian
            Ki = (Vm @ _inv_sqrt_hermitian(Vm.mH @ Vm)).reshape(vocab_size, W, n, n)
        self.Kr = nn.Parameter(Ki.real.contiguous())
        self.Ki = nn.Parameter(Ki.imag.contiguous())
        self.psi0_r = nn.Parameter(torch.randn(n) / math.sqrt(n))
        self.psi0_i = nn.Parameter(torch.randn(n) / math.sqrt(n))
        self.readout = nn.Linear(2 * n * n, d_model)
        # Tame the readout: the 2n^2 vectorised-rho features have entries O(1/n)
        # but there are 2n^2 of them, so default init produces large logits and a
        # hot, unstable start (train loss spiking into the hundreds early). Scale
        # the readout weights down so the initial block output is O(1), matching
        # the residual stream, and let training grow it if useful.
        with torch.no_grad():
            self.readout.weight.mul_(1.0 / math.sqrt(2 * n * n))
            self.readout.bias.zero_()

    def kraus(self):
        K = torch.complex(self.Kr, self.Ki)
        return _KrausProjection.apply(K, 1e-6, 1e4)

    def initial_rho(self, B, device, decohere):
        psi = torch.complex(self.psi0_r, self.psi0_i)
        psi = psi / (psi.norm() + 1e-12)
        rho = torch.outer(psi, psi.conj())
        if decohere:
            rho = torch.diag(torch.diagonal(rho).real).to(rho.dtype)
            rho = rho / rho.diagonal().real.sum().clamp_min(1e-12)
        return rho.unsqueeze(0).expand(B, -1, -1).contiguous().to(device)

    def forward(self, tokens, rho0=None, decohere=False):
        """tokens: (B, L) int64. Returns (readout (B,L,d_model), final_rho (B,n,n))."""
        B, L = tokens.shape
        dev = tokens.device
        K = self.kraus()                              # (V,W,n,n)
        Kr_seq = K.real[tokens].contiguous()          # (B,L,W,n,n)
        Ki_seq = K.imag[tokens].contiguous()
        rho = self.initial_rho(B, dev, decohere) if rho0 is None else rho0
        rho_r, rho_i = rho.real.contiguous(), rho.imag.contiguous()
        eps = 1e-6
        scan = _QFEAT_SCAN if (self.compile_scan and dev.type == "cuda") \
            else _qfeat_scan_impl
        feats, rho_r, rho_i = scan(Kr_seq, Ki_seq, rho_r, rho_i, eps, decohere)
        out = self.readout(feats)                     # one (B,L,2n^2)->(B,L,d_model) matmul
        return out, torch.complex(rho_r, rho_i)


# ───────────────────────────── attention + MLP ──────────────────────────────
# On some GPUs (e.g. GB10 / sm121) the cutlass memory-efficient SDPA kernel has
# no build for the chip and aborts with a FATAL kernel-mismatch. Prefer the
# flash and math backends, which do have builds, and exclude mem-efficient.
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]

    def _sdpa(q, k, v, is_causal, dropout_p):
        # set_priority isn't available everywhere; passing a list selects the
        # allowed backends (tried in order) and disables the rest.
        with sdpa_kernel(_SDPA_BACKENDS):
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, dropout_p=dropout_p)
except Exception:  # very old torch: no sdpa_kernel context manager
    def _sdpa(q, k, v, is_causal, dropout_p):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=dropout_p)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = dropout

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, L, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, L, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, L, self.h, D // self.h).transpose(1, 2)
        y = _sdpa(q, k, v, is_causal=True,
                  dropout_p=self.drop if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


class Block(nn.Module):
    """quantum sublayer -> attn -> MLP, each with prenorm + residual."""

    def __init__(self, vocab_size, n, W, d_model, n_heads):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.q = QuantumChannelSublayer(vocab_size, n, W, d_model)
        self.ln_a = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_m = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x, tokens, rho0, decohere):
        # quantum sublayer reads the ORIGINAL tokens (its Kraus ops are token-indexed);
        # the attention/MLP refine the d_model stream. Residual around the q readout.
        q_out, rho_final = self.q(tokens, rho0=rho0, decohere=decohere)
        x = x + q_out
        x = x + self.attn(self.ln_a(x))
        x = x + self.mlp(self.ln_m(x))
        return x, rho_final


class HybridLM(nn.Module):
    def __init__(self, vocab_size, n_layers=4, n=32, W=4, d_model=256,
                 n_heads=8, block_size=256, persist="fresh"):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.persist = persist
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.blocks = nn.ModuleList(
            [Block(vocab_size, n, W, d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.n = n
        # GPT-2-style init: small embedding so the weight-tied logits start at a
        # sane scale (with std~1 residual and std~1 tied head over d_model dims,
        # default init gives logit std ~sqrt(d_model) -> loss in the dozens). Also
        # scale residual-projection layers by 1/sqrt(2*n_layers) for stable depth.
        self.apply(self._init_weights)
        for nm, p in self.named_parameters():
            if nm.endswith("proj.weight") or nm.endswith("mlp.2.weight"):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, decohere=False):
        B, L = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :L]
        rho = None
        for blk in self.blocks:
            # "thread": pass each layer's final rho as the next layer's seed.
            # "fresh": each layer seeds its own psi0 (rho0=None).
            rho0 = rho if (self.persist == "thread") else None
            x, rho = blk(x, idx, rho0, decohere)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=None, decohere=False):
        for _ in range(n_new):
            idx_c = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_c, decohere=decohere)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], 1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
