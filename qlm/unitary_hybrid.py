"""Unitary-Memory Hybrid LM: lossless quantum recall track + attention.

The result that motivates this (recall_probe.py sweeps):
  - Lossless UNITARY memory (state evolves by token-conditioned unitaries, NO
    mid-sequence measurement) holds information across long delays where a
    classical recurrence of equal size cannot.
  - The advantage GROWS with recall distance AND with state size, and is
    DESTROYED by any mid-sequence measurement -- i.e. the advantage is precisely
    the coherent phase. (Unlike coherence/entanglement in a dissipative channel,
    which were real but BOUNDED.)

Design constraints taken directly from those sweeps:
  1. The memory track is LOSSLESS: psi <- U(x_t) psi, never measured mid-sequence.
     Nonlinearity comes from the attention/MLP layers, NOT from collapsing psi.
  2. State size n is the lever: bigger n -> more recall capacity (it pays).
  3. The complex state is read NON-DESTRUCTIVELY ([Re psi, Im psi]) into the
     residual stream each layer; reading does not collapse the memory.

Architecture (per layer):
    x <- x + memory_readout(psi_track)        # lossless quantum recall, phase-carrying
    x <- x + attention(x)                     # nonlinear all-to-all routing (rung-2 lesson)
    x <- x + mlp(x)
where psi_track is ONE lossless unitary recurrence over the input tokens, shared
read-points across layers (computed once, read many times).

Decoherence ablation (decohere=True): zero the imaginary part / off-diagonal phase
in the memory readout so attention sees only |amplitude|^2 (classical). This
isolates whether the PHASE memory is load-bearing for language, exactly as the
recall probe isolated it for copying.

Scale knobs: n (memory state dim, the proven lever), n_layers, d_model, block.
Reuses the Cayley unitary (exact, stable, differentiable) from the recall probe.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the validated SDPA-backend wrapper from hybrid.py (sm121-safe)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]

    def _sdpa(q, k, v, is_causal, dropout_p):
        with sdpa_kernel(_SDPA_BACKENDS):
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, dropout_p=dropout_p)
except Exception:
    def _sdpa(q, k, v, is_causal, dropout_p):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=dropout_p)


def cayley_unitary(A):
    """U = (I - iH/2)(I + iH/2)^{-1}, H = Herm(A). Exactly unitary, differentiable."""
    H = 0.5 * (A + A.conj().transpose(-2, -1))
    n = H.shape[-1]
    I = torch.eye(n, dtype=H.dtype, device=H.device).expand_as(H)
    return torch.linalg.solve(I + 0.5j * H, I - 0.5j * H)


class UnitaryMemory(nn.Module):
    """Lossless token-conditioned unitary recurrence. Produces a per-position
    complex state psi_t (B,L,n) by psi_t = U(x_t)...U(x_1) psi_0, NEVER measured.
    Returns the per-position [Re, Im] features (B,L,2n) for non-destructive readout.
    """

    def __init__(self, vocab, n):
        super().__init__()
        self.vocab, self.n = vocab, n
        s = 0.3 / math.sqrt(n)
        self.Ar = nn.Parameter(torch.randn(vocab, n, n) * s)
        self.Ai = nn.Parameter(torch.randn(vocab, n, n) * s)
        self.psi0_r = nn.Parameter(torch.randn(n) / math.sqrt(n))
        self.psi0_i = nn.Parameter(torch.randn(n) / math.sqrt(n))

    def forward(self, x, decohere=False):
        B, L = x.shape
        A = torch.complex(self.Ar, self.Ai)
        U = cayley_unitary(A)                          # (vocab,n,n) exact unitary
        Useq = U[x]                                    # (B,L,n,n)
        psi = torch.complex(self.psi0_r, self.psi0_i)
        psi = (psi / psi.norm()).unsqueeze(0).expand(B, -1).contiguous()
        feats = []
        for t in range(L):
            psi = torch.einsum('bij,bj->bi', Useq[:, t], psi)
            psi = psi / (psi.norm(dim=-1, keepdim=True) + 1e-8)  # numerical renorm only
            if decohere:
                # classical readout: expose only |amplitude|^2, drop phase.
                p = psi.abs().pow(2)
                feats.append(torch.cat([p, torch.zeros_like(p)], -1))
            else:
                feats.append(torch.cat([psi.real, psi.imag], -1))  # phase-carrying
        return torch.stack(feats, 1)                   # (B,L,2n)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, L, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, L, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, L, self.h, D // self.h).transpose(1, 2)
        y = _sdpa(q, k, v, is_causal=True,
                  dropout_p=self.attn_dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    """memory-readout -> attention -> MLP, prenorm + residual.

    The memory readout is projected from the shared lossless track's (B,L,2n)
    features into d_model and added to the residual stream. Each layer has its
    own readout projection (reads the same memory differently), but the memory
    track itself is computed ONCE and shared (lossless, never collapsed).
    """

    def __init__(self, d_model, n_heads, mem_dim, dropout=0.0):
        super().__init__()
        self.mem_proj = nn.Linear(2 * mem_dim, d_model)
        self.mem_drop = nn.Dropout(dropout)        # memory is the overfitting vector
        self.ln_a = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln_m = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x, mem_feats):
        x = x + self.mem_drop(self.mem_proj(mem_feats))  # lossless quantum recall injection
        x = x + self.attn(self.ln_a(x))                  # nonlinear all-to-all routing
        x = x + self.mlp(self.ln_m(x))
        return x


class UnitaryMemoryHybridLM(nn.Module):
    def __init__(self, vocab_size, n_layers=6, mem_dim=128, d_model=384,
                 n_heads=6, block_size=256, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.emb_drop = nn.Dropout(dropout)
        self.memory = UnitaryMemory(vocab_size, mem_dim)   # ONE lossless track
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mem_dim, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        for nm, p in self.named_parameters():
            if nm.endswith("proj.weight") or nm.endswith("mlp.2.weight"):
                torch.nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layers))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, decohere=False):
        B, L = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :L]
        x = self.emb_drop(x)
        mem_feats = self.memory(idx, decohere=decohere)   # (B,L,2*mem_dim), lossless
        for blk in self.blocks:
            x = blk(x, mem_feats)
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
            idx = torch.cat([idx, torch.multinomial(probs, 1)], 1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
