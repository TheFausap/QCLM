"""Rung 5 probe: does LOSSLESS UNITARY quantum memory beat a classical recurrence
at LONG-RANGE RECALL?

The whole-session finding so far: every quantum resource (coherence ~0.16,
entanglement ~0.10) is real but BOUNDED and does not scale, because the quantum
part was always OPTIONAL -- the classical machinery could route around it. The
unitary model is different: its state evolves by token-conditioned UNITARIES
U(x)=Cayley(H(x)), which are LOSSLESS (norm-preserving, reversible). The entire
sequence history is preserved as PHASE in a fixed-size state, with the only
nonlinearity being periodic Born-rule measurement.

This probe tests the cleanest possible claim: lossless phase memory should excel
at COPYING -- reproduce a sequence after a delay. A model whose memory decays
(linear recurrence, or a heavily-measured quantum model) fails at long delay; a
truly lossless memory does not.

Task (copy):
    [S random symbols from alphabet] [DELIM] [pad*delay] -> recall the S symbols
The "range" is the delay between seeing a symbol and having to reproduce it.

Three configs compared at matched state size:
    --model unitary            : Cayley-unitary recurrence + Born readout
    --model unitary --decohere : same, but coherences zeroed at each measurement
                                 (kills the PHASE memory -> isolates its value)
    --model gru                : classical GRU floor (real-valued gated recurrence)

The knob --measure_every controls how often the quantum state is measured
(collapsed). measure_every=0 means measure ONLY at readout (maximally coherent /
lossless); =1 means every step (maximally nonlinear, closest to the HQMM that
plateaus). Sweeping it maps the coherence-vs-nonlinearity tradeoff.
"""
from __future__ import annotations
import os, sys, math, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────── task ─────────────────────────
class CopyTask:
    """Random-sequence copy with a delay. Vocabulary layout:
        0..A-1   : data symbols
        A        : DELIM (start recalling)
        A+1      : PAD/blank
    Input  = [s_1..s_S, DELIM, PAD*delay]
    Target = [ignore over the first S+1,  then s_1..s_S ] during the recall window
    (we score only the recall positions).
    """

    def __init__(self, alphabet=8, seq_len=10, delay=20, seed=0):
        self.A = alphabet
        self.S = seq_len
        self.delay = delay
        self.DELIM = alphabet
        self.PAD = alphabet + 1
        self.vocab = alphabet + 2
        self.rng = torch.Generator().manual_seed(seed)

    def batch(self, B, device):
        S, A = self.S, self.A
        data = torch.randint(0, A, (B, S), generator=self.rng)
        delim = torch.full((B, 1), self.DELIM)
        pad = torch.full((B, self.delay), self.PAD)
        # input sequence: data, DELIM, pad(delay)  -> then the model must emit data
        # we frame as next-token prediction over the recall window:
        # full input  = [data, DELIM, pad*(delay-1), pad]   (length S+1+delay)
        # full target = [*, ..., *, data]                   (recall the S symbols)
        x = torch.cat([data, delim, pad], dim=1)            # (B, S+1+delay)
        # targets: -100 (ignore) everywhere except the LAST S positions, which
        # must reproduce the data in order.
        y = torch.full_like(x, -100)
        y[:, -S:] = data
        return x.to(device), y.to(device)


# ───────────────────────── unitary model ─────────────────────────
def cayley_unitary(A):
    """U = (I - iH/2)(I + iH/2)^{-1}, H = Herm(A). Exactly unitary, differentiable."""
    H = 0.5 * (A + A.conj().transpose(-2, -1))
    n = H.shape[-1]
    I = torch.eye(n, dtype=H.dtype, device=H.device).expand_as(H)
    return torch.linalg.solve(I + 0.5j * H, I - 0.5j * H)


class UnitaryRecallModel(nn.Module):
    """Token-conditioned unitary recurrence with Born-rule readout.

    state |psi> in C^n.  Step: |psi> <- U(x_t)|psi>.  Optionally measure (collapse
    to the diagonal mixture) every `measure_every` steps. Readout at each position:
    p(token) via a learned POVM-like Born layer (project |psi> onto learned
    measurement vectors and square).
    """

    def __init__(self, vocab, n=32, measure_every=0):
        super().__init__()
        self.vocab, self.n, self.measure_every = vocab, n, measure_every
        s = 0.3 / math.sqrt(n)
        # token-conditioned Hermitian generators (real+imag of a general matrix)
        self.Ar = nn.Parameter(torch.randn(vocab, n, n) * s)
        self.Ai = nn.Parameter(torch.randn(vocab, n, n) * s)
        # learned initial state
        self.psi0_r = nn.Parameter(torch.randn(n) / math.sqrt(n))
        self.psi0_i = nn.Parameter(torch.randn(n) / math.sqrt(n))
        # Born readout: M measurement directions -> logits over vocab
        self.readout = nn.Linear(2 * n, vocab)

    def forward(self, x, targets=None, decohere=False):
        B, L = x.shape
        dev = x.device
        A = torch.complex(self.Ar, self.Ai)
        U = cayley_unitary(A)                       # (vocab, n, n) exact unitary
        psi = torch.complex(self.psi0_r, self.psi0_i)
        psi = (psi / psi.norm()).unsqueeze(0).expand(B, -1).contiguous()  # (B,n)
        logits_all = []
        for t in range(L):
            Ut = U[x[:, t]]                          # (B,n,n)
            psi = torch.einsum('bij,bj->bi', Ut, psi)
            psi = psi / (psi.norm(dim=-1, keepdim=True) + 1e-8)
            if decohere or (self.measure_every and (t + 1) % self.measure_every == 0):
                # collapse to diagonal mixture: keep |amplitude|^2, drop phase.
                # (re-embed as a real-amplitude state = sqrt of the probabilities)
                p = psi.abs().pow(2)
                psi = torch.complex(p.sqrt(), torch.zeros_like(p))
            feat = torch.cat([psi.real, psi.imag], dim=-1)   # (B,2n)
            logits_all.append(self.readout(feat))
        logits = torch.stack(logits_all, 1)          # (B,L,vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss


# ───────────────────────── classical floor ─────────────────────────
class GRUFloor(nn.Module):
    def __init__(self, vocab, n=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, n)
        self.gru = nn.GRU(n, 2 * n, batch_first=True)
        self.head = nn.Linear(2 * n, vocab)

    def forward(self, x, targets=None, decohere=False):
        h, _ = self.gru(self.emb(x))
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss


@torch.no_grad()
def recall_accuracy(model, task, B, device, decohere=False):
    x, y = task.batch(B, device)
    logits, _ = model(x, decohere=decohere)
    pred = logits.argmax(-1)
    mask = (y != -100)
    return (pred[mask] == y[mask]).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["unitary", "gru"], default="unitary")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--alphabet", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--delay", type=int, default=20)
    ap.add_argument("--measure_every", type=int, default=0,
                    help="0 = measure only at readout (lossless); k = collapse every k steps")
    ap.add_argument("--decohere", action="store_true",
                    help="kill phase memory (collapse to diagonal every step) -- ablation")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else (args.device if args.device != "auto" else "cpu"))
    task = CopyTask(args.alphabet, args.seq_len, args.delay, seed=args.seed)

    if args.model == "unitary":
        model = UnitaryRecallModel(task.vocab, n=args.n,
                                   measure_every=args.measure_every).to(dev)
    else:
        model = GRUFloor(task.vocab, n=args.n).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"device {dev} | model={args.model} | n={args.n} | params {nparams/1e3:.1f}K "
          f"| seq_len={args.seq_len} delay={args.delay} | measure_every={args.measure_every} "
          f"| decohere={args.decohere}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        x, y = task.batch(args.batch, dev)
        _, loss = model(x, y, decohere=args.decohere)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 200 == 0:
            acc = recall_accuracy(model, task, 256, dev, decohere=args.decohere)
            sp = (step + 1) * args.batch / (time.time() - t0)
            print(f"step {step+1:5d} | loss {loss.item():.4f} | recall acc {acc:.3f} "
                  f"| {sp:.0f} seq/s")
    acc = recall_accuracy(model, task, 1024, dev, decohere=args.decohere)
    chance = 1.0 / args.alphabet
    print(f"\nFINAL recall accuracy: {acc:.3f}  (chance = {chance:.3f}, perfect = 1.000)")
    print(f"  delay={args.delay} symbols between seeing and recalling.")


if __name__ == "__main__":
    main()
