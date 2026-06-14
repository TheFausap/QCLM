"""Train the rung-2 hybrid on char-level tiny-Shakespeare and PROBE whether it
holds a prompt -- the capability the pure HQMM lacks.

Runs fast (CPU minutes / GPU seconds) at the default tiny config. Trains both
the intact model and, with --decohere, the matched classical ablation, so you
can read the quantum advantage directly. After training it prints a prompted
continuation: the thing to watch is whether the sample STAYS ON the prompt's
subject/structure across a sentence, not just whether the words are English.

Usage:
  python qlm/train_hybrid.py --persist fresh  --steps 1500
  python qlm/train_hybrid.py --persist thread --steps 1500
  python qlm/train_hybrid.py --persist fresh  --steps 1500 --decohere   # ablation
"""
from __future__ import annotations
import os, sys, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from qlm.data import CharTokenizer, CharDataset, load_text
from qlm.hybrid import HybridLM

LN2 = math.log(2.0)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def evaluate(model, ds, batch, n_batches, device, decohere):
    model.eval()
    tot, ntok = 0.0, 0
    for _ in range(n_batches):
        seqs = ds.sample_batch(batch).to(device)
        x, y = seqs[:, :-1], seqs[:, 1:]
        _, loss = model(x, y, decohere=decohere)
        tot += loss.item() * y.numel(); ntok += y.numel()
    model.train()
    return (tot / ntok) / LN2  # bits/char


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "tinyshakespeare.txt"))
    ap.add_argument("--persist", choices=["fresh", "thread"], default="fresh")
    ap.add_argument("--attn", choices=["dot", "born"], default="dot",
                    help="attention routing kernel. 'dot' = ordinary dot-product "
                         "(rung 2). 'born' = Born-rule quantum attention, scores = "
                         "|<psi_i|psi_j>|^2 (rung 3); --decohere then removes the "
                         "interference cross-terms from the ROUTING, isolating the "
                         "quantum contribution to long-range mixing.")
    ap.add_argument("--n_q", type=int, default=16,
                    help="per-head pure-state dimension for Born attention")
    ap.add_argument("--bipartite", action="store_true",
                    help="rung 4: quantum state on C^d1 (x) C^d2 (n=d1*d2). Compare "
                         "--entangle (general Kraus) vs --factorize (product Kraus "
                         "{K1_a (x) K2_b}, cannot entangle) to isolate the value of "
                         "entanglement. Logs negativity (the entanglement measure).")
    ap.add_argument("--d1", type=int, default=8)
    ap.add_argument("--d2", type=int, default=8)
    ap.add_argument("--Wa", type=int, default=2)
    ap.add_argument("--Wb", type=int, default=2)
    ap.add_argument("--factorize", action="store_true",
                    help="bipartite: use PRODUCT Kraus (cannot entangle). Omit for "
                         "the ENTANGLING (general Kraus) config.")
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n", type=int, default=32, help="Hilbert dim per quantum sublayer")
    ap.add_argument("--kraus", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--decohere", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--no_compile", action="store_true",
                    help="disable torch.compile on the quantum scan (use eager). "
                         "At small n the compile overhead/recompiles can cost more "
                         "than they save -- A/B this against the default on your GPU.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else (args.device if args.device != "auto" else "cpu"))
    text = load_text(args.data)
    tok = CharTokenizer(text)
    train_ds = CharDataset(text, tok, block_size=args.block, split="train", seed=args.seed)
    val_ds = CharDataset(text, tok, block_size=args.block, split="val", seed=args.seed + 1)

    if args.bipartite:
        model = HybridLM(tok.vocab_size, n_layers=args.n_layers, d_model=args.d_model,
                         n_heads=args.n_heads, block_size=args.block, persist=args.persist,
                         attn_kind=args.attn, n_q=args.n_q,
                         bipartite=True, d1=args.d1, d2=args.d2,
                         Wa=args.Wa, Wb=args.Wb, factorize=args.factorize).to(dev)
    else:
        model = HybridLM(tok.vocab_size, n_layers=args.n_layers, n=args.n, W=args.kraus,
                         d_model=args.d_model, n_heads=args.n_heads,
                         block_size=args.block, persist=args.persist,
                         attn_kind=args.attn, n_q=args.n_q).to(dev)
    if args.no_compile:
        from qlm.hybrid import QuantumChannelSublayer
        for mod in model.modules():
            if isinstance(mod, QuantumChannelSublayer):
                mod.compile_scan = False
    print(f"device {dev} | persist={args.persist} | attn={args.attn} "
          f"| params {model.num_params()/1e6:.2f}M "
          f"| decohere={args.decohere} | compile={'off' if args.no_compile else 'on'}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    model.train()
    t0 = time.time(); run, rn = 0.0, 0
    for step in range(args.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        seqs = train_ds.sample_batch(args.batch).to(dev)
        x, y = seqs[:, :-1], seqs[:, 1:]
        _, loss = model(x, y, decohere=args.decohere)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        run += loss.item() * y.numel(); rn += y.numel()
        if (step + 1) % 50 == 0:
            tr = (run / rn) / LN2; run, rn = 0.0, 0
            sp = (step + 1) * args.batch * args.block / (time.time() - t0)
            print(f"step {step+1:5d} | train {tr:.3f} b/char | lr {lr_at(step):.1e} | "
                  f"gnorm {gn:.2f} | {sp:.0f} tok/s")
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            vb = evaluate(model, val_ds, args.batch, 10, dev, args.decohere)
            print(f"   >>> step {step+1}: VAL {vb:.3f} bits/char")

    # ── coherence probe: continue the prompt, watch for on-topic structure ──
    model.eval()
    if args.bipartite:
        # measure entanglement actually used: mean negativity of layer-0 final rho
        from qlm.hybrid import negativity, BipartiteChannelSublayer
        seqs = val_ds.sample_batch(args.batch).to(dev)
        with torch.no_grad():
            blk0 = model.blocks[0].q
            _, rho = blk0(seqs[:, :-1])
            neg = negativity(rho.real, rho.imag, args.d1, args.d2)
        cfg = "PRODUCT (cannot entangle)" if args.factorize else "ENTANGLING (general)"
        print(f"\n[entanglement] config={cfg} | negativity mean={neg.mean().item():.4f} "
              f"max={neg.max().item():.4f}")
        print(f"   (product config should be ~0; entangling >0 IFF the model learned to use it)")
    ids = torch.tensor([tok.encode(args.prompt)], device=dev)
    out = model.generate(ids, n_new=300, temperature=0.7, top_k=20,
                         decohere=args.decohere)
    print("\n===== PROMPT =====")
    print(repr(args.prompt))
    print("===== CONTINUATION =====")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
