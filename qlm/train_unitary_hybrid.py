"""Train the Unitary-Memory Hybrid LM on char-level tiny-Shakespeare.

The test: does the lossless unitary memory track (proven to give a recall
advantage that GROWS with range and state size) help LANGUAGE modeling, and is
the PHASE load-bearing? Compare intact vs --decohere at matched params, and read
both VAL bits/char AND the prompted continuation (does it hold context across
lines better than the bounded-coherence rung-2 hybrid?).

  python qlm/train_unitary_hybrid.py --mem_dim 128 --n_layers 6 --steps 2000
  python qlm/train_unitary_hybrid.py --mem_dim 128 --n_layers 6 --steps 2000 --decohere
"""
from __future__ import annotations
import os, sys, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from qlm.data import CharTokenizer, CharDataset, load_text
from qlm.unitary_hybrid import UnitaryMemoryHybridLM

LN2 = math.log(2.0)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def evaluate(model, ds, batch, n_batches, dev, decohere):
    model.eval(); tot, ntok = 0.0, 0
    for _ in range(n_batches):
        seqs = ds.sample_batch(batch).to(dev)
        _, loss = model(seqs[:, :-1], seqs[:, 1:], decohere=decohere)
        tot += loss.item() * (seqs[:, 1:].numel()); ntok += seqs[:, 1:].numel()
    model.train()
    return (tot / ntok) / LN2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "tinyshakespeare.txt"))
    ap.add_argument("--mem_dim", type=int, default=128, help="unitary memory state dim (the lever)")
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="dropout on attention, MLP, memory readout, embeddings. "
                         "The lossless memory has high capacity and overfits small "
                         "corpora; 0.1-0.2 helps on tiny-shakespeare.")
    ap.add_argument("--wd", type=float, default=0.1, help="AdamW weight decay")
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts"),
                    help="dir for best-VAL checkpoint")
    ap.add_argument("--tag", default="uhybrid")
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--decohere", action="store_true")
    ap.add_argument("--complex_embed", action="store_true",
                    help="quantum-native input: learned amplitude+phase token "
                         "embedding (phase first-class in the data). Tests whether "
                         "the classical real-lookup embedding was starving the "
                         "quantum mechanism. --decohere zeros the embedding phase.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="ROMEO:")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else (args.device if args.device != "auto" else "cpu"))
    text = load_text(args.data)
    tok = CharTokenizer(text)
    train_ds = CharDataset(text, tok, block_size=args.block, split="train", seed=args.seed)
    val_ds = CharDataset(text, tok, block_size=args.block, split="val", seed=args.seed + 1)

    model = UnitaryMemoryHybridLM(tok.vocab_size, n_layers=args.n_layers,
                                  mem_dim=args.mem_dim, d_model=args.d_model,
                                  n_heads=args.n_heads, block_size=args.block,
                                  dropout=args.dropout,
                                  complex_embed=args.complex_embed).to(dev)
    print(f"device {dev} | mem_dim={args.mem_dim} | params {model.num_params()/1e6:.2f}M "
          f"| dropout={args.dropout} wd={args.wd} | complex_embed={args.complex_embed} "
          f"| decohere={args.decohere}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.wd)
    os.makedirs(args.out, exist_ok=True)
    best_val = float("inf")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    model.train(); t0 = time.time(); run, rn = 0.0, 0
    for step in range(args.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        seqs = train_ds.sample_batch(args.batch).to(dev)
        _, loss = model(seqs[:, :-1], seqs[:, 1:], decohere=args.decohere)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += loss.item() * seqs[:, 1:].numel(); rn += seqs[:, 1:].numel()
        if (step + 1) % 50 == 0:
            tr = (run / rn) / LN2; run, rn = 0.0, 0
            sp = (step + 1) * args.batch * args.block / (time.time() - t0)
            print(f"step {step+1:5d} | train {tr:.3f} b/char | lr {lr_at(step):.1e} | "
                  f"gnorm {gn:.2f} | {sp:.0f} tok/s")
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            vb = evaluate(model, val_ds, args.batch, 10, dev, args.decohere)
            flag = ""
            if vb < best_val:
                best_val = vb
                tag = f"{args.tag}_{'dec' if args.decohere else 'int'}"
                torch.save({"model": model.state_dict(), "val_bits": vb,
                            "step": step + 1, "args": vars(args)},
                           os.path.join(args.out, f"{tag}_best.pt"))
                flag = " *best (checkpointed)"
            print(f"   >>> step {step+1}: VAL {vb:.3f} bits/char{flag}")

    model.eval()
    ids = torch.tensor([tok.encode(args.prompt)], device=dev)
    out = model.generate(ids, 300, temperature=0.7, top_k=20, decohere=args.decohere)
    print("\n===== PROMPT =====\n" + repr(args.prompt))
    print("===== CONTINUATION =====\n" + tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
