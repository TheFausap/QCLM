"""Scale-up training for the Quantum Channel LM (DGX Spark target).

Same QCLM as the CPU PoC, wired for a real corpus and GPU:
  - byte-level BPE tokenizer (qlm/tokenizer.py)
  - streaming FineWeb / FineWeb-Edu, packed into blocks (qlm/data_fineweb.py)
  - device auto-select, gradient accumulation, truncated BPTT, cosine LR
  - periodic held-out eval + samples + checkpointing

The model arithmetic is complex64 (CUDA-capable). For peak Blackwell tensor-core
throughput, implement complex matmuls as real 2x2 blocks in bf16 (see PLAN_2B.md);
that is a drop-in kernel change, not an architecture change.

Runs on CPU with a tiny config for a smoke test; scales up purely via flags.
"""
from __future__ import annotations
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from qlm.model import QuantumChannelLM
from qlm.tokenizer import BPETokenizer
from qlm.data_fineweb import make_stream, fineweb_doc_iter, local_doc_iter

LN2 = math.log(2.0)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def build_or_load_tokenizer(args) -> BPETokenizer:
    if args.tokenizer and os.path.exists(args.tokenizer):
        print("loading tokenizer:", args.tokenizer)
        return BPETokenizer.load(args.tokenizer)
    # train a fresh BPE on a slice of the corpus
    print(f"training byte-level BPE (vocab={args.vocab}) on {args.bpe_train_docs} docs ...")
    if args.source == "fineweb":
        docs = fineweb_doc_iter(name=args.fineweb_name, dataset=args.fineweb_dataset)
    else:
        docs = local_doc_iter(args.local_path)

    def limited(it, k):
        for i, d in enumerate(it):
            if i >= k:
                break
            yield d
    tok = BPETokenizer.train(limited(docs, args.bpe_train_docs), vocab_size=args.vocab)
    out = args.tokenizer or os.path.join(args.out, f"bpe_{args.vocab}.json")
    tok.save(out)
    print("saved tokenizer:", out)
    return tok


@torch.no_grad()
def evaluate(model, eval_batches, device, tbptt):
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    for seqs in eval_batches:
        out = model.forward(seqs.to(device), tbptt=tbptt)
        tot_nll += out["nll_sum"].item(); tot_tok += out["n_tokens"]
    model.train()
    nll = tot_nll / max(tot_tok, 1)
    return nll / LN2


def main():
    ap = argparse.ArgumentParser()
    # data / tokenizer
    ap.add_argument("--source", choices=["fineweb", "local"], default="fineweb")
    ap.add_argument("--fineweb_dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--fineweb_name", default="sample-10BT")
    ap.add_argument("--local_path", default=os.path.join(HERE, "data", "tinyshakespeare.txt"))
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--bpe_train_docs", type=int, default=50000)
    # model
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--kraus", type=int, default=4)
    ap.add_argument("--fast_kernels", action="store_true",
                    help="use real 2x2-block bf16 matmuls instead of complex64 einsums (CUDA only)")
    # optimization
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--tbptt", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0,
                    help="AdamW weight decay (0 = off; rare-token operators get no gradient so "
                         "positive wd drives them to zero and worsens G conditioning)")
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--clip", type=float, default=1.0)
    # infra
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts"))
    ap.add_argument("--tag", default="qclm_scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", metavar="CKPT",
                    help="path to a _crash / _last checkpoint to resume from")
    ap.add_argument("--reproject_every", type=int, default=1,
                    help="every N optimizer steps, snap Kr/Ki back onto the isometry "
                         "manifold (exactly function-preserving retraction, ~0.1s). "
                         "The projection is scale-invariant, so Adam steps inflate "
                         "||V|| monotonically and shrink the EFFECTIVE lr by 1/||V||; "
                         "retracting every step (Riemannian Adam) prevents this and "
                         "keeps the Loewner backward well-conditioned. 0=off")
    ap.add_argument("--keep_opt", action="store_true",
                    help="also load optimizer state on --resume. ONLY safe if the "
                         "checkpoint was saved by this exact code version: Adam moments "
                         "are calibrated to a specific backward, and load_state_dict "
                         "silently restores the OLD hyperparameters (e.g. weight_decay).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    device = get_device(args.device)
    print("device:", device)

    tok = build_or_load_tokenizer(args)
    V = tok.vocab_size
    print("vocab_size:", V)

    def new_stream():
        kw = dict(source=args.source)
        if args.source == "fineweb":
            kw.update(name=args.fineweb_name, dataset=args.fineweb_dataset)
        else:
            kw.update(path=args.local_path, loop=True)
        return make_stream(tok, args.block, args.batch, tok.eot_id, **kw)

    stream = new_stream()
    print("pulling held-out eval set ...")
    eval_batches = [next(stream) for _ in range(args.eval_batches)]

    model = QuantumChannelLM(V, dim=args.dim, kraus=args.kraus,
                             fast_kernels=args.fast_kernels).to(device)
    params = model.num_params()
    print(f"model params: {params/1e9:.3f}B  (dim={args.dim} kraus={args.kraus} V={V})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_step = 0; seen_tok = 0; best = float("inf")
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume: checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        # Re-normalise the loaded parametrisation onto the isometry manifold.
        # This does NOT change the model (forward projects anyway), but makes
        # G ~= I so the Loewner backward starts well-scaled and well-conditioned.
        model.reproject_()
        if args.keep_opt:
            old_wd = ckpt["opt"]["param_groups"][0].get("weight_decay", None)
            opt.load_state_dict(ckpt["opt"])
            # load_state_dict restored the checkpoint's hyperparams; the training
            # loop only overrides lr, so re-assert the CLI values here. (A stale
            # weight_decay from an old run would silently come back otherwise.)
            for pg in opt.param_groups:
                pg["weight_decay"] = args.wd
                pg["betas"] = (0.9, 0.95)
            print(f"  loaded optimizer state (ckpt wd={old_wd} -> using wd={args.wd})")
        else:
            print("  fresh optimizer state (stale Adam moments discarded; "
                  "pass --keep_opt only if the ckpt was saved by this code version)")
        start_step = ckpt.get("step", 0)
        seen_tok   = ckpt.get("seen_tokens", 0)
        best       = ckpt.get("val_bits", float("inf"))
        print(f"resumed from {args.resume}")
        print(f"  start_step={start_step}  seen={seen_tok/1e6:.1f}M tok  best={best:.3f}")

    cfg = vars(args).copy(); cfg.update(vocab_size=V, num_params=params, device=str(device))
    # Drop keys that are irrelevant for the chosen source to keep the banner clean.
    if args.source == "fineweb":
        cfg.pop("local_path", None)
    else:
        cfg.pop("fineweb_dataset", None); cfg.pop("fineweb_name", None)
    # Append to the existing log when resuming so history is preserved.
    log_path = os.path.join(args.out, f"{args.tag}_log.jsonl")
    logf = open(log_path, "a" if args.resume else "w")
    logf.write(json.dumps({"type": "config", "resumed_from": args.resume, **cfg}) + "\n")
    logf.flush()
    print("CONFIG:", json.dumps(cfg))

    def save_ckpt(label: str, extra: dict | None = None):
        path = os.path.join(args.out, f"{args.tag}_{label}.pt")
        payload = {"model": model.state_dict(), "opt": opt.state_dict(),
                   "cfg": cfg, "step": step, "seen_tokens": seen_tok}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    model.train()
    t0 = time.time(); run_nll, run_tok = 0.0, 0
    tok0 = seen_tok  # tokens seen before this process started (for correct tok/s)
    step = start_step
    try:
        for step in range(start_step, args.steps):
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            acc_nll = 0.0
            for _ in range(args.grad_accum):
                seqs = next(stream).to(device)
                out = model.forward(seqs, tbptt=args.tbptt)
                (out["loss"] / args.grad_accum).backward()
                acc_nll += out["nll_sum"].item(); run_tok += out["n_tokens"]; seen_tok += out["n_tokens"]
            run_nll += acc_nll
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if torch.isnan(gnorm) or torch.isinf(gnorm):
                print(f"step {step+1}: NaN/Inf gnorm — skipping optimizer step")
                opt.zero_grad(set_to_none=True)
                run_nll = float("nan"); run_tok = max(run_tok, 1)  # force nan into the log
                continue
            opt.step()
            if args.reproject_every and (step + 1) % args.reproject_every == 0:
                # Log G's spectrum BEFORE retraction (this is the drift telemetry),
                # then snap back onto the manifold. The model's function is unchanged.
                with torch.no_grad():
                    Vm = torch.complex(model.Kr, model.Ki).reshape(-1, model.n)
                    s = torch.linalg.eigvalsh(Vm.mH @ Vm).real
                    g_min, g_max = s.min().item(), s.max().item()
                model.reproject_()
                cond = g_max / max(g_min, 1e-12)
                if cond > 10.0 or g_max > 4.0 or g_min < 0.25:
                    print(f"   [reproject] step {step+1}: G eigs [{g_min:.3e}, {g_max:.3e}] "
                          f"cond {cond:.1f} -> 1")
                if (step + 1) % 20 == 0:
                    logf.write(json.dumps({"type": "reproject", "step": step + 1,
                                           "G_min": g_min, "G_max": g_max,
                                           "G_cond": cond}) + "\n"); logf.flush()

            if (step + 1) % 20 == 0:
                tr_bits = (run_nll / run_tok) / LN2; run_nll, run_tok = 0.0, 0
                tps = (seen_tok - tok0) / (time.time() - t0)
                print(f"step {step+1:6d} | train {tr_bits:.3f} b/tok | lr {lr_at(step):.1e} | "
                      f"gnorm {gnorm:.2f} | {tps:.0f} tok/s | seen {seen_tok/1e6:.1f}M")
                logf.write(json.dumps({"type": "train", "step": step + 1,
                                       "train_bits": tr_bits, "gnorm": gnorm.item(),
                                       "lr": lr_at(step), "tok_per_s": tps,
                                       "seen_tokens": seen_tok}) + "\n"); logf.flush()
            if (step + 1) % args.eval_every == 0:
                vb = evaluate(model, eval_batches, device, args.tbptt)
                print(f"   >>> step {step+1}: VAL {vb:.3f} bits/token | {seen_tok/1e6:.1f}M tokens")
                logf.write(json.dumps({"type": "eval", "step": step + 1, "val_bits": vb,
                                       "seen_tokens": seen_tok, "time": time.time() - t0}) + "\n"); logf.flush()
                try:
                    ids = model.generate(tok.encode("\n"), n_new=80, temperature=0.8, seed=1)
                    print("   SAMPLE:", repr(tok.decode(ids)[:200]))
                except Exception as e:
                    print("   sample failed:", e)
                if vb < best:
                    best = vb
                    save_ckpt("best", {"tokenizer_path": args.tokenizer, "val_bits": vb})
            if (step + 1) % args.ckpt_every == 0:
                save_ckpt("last")

    except Exception as e:
        path = save_ckpt("crash")
        print(f"\n*** CRASH at step {step}: {e}")
        print(f"*** Emergency checkpoint saved → {path}")
        logf.write(json.dumps({"type": "crash", "step": step, "error": str(e),
                               "seen_tokens": seen_tok}) + "\n"); logf.flush()
        raise

    logf.write(json.dumps({"type": "final", "best_val_bits": best,
                           "wall_time": time.time() - t0, "seen_tokens": seen_tok}) + "\n")
    logf.close()
    print(f"DONE best={best:.3f} bits/tok, {seen_tok/1e6:.1f}M tokens, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
