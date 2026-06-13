"""Catch the FIRST non-finite chunk in training_step and dump everything needed
to tell apart the three hypotheses for the chunk-drops:

  (A) FLOOR event       -- some timestep's probability p hit the clamp (eps),
                           so the rho = E/p backward amplified by ~1/eps.
                           -> fix is to clamp the 1/p amplification in BACKWARD,
                              not raise the forward floor (treadmill).
  (B) CONDITIONING event -- some token's projected operator K_iso is built from
                           a near-singular G = Vmat^H Vmat; the Loewner backward
                           through G^{-1/2} blows up.  -> fix is in kraus_operators
                           / more frequent retraction for those tokens.
  (C) KERNEL artifact    -- the eager fp64 re-run of the SAME chunk is perfectly
                           finite (forward and backward), i.e. only the fused
                           Triton bf16 kernel produced the NaN. -> kernel-level
                           workaround / dtype change for the compiled path.

Usage (run a SEPARATE short job; do NOT attach to your live run):

    python -m qlm.diagnose_nan --resume artifacts/qclm_2b_opt_last.pt \
        --tokenizer artifacts/bpe_16384.json \
        --vocab 16384 --dim 128 --kraus 4 --fast_kernels \
        --source fineweb --fineweb_name sample-10BT \
        --batch 16 --grad_accum 8 --block 512 --tbptt 128 \
        --max_steps 200 --dump artifacts/nan_dump.pt

It loads your checkpoint, trains with the SAME settings, and the instant a chunk
goes non-finite it writes the dump and exits. Then send me artifacts/nan_dump.pt
(it is small -- just the offending chunk's tokens + per-step diagnostics).
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from qlm.model import QuantumChannelLM, _cplx_mm_block, _cplx_mm_dag_block


@torch.no_grad()
def reference_scan_f64(Kr_chunk, Ki_chunk, rho_r, rho_i, eps):
    """Eager fp64 forward of one chunk, recording per-step p and operator norms.
    Returns (p_list, finite_forward). No compiled kernel, max precision: if THIS
    is finite while the live kernel produced NaN, the cause is the kernel (C)."""
    Kr = Kr_chunk.double(); Ki = Ki_chunk.double()
    rr = rho_r.double(); ri = rho_i.double()
    B, C, W, n, _ = Kr.shape
    p_steps = []
    for t in range(C):
        Kx_r = Kr[:, t]; Kx_i = Ki[:, t]
        Qr, Qi = _cplx_mm_block(Kx_r, Kx_i, rr.unsqueeze(1), ri.unsqueeze(1), torch.complex128)
        Er, Ei = _cplx_mm_dag_block(Qr, Qi, Kx_r, Kx_i, torch.complex128)
        Er = Er.sum(1); Ei = Ei.sum(1)
        p = Er.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(eps)
        p_steps.append(p.detach().cpu())
        rr = Er / p[:, None, None]; ri = Ei / p[:, None, None]
    P = torch.stack(p_steps, 1)  # (B, C)
    return P, bool(torch.isfinite(P).all())


def operator_conditioning(model, token_ids):
    """For each unique token in the chunk, cond(G) where G = Vmat^H Vmat over the
    flattened (W*n, n) Kraus stack -- i.e. how singular the pre-projection is."""
    uniq = torch.unique(token_ids)
    Kr = model.Kr.detach(); Ki = model.Ki.detach()
    out = {}
    for tid in uniq.tolist():
        V = torch.complex(Kr[tid], Ki[tid])           # (W, n, n)
        W, n, _ = V.shape
        Vm = V.reshape(W * n, n)
        G = (Vm.conj().mT @ Vm).cpu()
        s = torch.linalg.eigvalsh(0.5 * (G + G.conj().mT)).real
        out[tid] = (float(s.min()), float(s.max()),
                    float(s.max() / max(s.min(), 1e-30)))
    return out


def make_instrumented_step(model, dump_path):
    eps_train = 1e-5
    orig_cdtype = model.compute_dtype

    def training_step(token_batches, tbptt=0, decohere=False):
        total_tok = sum(int(t.numel()) for t in token_batches)
        inv = 1.0 / float(total_tok)
        eps = eps_train
        cdtype = model.compute_dtype
        K = model.kraus_operators()
        Kr_full = K.real.detach().contiguous()
        Ki_full = K.imag.detach().contiguous()
        gKr = torch.zeros_like(Kr_full); gKi = torch.zeros_like(Ki_full)
        fast = model.fast_kernels and token_batches[0].device.type == "cuda"
        from qlm.model import _SCAN_CHUNK, _scan_chunk_impl
        scan = _SCAN_CHUNK if fast else _scan_chunk_impl
        nll_total = torch.zeros((), dtype=model.rdtype, device=token_batches[0].device)
        tail = gKr.shape[1:]
        for tokens in token_batches:
            B, L = tokens.shape
            rho = model.initial_rho(B, decohere=decohere)
            rho_r = rho.real.clone(); rho_i = rho.imag.clone()
            chunk = tbptt if tbptt > 0 else L
            for start in range(0, L, chunk):
                tok_c = tokens[:, start:start + chunk]
                # snapshot the ENTERING state (fp32) before the live kernel runs,
                # so we can replay this exact chunk in the fp64 reference.
                rin_r = rho_r.detach().clone(); rin_i = rho_i.detach().clone()
                Kr_chunk = Kr_full[tok_c].requires_grad_(True)
                Ki_chunk = Ki_full[tok_c].requires_grad_(True)
                nll_chunk, rho_r, rho_i = scan(
                    Kr_chunk, Ki_chunk, rho_r, rho_i, eps, decohere, cdtype)
                nll_chunk.backward(torch.ones((), dtype=nll_chunk.dtype,
                                               device=nll_chunk.device) * inv)
                gr_fin = bool(torch.isfinite(Kr_chunk.grad).all()
                              and torch.isfinite(Ki_chunk.grad).all())
                fwd_fin = bool(torch.isfinite(nll_chunk))
                if not (gr_fin and fwd_fin):
                    print("\n=== CAUGHT non-finite chunk ===")
                    print(f"  forward finite : {fwd_fin}")
                    print(f"  grad    finite : {gr_fin}")
                    # eager fp64 replay of the SAME chunk
                    P, ref_fwd_fin = reference_scan_f64(
                        Kr_chunk.detach(), Ki_chunk.detach(), rin_r, rin_i, eps)
                    n_at_floor = int((P <= eps * 1.0001).sum())
                    print(f"  fp64 forward finite (same chunk): {ref_fwd_fin}")
                    print(f"  per-step p: min={float(P.min()):.3e} "
                          f"max={float(P.max()):.3e}  steps_at_floor={n_at_floor}")
                    cond = operator_conditioning(model, tok_c)
                    worst = max(cond.values(), key=lambda v: v[2]) if cond else None
                    print(f"  worst token cond(G): {worst}")
                    torch.save({
                        "tok_c": tok_c.cpu(),
                        "p_per_step_f64": P,
                        "eps": eps,
                        "fwd_finite_live": fwd_fin,
                        "grad_finite_live": gr_fin,
                        "fwd_finite_f64": ref_fwd_fin,
                        "n_steps_at_floor": n_at_floor,
                        "cond_by_token": cond,
                        "compute_dtype": str(orig_cdtype),
                        "grad_nan_count_r": int((~torch.isfinite(Kr_chunk.grad)).sum()),
                        "grad_nan_count_i": int((~torch.isfinite(Ki_chunk.grad)).sum()),
                    }, dump_path)
                    print(f"\n  >>> dump written: {dump_path}")
                    print("  >>> diagnosis key:")
                    if ref_fwd_fin and (fwd_fin and not gr_fin):
                        print("      forward fine + only BACKWARD non-finite + fp64 fine")
                        print("      => (C) Triton bf16 KERNEL artifact, or (A) if many steps_at_floor")
                    if n_at_floor > 0:
                        print(f"      {n_at_floor} step(s) at the probability floor => (A) FLOOR event likely")
                    if worst and worst[2] > 1e3:
                        print(f"      a token's cond(G)={worst[2]:.1e} => (B) CONDITIONING event likely")
                    raise SystemExit(0)
                idx = tok_c.reshape(-1)
                gKr.index_put_((idx,), Kr_chunk.grad.reshape(-1, *tail), accumulate=True)
                gKi.index_put_((idx,), Ki_chunk.grad.reshape(-1, *tail), accumulate=True)
                nll_total += nll_chunk.detach()
                rho_r = rho_r.detach(); rho_i = rho_i.detach()
        K.backward(torch.complex(gKr, gKi))
        return float(nll_total) * inv

    return training_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--kraus", type=int, default=4)
    ap.add_argument("--fast_kernels", action="store_true")
    ap.add_argument("--source", default="fineweb")
    ap.add_argument("--fineweb_name", default="sample-10BT")
    ap.add_argument("--fineweb_dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--local_path", default="")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--tbptt", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--dump", default="artifacts/nan_dump.pt")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    from qlm.tokenizer import BPETokenizer
    tok = BPETokenizer.load(args.tokenizer)

    model = QuantumChannelLM(args.vocab, dim=args.dim, kraus=args.kraus,
                             fast_kernels=args.fast_kernels).to(dev)
    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    if hasattr(model, "reproject_"):
        model.reproject_()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)

    from qlm.data_fineweb import make_stream
    kw = dict(source=args.source)
    if args.source == "fineweb":
        kw.update(name=args.fineweb_name, dataset=args.fineweb_dataset)
    else:
        kw.update(path=args.local_path, loop=True)
    stream = make_stream(tok, args.block, args.batch, tok.eot_id, **kw)

    step_fn = make_instrumented_step(model, args.dump)
    print(f"hunting for the first non-finite chunk (max {args.max_steps} steps)...")
    for step in range(args.max_steps):
        opt.zero_grad(set_to_none=True)
        batches = [next(stream).to(dev) for _ in range(args.grad_accum)]
        nll = step_fn(batches, tbptt=args.tbptt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if hasattr(model, "reproject_"):
            model.reproject_()
        if step % 20 == 0:
            print(f"step {step}: nll/tok {nll:.4f}")
    print("no non-finite chunk encountered in the step budget.")


if __name__ == "__main__":
    main()
