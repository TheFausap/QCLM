"""Profiling harness for the QCLM training step.

Reproduces one real optimizer step (grad_accum x forward/backward + clip +
Adam + retraction) on RANDOM tokens (data content is irrelevant for speed),
then reports:

  1. compile cost            -- wall time of the first step (Triton compilation)
  2. phase breakdown         -- forward / backward / clip+opt / reproject, in ms
                                and as a share of the step, plus implied tok/s
  3. dynamo recompile count  -- if this grows past the first step, guards are
                                forcing recompilation (a classic silent killer)
  4. kernel table            -- top CUDA ops by self time (torch.profiler)
  5. chrome trace            -- artifacts/profile_trace.json
                                (open in chrome://tracing or https://ui.perfetto.dev)

Usage (matches the 2B run):
    python qlm/profile_run.py --dim 128 --kraus 4 --vocab 16384 \
        --batch 16 --block 512 --tbptt 128 --grad_accum 8 --fast_kernels
"""
from __future__ import annotations
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from qlm.model import QuantumChannelLM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def recompile_count() -> int:
    try:
        from torch._dynamo.utils import counters
        return sum(counters["stats"].values()) if "stats" in counters else \
            counters["frames"].get("total", 0)
    except Exception:
        return -1


def one_step(model, opt, tokens, args, phases=None):
    """One full optimizer step; if `phases` dict given, accumulate per-phase ms."""
    dev = tokens.device

    def stamp():
        sync(dev); return time.perf_counter()

    opt.zero_grad(set_to_none=True)
    if hasattr(model, "training_step"):
        # chunk-interleaved fused fwd+bwd path
        t0 = stamp()
        model.training_step([tokens] * args.grad_accum, tbptt=args.tbptt)
        t1 = stamp()
        if phases is not None:
            phases["fwd+bwd"] += (t1 - t0) * 1e3
    else:
      for _ in range(args.grad_accum):
        t0 = stamp()
        out = model.forward(tokens, tbptt=args.tbptt)
        t1 = stamp()
        (out["loss"] / args.grad_accum).backward()
        t2 = stamp()
        if phases is not None:
            phases["forward"] += (t1 - t0) * 1e3
            phases["backward"] += (t2 - t1) * 1e3
    t3 = stamp()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    t4 = stamp()
    model.reproject_()
    t5 = stamp()
    if phases is not None:
        phases["clip+opt"] += (t4 - t3) * 1e3
        phases["reproject"] += (t5 - t4) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--kraus", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--tbptt", type=int, default=128)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--fast_kernels", action="store_true")
    ap.add_argument("--timed_steps", type=int, default=3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else (args.device if args.device != "auto" else "cpu"))
    print(f"device: {dev}")
    if dev.type != "cuda":
        print("WARNING: not on CUDA -- numbers will not reflect the Spark.")

    torch.manual_seed(0)
    model = QuantumChannelLM(args.vocab, dim=args.dim, kraus=args.kraus,
                             fast_kernels=args.fast_kernels).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.95),
                            weight_decay=0.0)
    tokens = torch.randint(0, args.vocab, (args.batch, args.block + 1), device=dev)
    tok_per_step = args.batch * (args.block + 1) * args.grad_accum

    # ---- step 1: compile cost ------------------------------------------------
    rc0 = recompile_count()
    t0 = time.perf_counter()
    one_step(model, opt, tokens, args)
    sync(dev)
    t_compile = time.perf_counter() - t0
    print(f"\nstep 1 (includes torch.compile / Triton build): {t_compile:.1f} s")

    # ---- step 2: post-compile sanity ------------------------------------------
    t0 = time.perf_counter()
    one_step(model, opt, tokens, args)
    sync(dev)
    t_warm = time.perf_counter() - t0
    rc1 = recompile_count()
    print(f"step 2 (should be fast if compile cached):      {t_warm:.1f} s")
    print(f"dynamo frame/compile counter after warmup: {rc1}  (delta from start: {rc1 - rc0})")

    # ---- timed phase breakdown -------------------------------------------------
    from collections import defaultdict
    phases = defaultdict(float)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rc_before = recompile_count()
    t0 = time.perf_counter()
    for _ in range(args.timed_steps):
        one_step(model, opt, tokens, args, phases)
    sync(dev)
    total_ms = (time.perf_counter() - t0) * 1e3
    rc_after = recompile_count()

    print(f"\n==== phase breakdown (avg over {args.timed_steps} steps, "
          f"{tok_per_step} tok/step) ====")
    for k, v in phases.items():
        avg = v / args.timed_steps
        print(f"  {k:10s} {avg:10.0f} ms   {100*v/total_ms:5.1f}%")
    step_ms = total_ms / args.timed_steps
    print(f"  {'TOTAL':10s} {step_ms:10.0f} ms   ->  {tok_per_step / (step_ms/1e3):,.0f} tok/s")
    print(f"recompiles during timed steps: {rc_after - rc_before} "
          f"(MUST be 0 -- if not, guards are recompiling every step)")
    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"peak CUDA memory during timed steps: {peak:.1f} GiB")

    # ---- torch.profiler over one step ------------------------------------------
    print("\nprofiling one step ...")
    acts = [torch.profiler.ProfilerActivity.CPU]
    if dev.type == "cuda":
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=acts, record_shapes=False) as prof:
        one_step(model, opt, tokens, args)
        sync(dev)
    sort_key = "self_cuda_time_total" if dev.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25, max_name_column_width=60))
    trace = os.path.join(HERE, "artifacts", "profile_trace.json")
    os.makedirs(os.path.dirname(trace), exist_ok=True)
    prof.export_chrome_trace(trace)
    print("chrome trace written:", trace)


if __name__ == "__main__":
    main()
