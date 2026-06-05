"""DGX Spark scaling calculator for the Quantum Channel LM.

Parameter / memory / compute model for the QCLM, used to size a scale-up run on
an NVIDIA DGX Spark (GB10 Grace Blackwell): 128 GB unified LPDDR5x @ ~273 GB/s,
~1 PFLOP FP4 (sparse) tensor performance (~500 dense FP4 TFLOP, ~125 dense BF16
TFLOP). Complex arithmetic is realized as real 2x2 blocks on the tensor cores,
costing ~4x the real FLOPs, so effective complex BF16 throughput ~ 30 TFLOP.

QCLM parameter count (counting real numbers; complex = 2 reals):
    params ~= 2 * V * W * n^2        (Kraus operators dominate)  + 2n (init state)
where V = vocab size, n = Hilbert-space dimension, W = Kraus operators per token.

Per-token compute is dominated by the Born-rule readout over the whole vocab,
p(x) = Tr(rho M_x), an (n,n) x (V,n,n) contraction  ->  V * n^2 complex MACs,
exactly analogous to the output projection of a standard LM. The channel update
touches only the observed token's W operators (cheap).
"""
from __future__ import annotations
import argparse

# ---- DGX Spark (GB10) verified envelope ----
SPARK = {
    'mem_gb': 128,
    'bw_gbs': 273,                 # LPDDR5x unified bandwidth
    'fp4_sparse_tflop': 1000,      # marketing peak
    'fp4_dense_tflop': 500,
    'bf16_dense_tflop': 125,       # approx dense BF16
    'complex_bf16_eff_tflop': 30,  # complex via real blocks (~/4)
}

CDTYPE_BYTES = 8   # complex64
BF16_BYTES = 2


def analyze(V, n, W, batch=64, T=512, cdtype_bytes=CDTYPE_BYTES, train=True):
    params = 2 * V * W * n * n + 2 * n            # real-number count
    # ---- memory ----
    w_complex = (params / 2) * cdtype_bytes       # complex64 storage of weights
    w_bf16 = params * BF16_BYTES                  # bf16 real storage
    adam = 2 * params * 4                         # fp32 m,v
    master = params * 4                           # fp32 master weights
    grads = params * BF16_BYTES
    # activations: store rho per (batch,position) for backprop (complex)
    acts = batch * T * (n * n) * cdtype_bytes
    train_mem = master + adam + grads + w_bf16 + acts
    infer_mem = w_bf16  # FP4 would be params*0.5 bytes

    # ---- compute per token (complex MACs); complex MAC ~ 4 real MAC, 2 FLOP/MAC ----
    # TRAINING needs only the observed token: p(x_t)=Tr(E_{x_t}(rho)) falls out of the
    # state update, because the POVM resolves the identity (denominator == 1 exactly).
    # => training compute is O(W n^3), INDEPENDENT of vocabulary size V.
    channel_macs = W * n * n * n                  # K rho K^dag for the observed token
    train_flops_tok = channel_macs * 4 * 2
    # GENERATION (sampling) needs the full-vocab Born readout Tr(rho M_x) for all x:
    gen_flops_tok = (V * n * n + channel_macs) * 4 * 2

    eff = SPARK['complex_bf16_eff_tflop'] * 1e12
    train_step_s = train_flops_tok * batch * T * 3 / eff   # x3 fwd+bwd
    train_tok_s_compute = (batch * T) / train_step_s

    # At 2B params the Adam step touches all params (the global isometry constraint
    # couples all Kraus operators), so the optimizer is memory-bound:
    bytes_per_optstep = params * (4 + 4 + 4 + 2)  # m,v,master(fp32) + grad(bf16)
    opt_step_s = bytes_per_optstep / (SPARK['bw_gbs'] * 1e9)

    return {
        'V': V, 'n': n, 'W': W, 'params_B': params / 1e9,
        'w_complex_GB': w_complex / 1e9, 'w_bf16_GB': w_bf16 / 1e9,
        'train_mem_GB': train_mem / 1e9, 'infer_w_bf16_GB': infer_mem / 1e9,
        'train_gflops_per_token': train_flops_tok / 1e9,
        'gen_gflops_per_token': gen_flops_tok / 1e9,
        'est_train_tok_per_s': train_tok_s_compute,
        'opt_step_s': opt_step_s,
    }


PRESETS = {
    # name: (V, n, W, batch, T)
    'cpu-poc (current)':      (65,    48,  4, 32, 64),
    'spark-1B (bpe16k)':      (16384, 96,  4, 64, 512),
    'spark-2B (bpe16k)*':     (16384,128,  4, 48, 512),   # <-- recommended 2B headline
    'spark-2B (bpe32k,W2)':   (32768,128,  2, 48, 512),
    'spark-2.4B (bpe32k)':    (32768, 96,  4, 48, 512),
}


def budget(V, n, W, target_tokens_B, tok_per_s):
    secs = target_tokens_B * 1e9 / tok_per_s
    return secs / 3600.0  # hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--V', type=int); ap.add_argument('--n', type=int)
    ap.add_argument('--W', type=int, default=4)
    ap.add_argument('--batch', type=int, default=48); ap.add_argument('--T', type=int, default=512)
    args = ap.parse_args()
    print('DGX Spark envelope:', SPARK)
    hdr = (f"{'preset':22s} {'V':>6s} {'n':>4s} {'W':>2s} {'params':>8s} {'w(GB)':>7s} "
           f"{'train(GB)':>9s} {'train GF/tok':>12s} {'gen GF/tok':>11s} {'train tok/s*':>12s}")
    print('\n' + hdr); print('-' * len(hdr))
    items = list(PRESETS.items())
    if args.V and args.n:
        items.append(('custom', (args.V, args.n, args.W, args.batch, args.T)))
    for name, (V, n, W, b, T) in items:
        r = analyze(V, n, W, batch=b, T=T)
        print(f"{name:22s} {V:6d} {n:4d} {W:2d} {r['params_B']:7.3f}B {r['w_complex_GB']:6.1f}G "
              f"{r['train_mem_GB']:8.1f}G {r['train_gflops_per_token']:12.3f} "
              f"{r['gen_gflops_per_token']:11.2f} {r['est_train_tok_per_s']:12.0f}")
    print("\n* compute-bound estimate at ~30 effective complex-BF16 TFLOP. KEY POINT: training")
    print("  compute is O(W n^3) and INDEPENDENT of vocab V (denominator==1 by POVM completeness),")
    print("  so the model is compute-light for its parameter count. At 2B the practical limiter is")
    print("  memory bandwidth (Adam over all params + Kraus gathers), not FLOPs.")
    r2 = analyze(16384, 128, 4, batch=48, T=512)
    print(f"\n2B headline (bpe16k,n128,W4): {r2['params_B']:.2f}B params, {r2['train_mem_GB']:.0f}GB train mem, "
          f"opt-step ~{r2['opt_step_s']*1000:.0f}ms (bandwidth-bound).")
    for tb, tps in [(1, 20000), (3, 20000), (1, 8000), (3, 8000)]:
        print(f"  budget {tb}B tokens @ {tps} tok/s  ->  ~{budget(16384,128,4,tb,tps):.0f} h "
              f"(~{budget(16384,128,4,tb,tps)/24:.1f} days)")


if __name__ == '__main__':
    main()
