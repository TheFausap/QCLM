"""Long-range COHERENCE metrics for the unitary-memory hybrid.

Motivation: perplexity (bits/char) consistently favored the DECOHERED model, yet
the INTACT (phase-on) samples looked qualitatively more coherent -- sustaining a
consistent cast of speakers across a scene. Perplexity scores LOCAL next-char
prediction and under-weights LONG-RANGE consistency. If quantum phase trades a
little local perplexity for better long-range coherence, that is (a) invisible to
perplexity and (b) exactly what matters for conversation.

This script measures coherence QUANTITATIVELY over many generated samples, for
intact vs decohered checkpoints, against real-corpus reference values:

  1. speaker_recurrence : 1 - distinct/total speaker-turns. Real Shakespeare =
     0.972 (speakers return constantly). Incoherent model invents a new name each
     turn -> near 0.
  2. real_cast_fraction : fraction of generated speaker labels that are ACTUAL
     corpus characters (not invented gibberish names).
  3. long_range_word_reuse : rate at which words (len>=4) reused after the model
     has emitted >= REUSE_GAP other words -- an entity/topic tracking proxy.
  4. self_bleu_distance : optional diversity check (avoid trivial repetition
     inflating reuse).

Run after training the four checkpoints (real/complex embed x intact/decohere):
  python qlm/coherence_metric.py --ckpt artifacts/uce_int_best.pt
  python qlm/coherence_metric.py --ckpt artifacts/uce_dec_best.pt --decohere
The decohere flag must match how the checkpoint was trained (so generation uses
the same path the model learned).
"""
from __future__ import annotations
import os, sys, re, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
import torch
from qlm.data import CharTokenizer, load_text
from qlm.unitary_hybrid import UnitaryMemoryHybridLM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def speaker_labels(t):
    """Line-initial mostly-uppercase NAME: on its own line (tiny-shakespeare style)."""
    out = []
    for line in t.split("\n"):
        m = re.match(r"^([A-Z][A-Z ]{1,20}):\s*$", line.strip())
        if m:
            out.append(m.group(1).strip())
    return out


def real_cast(corpus):
    return set(speaker_labels(corpus))


def long_range_word_reuse(t, gap=40, minlen=4):
    words = re.findall(r"[a-zA-Z]{%d,}" % minlen, t.lower())
    seen = {}
    reuse, total = 0, 0
    for i, w in enumerate(words):
        if w in seen and (i - seen[w]) >= gap:
            reuse += 1
        seen[w] = i
        total += 1
    return reuse / max(total, 1)


def score_text(t, cast):
    labs = speaker_labels(t)
    tot = len(labs)
    distinct = len(set(labs))
    recurrence = (1 - distinct / tot) if tot else 0.0
    real_frac = (sum(1 for l in labs if l in cast) / tot) if tot else 0.0
    reuse = long_range_word_reuse(t)
    return dict(turns=tot, distinct=distinct, recurrence=recurrence,
                real_cast_fraction=real_frac, word_reuse=reuse)


def load_model(ckpt_path, vocab, dev):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    a = ck["args"]
    m = UnitaryMemoryHybridLM(vocab, n_layers=a["n_layers"], mem_dim=a["mem_dim"],
                              d_model=a["d_model"], n_heads=a["n_heads"],
                              block_size=a["block"], dropout=0.0,
                              complex_embed=a.get("complex_embed", False)).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck.get("val_bits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "tinyshakespeare.txt"))
    ap.add_argument("--decohere", action="store_true",
                    help="must match the checkpoint's training mode")
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--gen_len", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else (args.device if args.device != "auto" else "cpu"))
    corpus = load_text(args.data)
    tok = CharTokenizer(corpus)
    cast = real_cast(corpus)
    model, val_bits = load_model(args.ckpt, tok.vocab_size, dev)
    print(f"ckpt {os.path.basename(args.ckpt)} | val_bits {val_bits} | decohere {args.decohere}")
    print(f"real-corpus reference: recurrence 0.972, cast size {len(cast)}, "
          f"word_reuse {long_range_word_reuse(corpus):.3f}")

    torch.manual_seed(args.seed)
    agg = Counter()
    accum = dict(recurrence=0.0, real_cast_fraction=0.0, word_reuse=0.0,
                 turns=0.0, distinct=0.0)
    prompts = ["ROMEO:\n", "KING:\n", "First Citizen:\n", "\n"]
    n = 0
    for s in range(args.n_samples):
        p = prompts[s % len(prompts)]
        ids = torch.tensor([tok.encode(p)], device=dev)
        out = model.generate(ids, args.gen_len, temperature=args.temperature,
                             top_k=args.top_k, decohere=args.decohere)
        txt = tok.decode(out[0].tolist())
        sc = score_text(txt, cast)
        for k in accum:
            accum[k] += sc[k]
        n += 1
    for k in accum:
        accum[k] /= n
    print(f"\n=== COHERENCE over {n} samples (gen_len {args.gen_len}) ===")
    print(f"  speaker turns/sample:     {accum['turns']:.1f}")
    print(f"  distinct speakers/sample: {accum['distinct']:.1f}")
    print(f"  speaker RECURRENCE:       {accum['recurrence']:.3f}   (real 0.972; higher=better)")
    print(f"  REAL-CAST fraction:       {accum['real_cast_fraction']:.3f}   (higher=better)")
    print(f"  long-range WORD REUSE:    {accum['word_reuse']:.3f}   (real {long_range_word_reuse(corpus):.3f})")


if __name__ == "__main__":
    main()
