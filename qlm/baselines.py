"""Classical n-gram baselines for context.

These contextualize the quantum model's bits/char against standard reference
points (uniform, unigram, bigram, trigram with add-k smoothing). The most
important *internal* baseline is the decohered model (a quantum-dimension-matched
classical HMM), trained separately via train.py --decohere.
"""
from __future__ import annotations
import os, sys, math, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qlm.data import CharTokenizer, load_text

LN2 = math.log(2.0)


def ngram_bits(text_train, text_val, tok, order, k=0.1):
    V = tok.vocab_size
    ctx_counts = defaultdict(lambda: defaultdict(float))
    enc_tr = tok.encode(text_train)
    for i in range(len(enc_tr)):
        ctx = tuple(enc_tr[max(0, i - (order - 1)):i])
        ctx_counts[ctx][enc_tr[i]] += 1.0
    # evaluate
    enc_va = tok.encode(text_val)
    nll = 0.0
    for i in range(len(enc_va)):
        ctx = tuple(enc_va[max(0, i - (order - 1)):i])
        counts = ctx_counts.get(ctx)
        if counts is None:
            p = 1.0 / V
        else:
            total = sum(counts.values()) + k * V
            p = (counts.get(enc_va[i], 0.0) + k) / total
        nll -= math.log(max(p, 1e-12))
    return nll / len(enc_va) / LN2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tinyshakespeare.txt'))
    ap.add_argument('--train_frac', type=float, default=0.9)
    args = ap.parse_args()
    text = load_text(args.data)
    tok = CharTokenizer(text)
    n = len(text)
    n_tr = int(n * args.train_frac)
    text_tr, text_va = text[:n_tr], text[n_tr:]
    print(f"vocab={tok.vocab_size}  uniform={math.log2(tok.vocab_size):.3f} bits/char")
    for order in (1, 2, 3, 4):
        b = ngram_bits(text_tr, text_va, tok, order)
        name = {1: 'unigram', 2: 'bigram', 3: 'trigram', 4: '4-gram'}[order]
        print(f"{name:8s} (order {order}): {b:.3f} bits/char")


if __name__ == '__main__':
    main()
