"""Interactive-style demo: the QCLM 'expresses itself' by continuing prompts.

This is the conditional-generation view of the model: we feed a prompt (which
the quantum state absorbs via the post-measurement update), then let the Born
rule drive generation. Shows the model 'replying' in the style of its training
corpus.
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qlm.analysis import load_model


def reply(model, tok, prompt, n_new=200, temperature=0.7, top_k=12, seed=0):
    ids = model.generate(tok.encode(prompt), n_new=n_new, temperature=temperature,
                         top_k=top_k, seed=seed)
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_k', type=int, default=12)
    ap.add_argument('--n_new', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prompts', nargs='*', default=None)
    args = ap.parse_args()
    model, tok, cfg, ck = load_model(args.ckpt)
    print(f"# QCLM demo | dim={cfg['dim']} kraus={cfg['kraus']} "
          f"params={model.num_params()} val_bits={ck.get('val_bits'):.3f}")
    prompts = args.prompts or [
        "ROMEO:",
        "To be, or not to be,",
        "KING RICHARD III:\n",
        "My lord, ",
        "First Citizen:\n",
        "What is",
    ]
    for i, p in enumerate(prompts):
        out = reply(model, tok, p, n_new=args.n_new, temperature=args.temperature,
                    top_k=args.top_k, seed=args.seed + 101 * i)
        print("\n" + "=" * 70)
        print("PROMPT >>> " + repr(p))
        print("-" * 70)
        print(out)


if __name__ == '__main__':
    main()
