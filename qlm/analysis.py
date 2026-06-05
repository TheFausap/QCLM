"""Analysis & visualization for the Quantum Channel LM.

Produces the figures and quantitative artifacts used in the write-up:
 - learning curves (bits/char vs step)
 - quantum-state diagnostics along a real text: purity Tr(rho^2) and
   l1-coherence C(rho)=sum_{i!=j}|rho_ij| (zero for any classical model)
 - a "Hilbert-space character map": 2D embedding of characters from their
   measurement operators M_x, revealing learned linguistic structure
 - temperature-controlled samples
"""
from __future__ import annotations
import os, sys, json, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qlm.data import CharTokenizer, load_text
from qlm.model import QuantumChannelLM

LN2 = math.log(2.0)


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    tok = CharTokenizer.from_state_dict(ck['tokenizer'])
    model = QuantumChannelLM(cfg['vocab_size'], dim=cfg['dim'], kraus=cfg['kraus'])
    model.load_state_dict(ck['model'])
    model.eval()
    return model, tok, cfg, ck


# ---------------------------------------------------------------------------
def plot_learning_curves(log_paths: dict, out_png: str, baselines: dict | None = None):
    plt.figure(figsize=(7, 4.5))
    for label, path in log_paths.items():
        steps, vals = [], []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get('type') == 'eval':
                    steps.append(r['step']); vals.append(r['val_bits'])
        if steps:
            plt.plot(steps, vals, marker='o', ms=3, label=label)
    if baselines:
        for name, b in baselines.items():
            plt.axhline(b, ls='--', lw=1, alpha=0.6)
            plt.text(plt.xlim()[1]*0.62, b+0.02, name, fontsize=7, alpha=0.8)
    plt.xlabel('training step'); plt.ylabel('validation bits/char')
    plt.title('Quantum Channel LM — learning curve')
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_png, dpi=130); plt.close()
    print('wrote', out_png)


# ---------------------------------------------------------------------------
@torch.no_grad()
def state_trajectory(model: QuantumChannelLM, tok: CharTokenizer, text: str):
    """Return purity and l1-coherence of rho after consuming each char of text."""
    K = model.kraus_operators()
    ids = tok.encode(text)
    rho = model.initial_rho(1)[0]
    n = model.n
    eps = 1e-12
    purity, coher = [], []
    offdiag_mask = ~torch.eye(n, dtype=torch.bool)
    for tid in ids:
        Kx = K[tid]                                   # (W,n,n)
        Krho = torch.einsum('wij,jk->wik', Kx, rho)
        E = torch.einsum('wik,wlk->il', Krho, Kx.conj())
        p = torch.einsum('ii->', E).real.clamp_min(eps)
        rho = E / p
        purity.append(torch.trace(rho @ rho).real.item())
        coher.append(rho.abs()[offdiag_mask].sum().item())
    return ids, np.array(purity), np.array(coher)


def plot_state_trajectory(model, tok, text, out_png):
    ids, purity, coher = state_trajectory(model, tok, text)
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    x = np.arange(len(ids))
    ax[0].plot(x, purity, color='tab:blue')
    ax[0].axhline(1.0, ls=':', color='gray', lw=1)
    ax[0].axhline(1.0/model.n, ls=':', color='red', lw=1)
    ax[0].set_ylabel('purity Tr(rho^2)')
    ax[0].set_title(f'Quantum state along: "{text[:60]}..."')
    ax[1].plot(x, coher, color='tab:purple')
    ax[1].set_ylabel('l1-coherence\nSum|rho_ij| i!=j')
    ax[1].set_xlabel('character position')
    # annotate characters sparsely
    chars = tok.decode(ids)
    for i in range(0, len(chars), max(1, len(chars)//60)):
        ax[1].annotate(chars[i] if chars[i] != '\n' else '\\n', (i, 0),
                       fontsize=6, ha='center', va='bottom', color='gray')
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()
    print('wrote', out_png, '| mean purity', purity.mean(), '| mean coherence', coher.mean())
    return purity.mean(), coher.mean()


# ---------------------------------------------------------------------------
@torch.no_grad()
def char_hilbert_map(model, tok, out_png):
    """2D PCA of characters using their POVM measurement operators M_x as features."""
    K = model.kraus_operators()
    M = model.povm(K).numpy()                          # (V, n, n) complex
    V, n, _ = M.shape
    feats = np.concatenate([M.real.reshape(V, -1), M.imag.reshape(V, -1)], axis=1)
    feats = feats - feats.mean(0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(feats, full_matrices=False)
    coords = U[:, :2] * S[:2]
    plt.figure(figsize=(7, 6))
    for i in range(V):
        c = tok.itos[i]
        label = {'\n': '\\n', ' ': '_'}.get(c, c)
        color = 'tab:red' if c in 'aeiouAEIOU' else ('tab:green' if c.isalpha() else 'tab:gray')
        plt.scatter(coords[i, 0], coords[i, 1], s=10, color=color, alpha=0.6)
        plt.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=8, color=color)
    plt.title('Characters embedded in Hilbert space (PCA of POVM operators)\n'
              'red=vowel  green=consonant  gray=other')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()
    print('wrote', out_png)


# ---------------------------------------------------------------------------
def samples(model, tok, prompt='\n', temps=(0.5, 0.8, 1.0), n_new=300, seed=7):
    outs = {}
    for T in temps:
        ids = model.generate(tok.encode(prompt), n_new=n_new, temperature=T, seed=seed)
        outs[T] = tok.decode(ids)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts'))
    args = ap.parse_args()
    model, tok, cfg, ck = load_model(args.ckpt)
    print('loaded', args.ckpt, '| val_bits', ck.get('val_bits'), '| params', model.num_params())
    tag = cfg.get('tag', 'model')
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = load_text(os.path.join(here, 'data', 'tinyshakespeare.txt'))
    snippet = text[1000:1320]
    plot_state_trajectory(model, tok, snippet, os.path.join(args.out, f'{tag}_trajectory.png'))
    char_hilbert_map(model, tok, os.path.join(args.out, f'{tag}_charmap.png'))
    outs = samples(model, tok)
    for T, s in outs.items():
        print(f'\n===== T={T} =====\n{s}')


if __name__ == '__main__':
    main()
