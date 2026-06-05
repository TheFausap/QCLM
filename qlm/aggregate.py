"""Aggregate all run logs into the headline figures and a results table."""
from __future__ import annotations
import os, sys, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts')
BASELINES = {'uniform': 6.022, 'unigram': 4.829, 'bigram': 3.583,
             'trigram': 2.952, '4-gram': 2.574}


def read_log(tag):
    path = os.path.join(ART, f'{tag}_log.jsonl')
    if not os.path.exists(path):
        return None
    steps, vals = [], []
    final = None
    cfg = None
    for line in open(path):
        r = json.loads(line)
        if r.get('type') == 'eval':
            steps.append(r['step']); vals.append(r['val_bits'])
        elif r.get('type') == 'final':
            final = r.get('best_val_bits')
        elif r.get('type') == 'config':
            cfg = r
    return {'steps': steps, 'vals': vals, 'final': final, 'cfg': cfg}


def main():
    dims = [24, 48]
    data = {}
    for d in dims:
        data[('q', d)] = read_log(f'q_d{d}')
        data[('c', d)] = read_log(f'c_d{d}')

    # ---- Figure 1: learning curves quantum vs decohered ----
    plt.figure(figsize=(7.5, 5))
    colors = {24: 'tab:blue', 48: 'tab:red'}
    for d in dims:
        q = data[('q', d)]; c = data[('c', d)]
        if q and q['steps']:
            plt.plot(q['steps'], q['vals'], '-o', ms=4, color=colors[d],
                     label=f'quantum  n={d}')
        if c and c['steps']:
            plt.plot(c['steps'], c['vals'], '--s', ms=4, color=colors[d], alpha=0.7,
                     label=f'classical (decohered) n={d}')
    for name in ['bigram', 'trigram', '4-gram']:
        b = BASELINES[name]
        plt.axhline(b, ls=':', lw=1, color='gray', alpha=0.6)
        plt.text(plt.xlim()[1]*0.02, b+0.01, name, fontsize=7, color='gray')
    plt.xlabel('training step'); plt.ylabel('validation bits/char')
    plt.title('Quantum interference improves a matched-dimension model\n'
              '(solid=quantum, dashed=decohered classical HMM)')
    plt.legend(fontsize=8, loc='upper right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    f1 = os.path.join(ART, 'fig_learning_curves.png')
    plt.savefig(f1, dpi=140); plt.close(); print('wrote', f1)

    # ---- Figure 2: bar chart of best val bits/char ----
    labels, vals, cols = [], [], []
    for name in ['bigram', 'trigram', '4-gram']:
        labels.append(name); vals.append(BASELINES[name]); cols.append('lightgray')
    for d in dims:
        c = data[('c', d)]; q = data[('q', d)]
        if c and c['final'] is not None:
            labels.append(f'classical n={d}'); vals.append(c['final']); cols.append('tab:orange')
        if q and q['final'] is not None:
            labels.append(f'quantum n={d}'); vals.append(q['final']); cols.append('tab:blue')
    plt.figure(figsize=(8.5, 4.5))
    bars = plt.bar(range(len(labels)), vals, color=cols)
    plt.xticks(range(len(labels)), labels, rotation=30, ha='right', fontsize=8)
    plt.ylabel('best validation bits/char')
    plt.title('Quantum Channel LM vs classical baselines (lower is better)')
    for b, v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', fontsize=7)
    plt.ylim(2.3, 4.0)
    plt.tight_layout()
    f2 = os.path.join(ART, 'fig_bits_bar.png')
    plt.savefig(f2, dpi=140); plt.close(); print('wrote', f2)

    # ---- table ----
    print('\n=== RESULTS TABLE (val bits/char) ===')
    print(f'{"model":28s} {"bits/char":>10s} {"vs classical":>14s}')
    for name in ['uniform', 'unigram', 'bigram', 'trigram', '4-gram']:
        print(f'{name:28s} {BASELINES[name]:10.3f}')
    rows = {}
    for d in dims:
        for kind, key in [('quantum', 'q'), ('classical(decohered)', 'c')]:
            r = data[(key, d)]
            if r and r['final'] is not None:
                rows[(kind, d)] = r['final']
    for d in dims:
        cval = rows.get(('classical(decohered)', d))
        qval = rows.get(('quantum', d))
        if cval is not None:
            print(f'{("classical(decohered) n=%d"%d):28s} {cval:10.3f}')
        if qval is not None:
            delta = (cval - qval) if cval else float("nan")
            print(f'{("quantum n=%d"%d):28s} {qval:10.3f} {delta:14.3f}')
    # save table to json
    out = {'baselines': BASELINES,
           'models': {f'{k}_n{d}': rows[(k, d)] for (k, d) in rows}}
    json.dump(out, open(os.path.join(ART, 'results.json'), 'w'), indent=2)
    print('\nwrote', os.path.join(ART, 'results.json'))


if __name__ == '__main__':
    main()
