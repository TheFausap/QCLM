"""Sequentially run a list of training configs (robust subprocess driver)."""
import os, sys, subprocess, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (dim, decohere, tag)
CONFIGS = [
    (24, True,  'c_d24'),
    (48, False, 'q_d48'),
    (48, True,  'c_d48'),
]
COMMON = ['--kraus', '4', '--block', '64', '--batch', '32', '--steps', '1800',
          '--lr', '3e-3', '--eval_every', '300', '--eval_batches', '20',
          '--threads', '4', '--seed', '0']

if __name__ == '__main__':
    # allow overriding the config list from argv as dim:deco:tag,...
    if len(sys.argv) > 1:
        CONFIGS = []
        for spec in sys.argv[1].split(','):
            d, deco, tag = spec.split(':')
            CONFIGS.append((int(d), deco == '1', tag))
    for dim, deco, tag in CONFIGS:
        cmd = [sys.executable, '-u', os.path.join(HERE, 'qlm', 'train.py'),
               '--dim', str(dim), '--tag', tag] + COMMON
        if deco:
            cmd.append('--decohere')
        print(f'\n##### START {tag} (dim={dim} decohere={deco}) #####', flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=HERE)
        print(f'##### END {tag} rc={r.returncode} ({time.time()-t0:.0f}s) #####', flush=True)
    print('MATRIX_DONE', flush=True)
