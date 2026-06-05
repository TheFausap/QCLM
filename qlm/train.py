"""Training loop for the Quantum Channel Language Model."""
from __future__ import annotations
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from qlm.data import CharTokenizer, CharDataset, load_text
from qlm.model import QuantumChannelLM

LN2 = math.log(2.0)


def evaluate(model, ds, batch_size, max_batches, decohere=False):
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for b, seqs in enumerate(ds.iter_eval_batches(batch_size, max_batches)):
            out = model.forward(seqs, decohere=decohere)
            tot_nll += out['nll_sum'].item()
            tot_tok += out['n_tokens']
    model.train()
    nll = tot_nll / max(tot_tok, 1)
    return nll, nll / LN2  # nats/char, bits/char


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tinyshakespeare.txt'))
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--kraus', type=int, default=4)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--eval_every', type=int, default=200)
    ap.add_argument('--eval_batches', type=int, default=20)
    ap.add_argument('--decohere', action='store_true', help='train the classical (decohered) ablation')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts'))
    ap.add_argument('--tag', default='qclm')
    ap.add_argument('--corpus_frac', type=float, default=1.0, help='use only this fraction of training data')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    text = load_text(args.data)
    tok = CharTokenizer(text)
    if args.corpus_frac < 1.0:
        text_used = text[:int(len(text) * args.corpus_frac)]
    else:
        text_used = text
    train_ds = CharDataset(text_used, tok, block_size=args.block, split='train', seed=args.seed)
    val_ds = CharDataset(text, tok, block_size=args.block, split='val', seed=args.seed + 1)

    model = QuantumChannelLM(tok.vocab_size, dim=args.dim, kraus=args.kraus)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        # cosine decay to 10%
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    log_path = os.path.join(args.out, f'{args.tag}_log.jsonl')
    logf = open(log_path, 'w')
    cfg = vars(args).copy()
    cfg['vocab_size'] = tok.vocab_size
    cfg['num_params'] = model.num_params()
    print('CONFIG:', json.dumps(cfg))
    logf.write(json.dumps({'type': 'config', **cfg}) + '\n'); logf.flush()

    model.train()
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    best_val = float('inf')
    for step in range(args.steps):
        for pg in opt.param_groups:
            pg['lr'] = lr_at(step)
        seqs = train_ds.sample_batch(args.batch)
        out = model.forward(seqs, decohere=args.decohere)
        loss = out['loss']
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        run_loss += out['nll_sum'].item(); run_n += out['n_tokens']

        if (step + 1) % 50 == 0:
            tr_bits = (run_loss / run_n) / LN2
            run_loss, run_n = 0.0, 0
            speed = (step + 1) * args.batch * args.block / (time.time() - t0)
            print(f"step {step+1:5d} | train {tr_bits:.3f} bits/char | lr {lr_at(step):.1e} | "
                  f"gnorm {gnorm:.2f} | {speed:.0f} tok/s")

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            _, val_bits = evaluate(model, val_ds, args.batch, args.eval_batches,
                                   decohere=args.decohere)
            print(f"   >>> step {step+1}: VAL {val_bits:.3f} bits/char")
            rec = {'type': 'eval', 'step': step + 1, 'val_bits': val_bits,
                   'time': time.time() - t0}
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            # sample
            try:
                ids = model.generate(tok.encode("\n"), n_new=240, temperature=0.8,
                                     decohere=args.decohere, seed=1234)
                sample = tok.decode(ids)
                print('   SAMPLE:', repr(sample[:200]))
                logf.write(json.dumps({'type': 'sample', 'step': step + 1, 'text': sample}) + '\n'); logf.flush()
            except Exception as e:
                print('   sample failed:', e)
            if val_bits < best_val:
                best_val = val_bits
                ckpt = {'model': model.state_dict(), 'cfg': cfg, 'tokenizer': tok.state_dict(),
                        'val_bits': val_bits, 'step': step + 1}
                torch.save(ckpt, os.path.join(args.out, f'{args.tag}_best.pt'))

    logf.write(json.dumps({'type': 'final', 'best_val_bits': best_val,
                           'wall_time': time.time() - t0}) + '\n')
    logf.close()
    print(f"DONE. best val bits/char = {best_val:.3f}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
