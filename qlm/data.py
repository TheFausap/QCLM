"""Data loading and character-level tokenization for the Quantum Channel LM.

Char-level keeps the vocabulary tiny (~65 for tiny-shakespeare) which is ideal
for a quantum state-space model where the per-token cost scales with |vocab|.
"""
from __future__ import annotations
import os
import numpy as np
import torch


class CharTokenizer:
    """Minimal reversible character tokenizer."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.itos = {i: c for i, c in enumerate(chars)}
        self.stoi = {c: i for i, c in self.itos.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def state_dict(self):
        return {"itos": self.itos}

    @classmethod
    def from_state_dict(cls, d):
        obj = cls.__new__(cls)
        obj.itos = {int(k): v for k, v in d["itos"].items()}
        obj.stoi = {c: i for i, c in obj.itos.items()}
        obj.vocab_size = len(obj.itos)
        return obj


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharDataset:
    """Holds an encoded corpus and yields contiguous (x, y) chunks.

    Each example is a length-T sequence of token ids; the model is trained to
    predict token t+1 from tokens <= t (standard autoregressive NLL), but the
    *mechanism* is a quantum channel, not attention or a gated recurrence.
    """

    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int,
                 split: str = "train", train_frac: float = 0.9, seed: int = 0):
        data = np.array(tokenizer.encode(text), dtype=np.int64)
        n = len(data)
        n_train = int(n * train_frac)
        if split == "train":
            self.data = data[:n_train]
        else:
            self.data = data[n_train:]
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def sample_batch(self, batch_size: int):
        """Random contiguous chunks. Returns (idx, ) of token ids shape (B, T+1)."""
        T = self.block_size
        ix = self.rng.integers(0, len(self.data) - T - 1, size=batch_size)
        seqs = np.stack([self.data[i:i + T + 1] for i in ix])
        return torch.from_numpy(seqs)  # (B, T+1) int64

    def iter_eval_batches(self, batch_size: int, max_batches: int | None = None):
        """Deterministic non-overlapping chunks for evaluation."""
        T = self.block_size
        starts = list(range(0, len(self.data) - T - 1, T))
        for b in range(0, len(starts), batch_size):
            chunk_starts = starts[b:b + batch_size]
            seqs = np.stack([self.data[i:i + T + 1] for i in chunk_starts])
            yield torch.from_numpy(seqs)
            if max_batches is not None and (b // batch_size) + 1 >= max_batches:
                return


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = load_text(os.path.join(here, "data", "tinyshakespeare.txt"))
    tok = CharTokenizer(text)
    print("vocab_size:", tok.vocab_size)
    ds = CharDataset(text, tok, block_size=32)
    b = ds.sample_batch(4)
    print("batch shape:", b.shape)
    print("decoded[0]:", repr(tok.decode(b[0].tolist())))
