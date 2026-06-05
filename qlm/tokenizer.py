"""Byte-level BPE tokenizer for the QCLM scale-up.

Why byte-level BPE for this model (and for FineWeb):
 - The QCLM parameter count is 2*V*W*n^2, i.e. LINEAR in vocab size V, and the
   per-step Born-rule readout costs V*n^2. So we want a vocabulary that is "just
   large enough" (16k-32k), not 100k+.
 - Subword tokens carry more information per step than characters, so a fixed-size
   quantum state covers MORE text per step -> better long-range coherence, and the
   model no longer spends capacity learning spelling.
 - Byte-level BPE (GPT-2 style) operates on raw UTF-8 bytes, so it NEVER hits an
   out-of-vocabulary token on messy web text (FineWeb) -- ideal for a noisy corpus.

This wraps HuggingFace `tokenizers`. Interface matches qlm.data.CharTokenizer
(encode -> list[int], decode -> str, .vocab_size) so the model code is unchanged.
"""
from __future__ import annotations
import os
from typing import Iterable

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

EOT = "<|endoftext|>"


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.eot_id = tokenizer.token_to_id(EOT)

    # ---- training ----
    @classmethod
    def train(cls, corpus_iter: Iterable[str], vocab_size: int = 32768,
              min_frequency: int = 2) -> "BPETokenizer":
        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tk.train_from_iterator(corpus_iter, trainer=trainer)
        return cls(tk)

    # ---- io ----
    def save(self, path: str):
        self.tk.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        return cls(Tokenizer.from_file(path))

    # ---- use ----
    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids) -> str:
        return self.tk.decode([int(i) for i in ids])


if __name__ == "__main__":
    # Smoke test: train a tiny BPE on tiny-shakespeare and round-trip some text.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(here, "data", "tinyshakespeare.txt")).read()

    def chunks(s, n=4000):
        for i in range(0, len(s), n):
            yield s[i:i + n]

    tok = BPETokenizer.train(chunks(text), vocab_size=2048, min_frequency=2)
    print("vocab_size:", tok.vocab_size, "| EOT id:", tok.eot_id)
    sample = "First Citizen:\nBefore we proceed any further, hear me speak."
    ids = tok.encode(sample)
    print("n_chars:", len(sample), "-> n_tokens:", len(ids),
          f"(compression {len(sample)/len(ids):.2f}x)")
    print("ids[:12]:", ids[:12])
    back = tok.decode(ids)
    print("roundtrip ok:", back.strip() == sample.strip(), "|", repr(back[:60]))
    out = os.path.join(here, "artifacts", "bpe_tinyshakespeare_2k.json")
    tok.save(out)
    print("saved", out)
