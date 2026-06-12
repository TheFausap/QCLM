"""Streaming corpus -> packed token blocks for the QCLM scale-up.

On the DGX Spark this streams FineWeb / FineWeb-Edu from the HuggingFace Hub
(no full download: `streaming=True`), tokenizes on the fly with the byte-level
BPE, separates documents with <|endoftext|>, and packs the token stream into
contiguous blocks of length `block_size+1`.

A LOCAL fallback (a plain text file split into pseudo-documents) lets the exact
same packing/batching path be tested without Hub access.
"""
from __future__ import annotations
import os
from typing import Iterable, Iterator
import numpy as np
import torch


# ---------------------------------------------------------------------------
def fineweb_doc_iter(name: str = "sample-10BT", split: str = "train",
                     dataset: str = "HuggingFaceFW/fineweb-edu",
                     text_key: str = "text", skip_docs: int = 0) -> Iterator[str]:
    """Stream documents from FineWeb(-Edu). Requires internet (use on the Spark).

    skip_docs: fast-forward the stream past the first `skip_docs` documents
    (used to resume training at the correct data position instead of
    re-reading the corpus from the start)."""
    from datasets import load_dataset
    ds = load_dataset(dataset, name=name, split=split, streaming=True)
    if skip_docs:
        ds = ds.skip(skip_docs)
    for ex in ds:
        t = ex.get(text_key)
        if t:
            yield t


def local_doc_iter(path: str, doc_sep: str = "\n\n",
                   skip_docs: int = 0) -> Iterator[str]:
    """Split a local text file into pseudo-documents (for offline testing)."""
    text = open(path, "r", encoding="utf-8").read()
    seen = 0
    for doc in text.split(doc_sep):
        doc = doc.strip()
        if doc:
            seen += 1
            if seen <= skip_docs:
                continue
            yield doc


class DocCounter:
    """Tracks the absolute document position of a stream.

    `base` is the number of documents skipped before this process attached
    (restored from a checkpoint); `count` is what this process has consumed.
    `total` is therefore an absolute position in the (deterministic) stream,
    suitable for checkpointing and for `skip_docs` on the next resume.
    """

    def __init__(self, base: int = 0):
        self.base = int(base)
        self.count = 0

    @property
    def total(self) -> int:
        return self.base + self.count

    def wrap(self, it: Iterable[str]) -> Iterator[str]:
        for doc in it:
            self.count += 1
            yield doc


# ---------------------------------------------------------------------------
def iter_token_blocks(doc_iter: Iterable[str], tokenizer, block_size: int,
                      eot_id: int, loop: bool = False) -> Iterator[np.ndarray]:
    """Tokenize + pack into contiguous (block_size+1,) int arrays."""
    need = block_size + 1
    buf: list[int] = []
    while True:
        for doc in doc_iter:
            ids = tokenizer.encode(doc)
            buf.extend(ids)
            buf.append(eot_id)
            while len(buf) >= need:
                yield np.asarray(buf[:need], dtype=np.int64)
                buf = buf[block_size:]   # overlap-by-1 so every token is a target
        if not loop:
            return
        # `loop=True`: restart the doc iterator (only meaningful for finite/local).
        if hasattr(doc_iter, "__iter__") and not hasattr(doc_iter, "__next__"):
            continue
        return


def batch_iter(block_iter: Iterable[np.ndarray], batch_size: int) -> Iterator[torch.Tensor]:
    """Stack packed blocks into (B, block_size+1) int64 tensors."""
    batch: list[np.ndarray] = []
    for block in block_iter:
        batch.append(block)
        if len(batch) == batch_size:
            yield torch.from_numpy(np.stack(batch))
            batch = []
    if batch:
        yield torch.from_numpy(np.stack(batch))


def make_stream(tokenizer, block_size: int, batch_size: int, eot_id: int,
                source: str = "fineweb", skip_docs: int = 0,
                counter: "DocCounter | None" = None, **kw) -> Iterator[torch.Tensor]:
    """Build the packed-batch stream.

    skip_docs: fast-forward past this many documents (resume support).
        Document-granular: the partial packing buffer at the skip point is
        not reconstructed, so the stream realigns within ~1 block.
    counter: optional DocCounter; if given, every document consumed by this
        stream advances it, so its .total can be checkpointed.
    """
    if source == "fineweb":
        docs = fineweb_doc_iter(skip_docs=skip_docs,
                                **{k: kw[k] for k in ("name", "split", "dataset", "text_key") if k in kw})
        loop = False
    else:  # local file path in kw['path']
        docs = local_doc_iter(kw["path"], kw.get("doc_sep", "\n\n"), skip_docs=skip_docs)
        loop = kw.get("loop", True)
    if counter is not None:
        docs = counter.wrap(docs)
    blocks = iter_token_blocks(docs, tokenizer, block_size, eot_id, loop=loop)
    return batch_iter(blocks, batch_size)


if __name__ == "__main__":
    # Offline smoke test of the full packing/batching path on tiny-shakespeare.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qlm.tokenizer import BPETokenizer
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tok = BPETokenizer.load(os.path.join(here, "artifacts", "bpe_tinyshakespeare_2k.json"))
    stream = make_stream(tok, block_size=64, batch_size=8, eot_id=tok.eot_id,
                         source="local", path=os.path.join(here, "data", "tinyshakespeare.txt"),
                         loop=False)
    b = next(stream)
    print("batch shape:", tuple(b.shape), "dtype:", b.dtype)
    print("decoded[0]:", repr(tok.decode(b[0, :40].tolist())))
    n = sum(x.shape[0] for x in [b] + [next(stream) for _ in range(20)])
    print("pulled ~", n, "sequences across 21 batches OK")
