"""Retriever tests with a fake encoder (no model download in CI)."""
from __future__ import annotations

import numpy as np

from halluguard.retriever import Chunk, CorpusIndex, chunk_documents


class FakeEncoder:
    """Deterministic toy encoder: maps each unique word to a one-hot dim.

    Good enough to verify the index/search math without pulling weights.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _vec(self, text: str) -> np.ndarray:
        import re
        v = np.zeros(64, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w not in self.vocab:
                if len(self.vocab) >= 64:
                    continue
                self.vocab[w] = len(self.vocab)
            v[self.vocab[w]] += 1.0
        return v

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
        arr = np.stack([self._vec(t) for t in texts])
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            arr = arr / norms
        return arr


def test_chunk_documents_short():
    chunks = chunk_documents(["short doc"], chunk_size=200, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].id == "d0_c0"


def test_chunk_documents_long_overlap():
    long = " ".join(["w"] * 500)
    chunks = chunk_documents([long], chunk_size=200, overlap=50)
    assert len(chunks) >= 3
    assert chunks[0].id == "d0_c0"
    # IDs ascend
    assert chunks[1].id == "d0_c1"


def test_corpus_index_empty():
    idx = CorpusIndex(chunks=[], encoder=FakeEncoder())
    assert idx.search("anything", top_k=5) == []


def test_corpus_index_topk_ordering():
    enc = FakeEncoder()
    chunks = [
        Chunk(id="m1", text="postgres database query optimization"),
        Chunk(id="m2", text="apple banana fruit basket"),
        Chunk(id="m3", text="postgres index btree"),
    ]
    idx = CorpusIndex(chunks=chunks, encoder=enc)
    hits = idx.search("postgres index", top_k=2)
    assert len(hits) == 2
    # m3 shares 2 words ("postgres", "index"), m1 shares 1
    assert hits[0][0].id == "m3"
    assert hits[1][0].id == "m1"
    assert hits[0][1] >= hits[1][1]
