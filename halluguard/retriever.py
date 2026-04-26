"""Bi-encoder retrieval over a corpus of chunks.

Same model you use for normal RAG. Built once, queried many times.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    id: str
    text: str


class CorpusIndex:
    """In-memory L2-normalized embedding index over corpus chunks.

    For very large corpora (>100k chunks) you'd swap this for FAISS or HNSW;
    the API contract (`encode`, `search`) stays the same.
    """

    def __init__(self, chunks: list[Chunk], encoder):
        self.chunks = chunks
        self.encoder = encoder
        if not chunks:
            self.embeddings = np.zeros((0, 1), dtype=np.float32)
        else:
            self.embeddings = encoder.encode(
                [c.text for c in chunks],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=64,
            )

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, cosine_score) by descending similarity."""
        if not self.chunks:
            return []
        q = self.encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ q  # (N,)
        k = min(top_k, len(self.chunks))
        # argpartition for top-k, then sort that slice
        idx_part = np.argpartition(-scores, k - 1)[:k]
        idx_sorted = idx_part[np.argsort(-scores[idx_part])]
        return [(self.chunks[i], float(scores[i])) for i in idx_sorted]


def chunk_documents(
    documents: list[str], chunk_size: int = 200, overlap: int = 50
) -> list[Chunk]:
    """Split documents into word-level sliding-window chunks.

    Document index becomes part of the chunk id (`d{di}_c{ci}`) so a flagged
    claim cites back to a specific span. For richer ids, build chunks yourself.
    """
    out: list[Chunk] = []
    for di, doc in enumerate(documents):
        words = doc.split()
        if not words:
            continue
        if len(words) <= chunk_size:
            out.append(Chunk(id=f"d{di}_c0", text=doc))
            continue
        step = max(1, chunk_size - overlap)
        ci = 0
        i = 0
        while i < len(words):
            end = min(i + chunk_size, len(words))
            out.append(Chunk(id=f"d{di}_c{ci}", text=" ".join(words[i:end])))
            if end == len(words):
                break
            i += step
            ci += 1
    return out
