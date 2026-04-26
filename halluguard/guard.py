"""High-level Guard API."""
from __future__ import annotations

from typing import Callable, Iterable

from halluguard.report import Claim, ClaimStatus, SupportReport
from halluguard.retriever import Chunk, CorpusIndex, chunk_documents
from halluguard.segment import Segmenter, split_sentences


class Guard:
    """Build once over a corpus, call `.check(answer)` repeatedly.

    Example:
        guard = Guard.from_documents(["..."], encoder=SentenceTransformer("all-MiniLM-L6-v2"))
        report = guard.check("The user prefers PostgreSQL.")
        print(report)
    """

    def __init__(
        self,
        index: CorpusIndex,
        threshold: float = 0.55,
        top_k: int = 5,
        segmenter: Segmenter | None = None,
    ):
        self.index = index
        self.threshold = threshold
        self.top_k = top_k
        self.segmenter: Callable[[str], Iterable[str]] = segmenter or split_sentences

    @classmethod
    def from_documents(
        cls,
        documents: list[str],
        encoder,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        **kwargs,
    ) -> "Guard":
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=chunk_overlap)
        index = CorpusIndex(chunks, encoder=encoder)
        return cls(index=index, **kwargs)

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], encoder, **kwargs) -> "Guard":
        index = CorpusIndex(chunks, encoder=encoder)
        return cls(index=index, **kwargs)

    def check(self, answer: str) -> SupportReport:
        claim_texts = list(self.segmenter(answer))
        claims: list[Claim] = []
        for ct in claim_texts:
            hits = self.index.search(ct, top_k=self.top_k)
            if not hits:
                claims.append(
                    Claim(text=ct, status=ClaimStatus.HALLUCINATION_FLAG, support_score=0.0)
                )
                continue
            best_score = hits[0][1]
            citation_ids = [c.id for c, _ in hits[: max(1, self.top_k // 2 + 1)]]
            status = (
                ClaimStatus.SUPPORTED
                if best_score >= self.threshold
                else ClaimStatus.HALLUCINATION_FLAG
            )
            claims.append(
                Claim(
                    text=ct,
                    status=status,
                    support_score=best_score,
                    citation_ids=citation_ids if status == ClaimStatus.SUPPORTED else [],
                )
            )
        return SupportReport(answer=answer, claims=claims, threshold=self.threshold)
