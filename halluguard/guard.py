"""High-level Guard API."""
from __future__ import annotations

from typing import Callable, Iterable

from halluguard.report import Claim, ClaimStatus, SupportReport
from halluguard.retriever import Chunk, CorpusIndex, chunk_documents
from halluguard.segment import Segmenter, split_sentences
from halluguard.verifier import NLIVerifier


class Guard:
    """Build once over a corpus, call `.check(answer)` repeatedly.

    Example:
        guard = Guard.from_documents(["..."], encoder=SentenceTransformer("all-MiniLM-L6-v2"))
        report = guard.check("The user prefers PostgreSQL.")
        print(report)

    NLI verifier (optional, recommended): catches subtle contradictions that
    bi-encoder cosine misses ("runs before render" vs "runs after render" share
    most words but logically contradict). Pass `verifier=NLIVerifier()` to
    enable. When enabled, a claim is SUPPORTED iff (cosine ≥ threshold AND
    entailment ≥ entail_threshold).
    """

    def __init__(
        self,
        index: CorpusIndex,
        threshold: float = 0.55,
        top_k: int = 5,
        segmenter: Segmenter | None = None,
        verifier: NLIVerifier | None = None,
        entail_threshold: float = 0.5,
    ):
        self.index = index
        self.threshold = threshold
        self.top_k = top_k
        self.segmenter: Callable[[str], Iterable[str]] = segmenter or split_sentences
        self.verifier = verifier
        self.entail_threshold = entail_threshold

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

            # Stage 1: cosine threshold gate
            cosine_pass = best_score >= self.threshold
            # Stage 2: NLI entailment (optional, only when cosine passes — cheaper)
            entail_pass = True
            if self.verifier is not None and cosine_pass:
                top_chunk_texts = [c.text for c, _ in hits]
                vr = self.verifier.verify(ct, top_chunk_texts)
                entail_pass = vr.entailment >= self.entail_threshold

            status = (
                ClaimStatus.SUPPORTED
                if (cosine_pass and entail_pass)
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
