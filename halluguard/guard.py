"""High-level Guard API."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

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
        min_entail_votes: int = 1,
    ):
        self.index = index
        self.threshold = threshold
        self.top_k = top_k
        self.segmenter: Callable[[str], Iterable[str]] = segmenter or split_sentences
        self.verifier = verifier
        self.entail_threshold = entail_threshold
        # Multi-evidence vote: at least this many top-K chunks must clear
        # `entail_threshold` for the claim to count as SUPPORTED. `1` keeps
        # the existing max-only behaviour (any single passing chunk wins);
        # raise to e.g. 2 for a stricter "agreement of two" policy.
        self.min_entail_votes = min_entail_votes

    @classmethod
    def from_documents(
        cls,
        documents: list[str],
        encoder: Any,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> "Guard":
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=chunk_overlap)
        index = CorpusIndex(chunks, encoder=encoder)
        return cls(index=index, **kwargs)

    @classmethod
    def from_chunks(
        cls, chunks: list[Chunk], encoder: Any, **kwargs: Any
    ) -> "Guard":
        index = CorpusIndex(chunks, encoder=encoder)
        return cls(index=index, **kwargs)

    @classmethod
    def from_daemon(
        cls,
        documents: list[str],
        daemon_url: str = "http://127.0.0.1:7800",
        timeout_s: float = 10.0,
        api_key: str | None = None,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> "Guard":
        """Build a Guard whose encoder is an `adaptmem serve` daemon.

        Same surface as `from_documents`, but the SentenceTransformer load
        happens once inside the daemon process — across many Guard
        instances if you have several. Useful when:
        - you don't want to load a model per Python process (claimcheck +
          halluguard + metis would each pay the same MiniLM cost);
        - you want the encoder cached across worker restarts (daemon
          stays up).

        Auth: pass `api_key` (or set `ADAPTMEM_API_KEY` env) when the
        daemon is started with `--api-key` / `ADAPTMEM_API_KEY`.

        Quietly checks `/healthz` first so a misconfigured daemon URL
        fails loudly here, not deep inside the first `.check()` call.
        """
        from halluguard.daemon import DaemonEncoder

        encoder = DaemonEncoder(daemon_url=daemon_url, timeout_s=timeout_s, api_key=api_key)
        encoder.healthz()  # raises if unreachable
        return cls.from_documents(
            documents,
            encoder=encoder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )

    @classmethod
    def from_adaptmem(cls, am: Any, **kwargs: Any) -> "Guard":
        """Build a Guard backed by an adaptmem-tuned encoder + its corpus.

        Bridges adaptmem (domain-tuned retrieval) and halluguard (claim
        verification): the bi-encoder gate uses the tuned model and the
        already-encoded corpus, so retrieval reflects domain adaptation
        rather than the generic baseline.

        Hypothesis: a domain-tuned bi-encoder lifts cosine precision
        because lexically-similar but semantically-irrelevant chunks
        score lower → fewer false positives at the cosine gate → higher
        downstream Guard precision after NLI. Measure on RAGTruth /
        HaluEval QA before claiming the lift.

        Duck-typed: any object exposing `encoder`, `corpus`
        (list of objects with `id` and `text`), and `embeddings` works.
        Pass adaptmem.AdaptMem instances, or any equivalent shim.
        """
        if am.encoder is None or am.embeddings is None:
            raise RuntimeError(
                "AdaptMem instance is not initialised — call .train() or "
                ".load() before passing it to Guard.from_adaptmem()."
            )
        chunks = [Chunk(id=c.id, text=c.text) for c in am.corpus]
        index = CorpusIndex.from_precomputed(
            chunks=chunks, embeddings=am.embeddings, encoder=am.encoder
        )
        return cls(index=index, **kwargs)

    def _check_claim_text(self, ct: str, question: str | None = None) -> Claim:
        """Score a single claim text (no segmentation). Pulled out so
        check() and check_stream() can share the gate logic.
        """
        hits = self.index.search(ct, top_k=self.top_k)
        if not hits:
            return Claim(
                text=ct, status=ClaimStatus.HALLUCINATION_FLAG, support_score=0.0
            )
        best_score = hits[0][1]
        citation_ids = [c.id for c, _ in hits[: max(1, self.top_k // 2 + 1)]]
        cosine_pass = best_score >= self.threshold
        entail_pass = True
        entail_votes: int | None = None
        entail_chunks: int | None = None
        if self.verifier is not None and cosine_pass:
            top_chunk_texts = [c.text for c, _ in hits]
            vr = self.verifier.verify(
                ct,
                top_chunk_texts,
                question=question,
                vote_threshold=self.entail_threshold,
            )
            entail_pass = (
                vr.entailment >= self.entail_threshold
                and vr.entail_votes >= self.min_entail_votes
            )
            entail_votes = vr.entail_votes
            entail_chunks = vr.n_chunks
        status = (
            ClaimStatus.SUPPORTED
            if (cosine_pass and entail_pass)
            else ClaimStatus.HALLUCINATION_FLAG
        )
        return Claim(
            text=ct,
            status=status,
            support_score=best_score,
            citation_ids=citation_ids if status == ClaimStatus.SUPPORTED else [],
            entail_votes=entail_votes,
            entail_chunks=entail_chunks,
        )

    def check(self, answer: str, question: str | None = None) -> SupportReport:
        """Check `answer` against the indexed corpus.

        When `question` is given, it is forwarded to the NLI verifier as
        extra premise context (`Question: {question}\\nContext: {chunk}`).
        RAG-derived answers often restate the question implicitly; without
        it, NLI can fail to align an otherwise correct answer-from-context.
        """
        claim_texts = list(self.segmenter(answer))
        claims: list[Claim] = []
        for ct in claim_texts:
            hits = self.index.search(ct, top_k=self.top_k)
            if not hits:
                claims.append(
                    Claim(text=ct, status=ClaimStatus.HALLUCINATION_FLAG, support_score=0.0)
                )
                continue
            claims.append(self._check_claim_text(ct, question))
        return SupportReport(answer=answer, claims=claims, threshold=self.threshold)

    def check_stream(
        self,
        answer_chunks: Iterable[str],
        question: str | None = None,
    ) -> Iterator[Claim]:
        """Streaming variant of `check`.

        Feeds answer text in pieces (e.g. tokens or substrings as an LLM
        emits them), buffers until the segmenter produces a complete
        sentence, and yields a Claim per completed sentence. Useful for
        live LLM responses where waiting for the full answer is too slow
        — flag a hallucinated sentence the moment it lands.

        The final partial fragment is flushed when the iterator ends.

        Note: each yielded Claim carries the same metadata as
        `check()` claims (entail_votes, citation_ids, etc.). Aggregating
        the claims into a SupportReport is the caller's responsibility
        if a final summary is needed.
        """
        buf = ""
        for chunk in answer_chunks:
            buf += chunk
            sentences = list(self.segmenter(buf))
            if len(sentences) <= 1:
                continue
            # Yield all but the last sentence (which may still be partial)
            for s in sentences[:-1]:
                yield self._check_claim_text(s, question)
            buf = sentences[-1]
        # Flush remainder
        if buf.strip():
            yield self._check_claim_text(buf, question)
