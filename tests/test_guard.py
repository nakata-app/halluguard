"""End-to-end Guard test with FakeEncoder."""
from __future__ import annotations

from halluguard import Guard, ClaimStatus
from halluguard.retriever import Chunk
from halluguard.verifier import VerifierResult
from tests.test_retriever import FakeEncoder


class _RecordingVerifier:
    """Captures kwargs the Guard passes to verify()."""

    def __init__(self):
        self.calls: list[dict] = []

    def verify(self, claim, chunks, question=None, vote_threshold=0.5):
        self.calls.append({"claim": claim, "chunks": chunks, "question": question})
        return VerifierResult(entailment=0.99, contradiction=0.0)


def test_guard_flags_unsupported_claim():
    documents = [
        "User Atakan prefers PostgreSQL because it has good JSON support.",
        "Atakan dislikes MongoDB due to inconsistent transaction semantics.",
    ]
    guard = Guard.from_documents(
        documents=documents, encoder=FakeEncoder(), threshold=0.4
    )
    answer = (
        "Atakan prefers PostgreSQL. The user owns three pet llamas and lives on Mars."
    )
    report = guard.check(answer)
    assert len(report.claims) == 2
    # First claim should be supported (overlap with corpus)
    assert report.claims[0].status == ClaimStatus.SUPPORTED
    # Second claim has zero corpus overlap (llamas, Mars)
    assert report.claims[1].status == ClaimStatus.HALLUCINATION_FLAG


def test_guard_empty_answer():
    guard = Guard.from_documents(documents=["doc"], encoder=FakeEncoder())
    report = guard.check("")
    assert report.claims == []
    assert report.support_rate == 0.0


def test_guard_threshold_effect():
    documents = ["postgres index btree gin"]
    enc = FakeEncoder()
    # Loose threshold: even partial-overlap claims pass
    loose = Guard.from_documents(documents=documents, encoder=enc, threshold=0.1)
    # Tight threshold: only near-perfect-overlap passes
    tight = Guard.from_documents(documents=documents, encoder=enc, threshold=0.9)
    answer = "postgres uses btree."
    assert loose.check(answer).n_flagged <= tight.check(answer).n_flagged


def test_report_markdown_renders():
    guard = Guard.from_documents(documents=["a sample doc"], encoder=FakeEncoder())
    md = guard.check("a sample claim. another claim.").to_markdown()
    assert "# Halluguard report" in md
    assert "claims:" in md


def test_guard_check_propagates_question_to_verifier():
    documents = ["postgres is great for json"]
    rv = _RecordingVerifier()
    guard = Guard.from_documents(
        documents=documents,
        encoder=FakeEncoder(),
        threshold=0.1,  # loose so cosine gate passes and verifier runs
        verifier=rv,
    )
    guard.check("postgres handles json well.", question="how does postgres handle json?")
    assert rv.calls, "verifier should have been invoked"
    assert rv.calls[0]["question"] == "how does postgres handle json?"


def test_guard_check_default_question_is_none():
    documents = ["postgres is great for json"]
    rv = _RecordingVerifier()
    guard = Guard.from_documents(
        documents=documents, encoder=FakeEncoder(), threshold=0.1, verifier=rv
    )
    guard.check("postgres handles json well.")
    assert rv.calls and rv.calls[0]["question"] is None


class _FixedVerifier:
    """Returns a fixed VerifierResult — used to drive vote-policy logic."""

    def __init__(self, result: VerifierResult):
        self.result = result

    def verify(self, claim, chunks, question=None, vote_threshold=0.5):
        return self.result


def test_guard_min_votes_blocks_when_only_one_chunk_entails():
    # Verifier returns: high max entailment but only 1 vote out of 5
    vr = VerifierResult(entailment=0.95, contradiction=0.0, entail_votes=1, n_chunks=5)
    guard = Guard.from_documents(
        documents=["postgres is good for json"],
        encoder=FakeEncoder(),
        threshold=0.1,
        verifier=_FixedVerifier(vr),
        entail_threshold=0.5,
        min_entail_votes=2,  # require at least 2 chunks to entail
    )
    report = guard.check("postgres handles json.")
    # Single supporting chunk falls below the 2-vote requirement → flagged
    assert report.claims[0].status == ClaimStatus.HALLUCINATION_FLAG


def test_guard_min_votes_allows_when_enough_chunks_entail():
    vr = VerifierResult(entailment=0.95, contradiction=0.0, entail_votes=3, n_chunks=5)
    guard = Guard.from_documents(
        documents=["postgres is good for json"],
        encoder=FakeEncoder(),
        threshold=0.1,
        verifier=_FixedVerifier(vr),
        entail_threshold=0.5,
        min_entail_votes=2,
    )
    report = guard.check("postgres handles json.")
    assert report.claims[0].status == ClaimStatus.SUPPORTED


def test_guard_default_min_votes_preserves_legacy_behaviour():
    # Default min_entail_votes=1 — a single passing chunk should be enough
    vr = VerifierResult(entailment=0.95, contradiction=0.0, entail_votes=1, n_chunks=5)
    guard = Guard.from_documents(
        documents=["postgres is good for json"],
        encoder=FakeEncoder(),
        threshold=0.1,
        verifier=_FixedVerifier(vr),
        entail_threshold=0.5,
    )
    report = guard.check("postgres handles json.")
    assert report.claims[0].status == ClaimStatus.SUPPORTED


def test_claim_carries_vote_metadata_when_verifier_runs():
    vr = VerifierResult(entailment=0.95, contradiction=0.0, entail_votes=3, n_chunks=5)
    guard = Guard.from_documents(
        documents=["postgres is good for json"],
        encoder=FakeEncoder(),
        threshold=0.1,
        verifier=_FixedVerifier(vr),
        entail_threshold=0.5,
    )
    claim = guard.check("postgres handles json.").claims[0]
    assert claim.entail_votes == 3
    assert claim.entail_chunks == 5
    assert claim.vote_str == "3/5"


def test_claim_vote_metadata_none_without_verifier():
    guard = Guard.from_documents(documents=["postgres"], encoder=FakeEncoder())
    claim = guard.check("postgres is good.").claims[0]
    assert claim.entail_votes is None
    assert claim.entail_chunks is None
    assert claim.vote_str == "—"


# ---- from_chunks: pre-chunked corpus path ---------------------------------


def test_guard_from_chunks_uses_supplied_ids():
    """When the caller already has chunks (e.g. from a custom splitter or
    an external store), Guard.from_chunks must use those ids verbatim and
    not re-chunk. The user-controlled id is what cites back to their source."""
    chunks = [
        Chunk(id="my-source/intro", text="postgres handles json natively"),
        Chunk(id="my-source/conclusion", text="redis is an in-memory cache"),
    ]
    guard = Guard.from_chunks(chunks=chunks, encoder=FakeEncoder(), threshold=0.1)
    report = guard.check("postgres handles json well.")
    supported = [c for c in report.claims if c.status == ClaimStatus.SUPPORTED]
    assert supported, "expected at least one supported claim"
    # The cited id is the one the caller passed in — not a re-derived d0_c0
    assert any("my-source/" in cid for cid in supported[0].citation_ids)


def test_guard_from_chunks_empty_chunks_flags_everything():
    """An empty chunks list yields an empty index → every claim flagged."""
    guard = Guard.from_chunks(chunks=[], encoder=FakeEncoder())
    report = guard.check("any claim at all.")
    assert all(c.status == ClaimStatus.HALLUCINATION_FLAG for c in report.claims)
    assert all(c.support_score == 0.0 for c in report.claims)


def test_guard_from_chunks_passes_kwargs_to_init():
    """Keyword-only args (top_k, threshold, verifier...) must reach Guard.__init__."""
    chunks = [Chunk(id="c0", text="postgres is good")]
    rv = _RecordingVerifier()
    guard = Guard.from_chunks(
        chunks=chunks,
        encoder=FakeEncoder(),
        threshold=0.1,
        top_k=3,
        verifier=rv,
        min_entail_votes=2,
    )
    assert guard.threshold == 0.1
    assert guard.top_k == 3
    assert guard.verifier is rv
    assert guard.min_entail_votes == 2


# ---- from_adaptmem: bridge to adaptmem-tuned encoders ---------------------


class _AdaptMemEntry:
    """Duck-typed CorpusEntry stand-in (id + text)."""

    def __init__(self, id: str, text: str):
        self.id = id
        self.text = text


class _StubAdaptMem:
    """Minimal AdaptMem-shaped object with the three properties Guard reads.

    Avoids importing the real adaptmem package in halluguard's test
    suite (which has no dep on adaptmem). The duck-typed contract is
    what we want to lock down anyway.
    """

    def __init__(self, encoder, corpus, embeddings):
        self.encoder = encoder
        self.corpus = corpus
        self.embeddings = embeddings


def test_guard_from_adaptmem_reuses_precomputed_embeddings():
    """The whole point of from_adaptmem is to skip re-encoding — assert
    the encoder is *not* called for the corpus side. Query side still uses
    it via search()."""
    enc = FakeEncoder()
    # Pre-compute the corpus embeddings ourselves so we know the matrix
    # the index will use; assert the encoder's vocab grew only when query
    # encoding happened, not corpus encoding.
    import numpy as np
    corpus = [
        _AdaptMemEntry(id="x1", text="postgres handles json natively"),
        _AdaptMemEntry(id="x2", text="redis is an in-memory cache"),
    ]
    pre = enc.encode([c.text for c in corpus])
    pre_vocab = dict(enc.vocab)  # snapshot

    am = _StubAdaptMem(encoder=enc, corpus=corpus, embeddings=pre)
    guard = Guard.from_adaptmem(am, threshold=0.1)

    # Index must hold the supplied embeddings (object identity, not a re-encode)
    assert guard.index.embeddings is pre
    # Index should preserve the user-supplied chunk ids
    assert [c.id for c in guard.index.chunks] == ["x1", "x2"]
    # Vocab snapshot before from_adaptmem matches: no extra encode happened
    assert dict(enc.vocab) == pre_vocab


def test_guard_from_adaptmem_uses_tuned_encoder_for_search():
    """When a query comes in, the AdaptMem-tuned encoder is the one called."""
    enc = FakeEncoder()
    import numpy as np
    corpus = [
        _AdaptMemEntry(id="y1", text="postgres json"),
        _AdaptMemEntry(id="y2", text="apple banana"),
    ]
    pre = enc.encode([c.text for c in corpus])
    am = _StubAdaptMem(encoder=enc, corpus=corpus, embeddings=pre)
    guard = Guard.from_adaptmem(am, threshold=0.1)
    report = guard.check("postgres json query.")
    # First claim should be SUPPORTED (cosine > 0.1 against y1)
    assert report.claims[0].status == ClaimStatus.SUPPORTED
    # Citation must come from the user-supplied corpus
    assert any("y" in cid for cid in report.claims[0].citation_ids)


def test_guard_from_adaptmem_rejects_uninitialised():
    """Passing a fresh, untrained AdaptMem must error rather than crash
    obscurely inside the retriever."""
    am = _StubAdaptMem(encoder=None, corpus=[], embeddings=None)
    import pytest
    with pytest.raises(RuntimeError, match="not initialised"):
        Guard.from_adaptmem(am)
