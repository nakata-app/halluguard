"""End-to-end Guard test with FakeEncoder."""
from __future__ import annotations

from halluguard import Guard, ClaimStatus
from tests.test_retriever import FakeEncoder


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
