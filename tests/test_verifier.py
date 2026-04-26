"""Unit tests for NLIVerifier — model-free paths only.

We don't load the real CrossEncoder weights here; tests cover the structural
contract (signature, dataclass defaults, empty-chunks short-circuit, and the
question-aware premise construction by stubbing the model attribute).
"""
from __future__ import annotations

import numpy as np

from halluguard.verifier import NLIVerifier, VerifierResult


class _StubCrossEncoder:
    """Captures the pairs it would score and returns a fixed (n,3) logits array."""

    def __init__(self, logits: np.ndarray):
        self.logits = logits
        self.last_pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs, show_progress_bar=False, batch_size=32):
        self.last_pairs = list(pairs)
        return self.logits


def test_verifier_result_defaults_to_zero_votes():
    r = VerifierResult(entailment=0.4, contradiction=0.1)
    assert r.entail_votes == 0
    assert r.n_chunks == 0


def test_verify_empty_chunks_returns_zero_without_loading_model():
    v = NLIVerifier()
    # _model stays None — no network fetch
    out = v.verify(claim="anything", chunks=[])
    assert out.entailment == 0.0
    assert out.contradiction == 0.0
    assert out.n_chunks == 0
    assert out.entail_votes == 0
    assert v._model is None


def test_verify_counts_votes_and_picks_max_entailment():
    # Two chunks: first has high entailment, second has high contradiction.
    # Logits order: [contradiction, entailment, neutral]
    logits = np.array(
        [
            [-2.0, 3.0, 0.0],   # softmax → entailment ~0.99
            [4.0, -1.0, 0.0],   # softmax → contradiction ~0.99, entailment ~0.006
        ]
    )
    v = NLIVerifier()
    v._model = _StubCrossEncoder(logits)

    out = v.verify(claim="claim", chunks=["chunk-a", "chunk-b"], vote_threshold=0.5)
    assert out.n_chunks == 2
    assert out.entail_votes == 1  # only chunk-a passes
    assert out.entailment > 0.9   # max entailment came from chunk-a
    # The contradiction we record belongs to the chunk we picked (chunk-a),
    # which has low contradiction — not the chunk with high contradiction.
    assert out.contradiction < 0.1


def test_verify_question_changes_premise_format():
    logits = np.array([[-1.0, 2.0, 0.0]])
    v = NLIVerifier()
    stub = _StubCrossEncoder(logits)
    v._model = stub

    v.verify(claim="C", chunks=["the chunk"], question="What is X?")

    assert stub.last_pairs is not None
    premise, hyp = stub.last_pairs[0]
    assert premise == "Question: What is X?\nContext: the chunk"
    assert hyp == "C"


def test_verify_no_question_uses_chunk_as_premise():
    logits = np.array([[-1.0, 2.0, 0.0]])
    v = NLIVerifier()
    stub = _StubCrossEncoder(logits)
    v._model = stub

    v.verify(claim="C", chunks=["the chunk"])

    assert stub.last_pairs == [("the chunk", "C")]


def test_verify_vote_threshold_changes_count():
    # Two chunks both at entailment ~0.7
    logits = np.array(
        [
            [-1.0, 1.5, 0.0],
            [-1.0, 1.5, 0.0],
        ]
    )
    v = NLIVerifier()
    v._model = _StubCrossEncoder(logits)

    high = v.verify(claim="c", chunks=["a", "b"], vote_threshold=0.95)
    low = v.verify(claim="c", chunks=["a", "b"], vote_threshold=0.3)

    assert high.entail_votes == 0
    assert low.entail_votes == 2
