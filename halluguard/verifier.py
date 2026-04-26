"""NLI verifier: claim ↔ retrieved chunk entailment.

Bi-encoder retrieval gives semantic neighborhood; NLI gives logical alignment.
"This runs before render" can be a tight cosine match to "This runs after
render" yet logically contradict it. NLI catches that.

Default model: cross-encoder/nli-deberta-v3-base (entailment / neutral /
contradiction). Verifier returns the entailment probability for
(premise=chunk, hypothesis=claim). Higher = better support.

For each claim, we run NLI against the top-k retrieved chunks (small k, so
the cost is bounded) and take the maximum entailment score.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifierResult:
    entailment: float
    """Probability the chunk entails the claim (0–1). Higher = more supported."""
    contradiction: float
    """Probability the chunk contradicts the claim. Higher = stronger flag."""
    entail_votes: int = 0
    """Number of chunks (within the supplied top-K) whose entailment ≥ vote_threshold.
    Multi-evidence signal: callers can require ≥2 to flag less aggressively."""
    n_chunks: int = 0
    """How many chunks were scored (top-K). For computing vote ratio."""


class NLIVerifier:
    """Wraps a HuggingFace cross-encoder NLI model.

    Lazily loaded — model isn't fetched until the first .verify() call.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def verify(
        self,
        claim: str,
        chunks: list[str],
        question: str | None = None,
        vote_threshold: float = 0.5,
    ) -> VerifierResult:
        """Return the best (max entailment) result across chunks.

        - When `question` is supplied, the premise is constructed as
          `f"Question: {question}\\nContext: {chunk}"`. RAG-derived claims often
          restate the question; without the question the NLI head fails to
          align answer-from-context patterns even when the answer is correct.
        - `vote_threshold` is forwarded to the multi-evidence stat so callers
          can ask "did at least N chunks entail this?" — accessible via
          `VerifierResult.entail_votes` (number of chunks with entailment ≥
          vote_threshold).
        """
        if not chunks:
            return VerifierResult(entailment=0.0, contradiction=0.0)
        self._ensure_model()
        # CrossEncoder NLI model output is [contradiction, entailment, neutral] logits
        # for cross-encoder/nli-deberta-v3-base. Different NLI checkpoints differ;
        # we select by `id2label` introspection when available, else assume the
        # standard order (0=contradiction, 1=entailment, 2=neutral).
        if question is not None:
            premises = [f"Question: {question}\nContext: {c}" for c in chunks]
        else:
            premises = chunks
        pairs = [(p, claim) for p in premises]
        scores = self._model.predict(pairs, show_progress_bar=False, batch_size=32)
        # scores: numpy array (n_chunks, 3)
        import numpy as np

        scores = np.asarray(scores)
        if scores.ndim != 2 or scores.shape[1] != 3:
            # Fallback: scalar output → treat as entailment score directly
            best = float(np.max(scores))
            return VerifierResult(
                entailment=_sigmoid(best),
                contradiction=0.0,
                entail_votes=1 if _sigmoid(best) >= vote_threshold else 0,
                n_chunks=len(chunks),
            )
        # Softmax over the 3 classes per chunk
        ex = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = ex / ex.sum(axis=1, keepdims=True)
        # Standard order: 0=contradiction, 1=entailment, 2=neutral
        contra = probs[:, 0]
        entail = probs[:, 1]
        # Pick the chunk with highest entailment, but record its contradiction too
        best_idx = int(np.argmax(entail))
        votes = int((entail >= vote_threshold).sum())
        return VerifierResult(
            entailment=float(entail[best_idx]),
            contradiction=float(contra[best_idx]),
            entail_votes=votes,
            n_chunks=len(chunks),
        )


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))
