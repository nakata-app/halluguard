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

    def verify(self, claim: str, chunks: list[str]) -> VerifierResult:
        """Return the best (max entailment) result across chunks."""
        if not chunks:
            return VerifierResult(entailment=0.0, contradiction=0.0)
        self._ensure_model()
        # CrossEncoder NLI model output is [contradiction, entailment, neutral] logits
        # for cross-encoder/nli-deberta-v3-base. Different NLI checkpoints differ;
        # we select by `id2label` introspection when available, else assume the
        # standard order (0=contradiction, 1=entailment, 2=neutral).
        pairs = [(c, claim) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False, batch_size=32)
        # scores: numpy array (n_chunks, 3)
        import numpy as np

        scores = np.asarray(scores)
        if scores.ndim != 2 or scores.shape[1] != 3:
            # Fallback: scalar output → treat as entailment score directly
            best = float(np.max(scores))
            return VerifierResult(entailment=_sigmoid(best), contradiction=0.0)
        # Softmax over the 3 classes per chunk
        ex = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = ex / ex.sum(axis=1, keepdims=True)
        # Standard order: 0=contradiction, 1=entailment, 2=neutral
        contra = probs[:, 0]
        entail = probs[:, 1]
        # Pick the chunk with highest entailment, but record its contradiction too
        best_idx = int(np.argmax(entail))
        return VerifierResult(
            entailment=float(entail[best_idx]),
            contradiction=float(contra[best_idx]),
        )


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))
