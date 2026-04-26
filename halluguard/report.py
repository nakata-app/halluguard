"""Result types: Claim, SupportReport."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    HALLUCINATION_FLAG = "HALLUCINATION_FLAG"


@dataclass
class Claim:
    text: str
    status: ClaimStatus
    support_score: float
    """Best alignment score against retrieved corpus (0.0–1.0). Higher = more supported."""
    citation_ids: list[str] = field(default_factory=list)
    """Top-k corpus chunk ids that align with this claim, descending by score."""
    entail_votes: int | None = None
    """How many of the top-K chunks cleared the entailment threshold. `None`
    means the NLI verifier was not invoked (cosine-only mode or cosine-failed
    short-circuit)."""
    entail_chunks: int | None = None
    """Top-K size the verifier scored against (denominator for entail_votes)."""

    @property
    def vote_str(self) -> str:
        if self.entail_votes is None or self.entail_chunks is None:
            return "—"
        return f"{self.entail_votes}/{self.entail_chunks}"

    def __str__(self) -> str:
        cites = ", ".join(self.citation_ids) if self.citation_ids else "—"
        return (
            f"[{self.status.value:18s}] score={self.support_score:.3f} "
            f"votes={self.vote_str} cites={cites}  {self.text!r}"
        )


@dataclass
class SupportReport:
    answer: str
    claims: list[Claim]
    threshold: float

    @property
    def n_supported(self) -> int:
        return sum(1 for c in self.claims if c.status == ClaimStatus.SUPPORTED)

    @property
    def n_flagged(self) -> int:
        return sum(1 for c in self.claims if c.status == ClaimStatus.HALLUCINATION_FLAG)

    @property
    def support_rate(self) -> float:
        if not self.claims:
            return 0.0
        return self.n_supported / len(self.claims)

    def to_markdown(self) -> str:
        out = ["# Halluguard report", ""]
        out.append(f"- claims: {len(self.claims)}")
        out.append(f"- supported: {self.n_supported}")
        out.append(f"- flagged: {self.n_flagged}")
        out.append(f"- support rate: {self.support_rate:.1%}")
        out.append(f"- threshold: {self.threshold:.2f}")
        out.append("")
        out.append("| status | score | votes | cites | claim |")
        out.append("|---|---|---|---|---|")
        for c in self.claims:
            cites = ", ".join(c.citation_ids) or "—"
            text = c.text.replace("|", "\\|")
            out.append(
                f"| {c.status.value} | {c.support_score:.3f} | {c.vote_str} | {cites} | {text} |"
            )
        return "\n".join(out)

    def __str__(self) -> str:
        lines = [
            f"SupportReport: {self.n_supported}/{len(self.claims)} supported "
            f"({self.support_rate:.1%}), {self.n_flagged} flagged, threshold={self.threshold:.2f}"
        ]
        for c in self.claims:
            lines.append("  " + str(c))
        return "\n".join(lines)

    @property
    def trust_score(self) -> float:
        """Response-level scalar in [0, 1] suitable for routing / gating.

        Mean of `support_score` across claims. 1.0 = every claim aligns
        with the corpus; 0.0 = no support at all. Empty answer → 0.0.
        """
        if not self.claims:
            return 0.0
        return sum(c.support_score for c in self.claims) / len(self.claims)

    def to_dict(self) -> dict:
        """JSON-ready representation. Suitable for `json.dumps`.

        Useful as the wire format for middleware / CI / dashboards that
        want to gate on `trust_score` or inspect individual flagged claims.
        """
        return {
            "answer": self.answer,
            "threshold": self.threshold,
            "n_claims": len(self.claims),
            "n_supported": self.n_supported,
            "n_flagged": self.n_flagged,
            "support_rate": self.support_rate,
            "trust_score": self.trust_score,
            "claims": [
                {
                    "text": c.text,
                    "status": c.status.value,
                    "support_score": c.support_score,
                    "citation_ids": list(c.citation_ids),
                    "entail_votes": c.entail_votes,
                    "entail_chunks": c.entail_chunks,
                }
                for c in self.claims
            ],
        }
