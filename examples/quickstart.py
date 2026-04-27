"""Quickstart: build a Guard, score one supported answer + one mixed.

Run from the repo root:

    pip install -e ".[dev]"
    python examples/quickstart.py

The first run downloads MiniLM (~90MB) and the NLI cross-encoder (~700MB)
on demand. CPU-only — no GPU required. ~1-2 minutes end-to-end on a Mac
mini, mostly model download.
"""
from sentence_transformers import SentenceTransformer

from halluguard import Guard
from halluguard.verifier import NLIVerifier


def main() -> None:
    documents = [
        "PostgreSQL ships native JSON and JSONB column types since version 9.4.",
        "MySQL added a JSON column type in version 5.7.7, released in 2015.",
        "SQLite has no native JSON column type; JSON is stored as TEXT and queried via the JSON1 extension.",
        "MongoDB stores documents in BSON, a binary-encoded superset of JSON.",
        "Redis can store JSON via the RedisJSON module but does not have a native JSON type.",
    ]

    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    guard = Guard.from_documents(
        documents,
        encoder=encoder,
        verifier=NLIVerifier(),
    )

    # 1. Fully grounded answer
    grounded = guard.check(
        "PostgreSQL has native JSON support since 9.4 and MySQL added JSON in 5.7.7.",
        question="Which databases have native JSON?",
    )
    print(f"GROUNDED  trust_score={grounded.trust_score:.3f}")
    for c in grounded.claims:
        marker = "ok  " if c.status.value == "SUPPORTED" else "FLAG"
        print(f"  {marker}  score={c.support_score:.2f}  {c.text!r}")

    print()

    # 2. Mixed: one fact + one hallucination
    mixed = guard.check(
        "PostgreSQL has native JSON since 9.4. Redis has had a native JSON column since version 4.0.",
        question="Which databases have native JSON?",
    )
    print(f"MIXED     trust_score={mixed.trust_score:.3f}")
    for c in mixed.claims:
        marker = "ok  " if c.status.value == "SUPPORTED" else "FLAG"
        print(f"  {marker}  score={c.support_score:.2f}  {c.text!r}")


if __name__ == "__main__":
    main()
