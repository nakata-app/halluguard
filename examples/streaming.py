"""Streaming: flag a hallucinated sentence the moment it lands.

Pattern for live LLM responses where waiting for the full answer is too
slow. The user sees the verdict per-sentence as the LLM types.

Run:

    pip install -e ".[dev]"
    python examples/streaming.py
"""
from __future__ import annotations

import time
from typing import Iterator

from sentence_transformers import SentenceTransformer

from halluguard import Guard
from halluguard.verifier import NLIVerifier


def fake_llm_stream(text: str, chunk_chars: int = 12, delay_s: float = 0.05) -> Iterator[str]:
    """Yield the text in small chunks with a delay, simulating an LLM token stream."""
    for i in range(0, len(text), chunk_chars):
        time.sleep(delay_s)
        yield text[i : i + chunk_chars]


def main() -> None:
    documents = [
        "The Eiffel Tower is in Paris, completed in 1889.",
        "The Statue of Liberty is in New York Harbor, dedicated in 1886.",
        "The Great Wall of China was built over multiple dynasties starting in the 7th century BC.",
    ]
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    guard = Guard.from_documents(documents, encoder=encoder, verifier=NLIVerifier())

    answer = (
        "The Eiffel Tower is in Paris and was completed in 1889. "
        "The Eiffel Tower is also in Berlin, built in 1920. "  # false
        "The Statue of Liberty is in New York Harbor."
    )

    print(">>> streaming…")
    for claim in guard.check_stream(fake_llm_stream(answer), question="Where is the Eiffel Tower?"):
        marker = "ok  " if claim.status.value == "SUPPORTED" else "FLAG"
        print(f"  {marker}  score={claim.support_score:.2f}  {claim.text!r}")


if __name__ == "__main__":
    main()
