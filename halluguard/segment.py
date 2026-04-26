"""Atomic-claim segmentation.

Default: sentence-level split using a regex tuned for English/Turkish prose.
Pluggable: pass a custom callable to `Guard(segmenter=...)` for domain-specific
splitting (legal clauses, code comments, JSON fields, etc.).
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

# Sentence boundary: period/exclamation/question followed by whitespace + capital letter
# OR end of string. Tuned to avoid breaking on abbreviations like "e.g.", "Dr.", "vs.".
_ABBREV = {"e.g.", "i.e.", "vs.", "etc.", "dr.", "mr.", "mrs.", "ms.", "st.", "sr.", "jr."}
# Split on sentence terminator + whitespace. We DON'T require the next char
# to be uppercase — code-style identifiers ("tokio::spawn") and lowercase
# starts (Markdown-rendered prose) would otherwise be missed.
# Abbreviations are handled afterwards by re-merging on the abbrev list below.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences.

    Conservative: defers on abbreviations. Empty/whitespace-only inputs return [].
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and _ends_with_abbrev(out[-1]):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def _ends_with_abbrev(s: str) -> bool:
    # Cheap check: last token (lowercased) in abbrev list
    last = s.split()[-1].lower() if s.split() else ""
    return last in _ABBREV


Segmenter = Callable[[str], Iterable[str]]
"""Type alias for any callable that takes an answer string and yields claims."""
