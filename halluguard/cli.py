"""halluguard <answer.txt> --corpus <dir>

Minimal CLI: read an answer file, build index from a corpus directory of .txt
files, run Guard.check, print markdown report. For programmatic use, import
`halluguard.Guard` directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(prog="halluguard")
    ap.add_argument("answer", help="path to answer text file (or '-' for stdin)")
    ap.add_argument("--corpus", required=True, help="directory of .txt files (corpus)")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers encoder")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--format", choices=["markdown", "plain"], default="markdown")
    args = ap.parse_args()

    if args.answer == "-":
        answer = sys.stdin.read()
    else:
        answer = Path(args.answer).read_text()

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        print(f"corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)
    documents = [p.read_text() for p in sorted(corpus_dir.glob("*.txt"))]
    if not documents:
        print(f"no .txt files in {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    from sentence_transformers import SentenceTransformer

    from halluguard import Guard

    encoder = SentenceTransformer(args.model)
    guard = Guard.from_documents(
        documents=documents,
        encoder=encoder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        threshold=args.threshold,
        top_k=args.top_k,
    )
    report = guard.check(answer)

    if args.format == "markdown":
        print(report.to_markdown())
    else:
        print(str(report))

    sys.exit(0 if report.n_flagged == 0 else 1)


if __name__ == "__main__":
    main()
