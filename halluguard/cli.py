"""halluguard <answer.txt> --corpus <dir>

Minimal CLI: read an answer file, build index from a corpus directory of .txt
files, run Guard.check, print markdown report. For programmatic use, import
`halluguard.Guard` directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    # Pre-route `serve` before main argparse so that answer positional paths
    # are not misidentified as subcommand names by argparse.
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        sp = argparse.ArgumentParser(prog="halluguard serve")
        sp.add_argument("--port", type=int, default=7801)
        sp.add_argument("--model", default="all-MiniLM-L6-v2")
        sargs = sp.parse_args(sys.argv[2:])
        from halluguard.server import serve
        serve(port=sargs.port, model_name=sargs.model)
        return

    ap = argparse.ArgumentParser(
        prog="halluguard",
        description="halluguard <answer.txt> --corpus <dir>  |  halluguard serve [--port N]",
    )

    ap.add_argument("answer", nargs="?", help="path to answer text file (or '-' for stdin)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--corpus", help="directory of .txt files (corpus)")
    grp.add_argument(
        "--corpus-text",
        help="Single string used as the entire corpus — useful for one-shot "
             "checks where a knowledge snippet is all you have.",
    )
    grp.add_argument(
        "--corpus-file",
        help="Path to a single text file used as the entire corpus.",
    )
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers encoder")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--format", choices=["markdown", "plain", "json"], default="markdown")
    ap.add_argument(
        "--question",
        default=None,
        help="Optional source question — when set, becomes part of the NLI premise "
             "(useful when the answer was generated to answer a specific question).",
    )
    ap.add_argument(
        "--nli",
        action="store_true",
        help="Enable the NLI cross-encoder verifier (claim-level entailment).",
    )
    ap.add_argument(
        "--nli-model",
        default="cross-encoder/nli-deberta-v3-base",
        help="NLI cross-encoder model (only used with --nli).",
    )
    ap.add_argument(
        "--entail-threshold",
        type=float,
        default=0.5,
        help="NLI entailment probability required for a claim to count as supported.",
    )
    ap.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help="Minimum number of top-K chunks that must clear --entail-threshold "
             "for a claim to count as SUPPORTED. Raise to require multi-evidence.",
    )
    args = ap.parse_args()

    if not args.answer:
        ap.error("answer file required when not using 'serve'")

    if args.answer == "-":
        answer = sys.stdin.read()
    else:
        answer = Path(args.answer).read_text()

    if args.corpus is None and args.corpus_text is None and args.corpus_file is None:
        ap.error("one of the arguments --corpus --corpus-text --corpus-file is required")

    if args.corpus_text is not None:
        documents = [args.corpus_text]
    elif args.corpus_file is not None:
        cf = Path(args.corpus_file)
        if not cf.is_file():
            print(f"corpus file not found: {cf}", file=sys.stderr)
            sys.exit(1)
        documents = [cf.read_text()]
    else:
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

    verifier = None
    if args.nli:
        from halluguard.verifier import NLIVerifier
        verifier = NLIVerifier(model_name=args.nli_model)

    guard = Guard.from_documents(
        documents=documents,
        encoder=encoder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        threshold=args.threshold,
        top_k=args.top_k,
        verifier=verifier,
        entail_threshold=args.entail_threshold,
        min_entail_votes=args.min_votes,
    )
    report = guard.check(answer, question=args.question)

    if args.format == "markdown":
        print(report.to_markdown())
    elif args.format == "json":
        import json as _json
        print(_json.dumps(report.to_dict(), indent=2))
    else:
        print(str(report))

    sys.exit(0 if report.n_flagged == 0 else 1)


if __name__ == "__main__":
    main()
