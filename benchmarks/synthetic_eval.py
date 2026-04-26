"""Synthetic hallucination benchmark for halluguard.

Each example is (corpus, answer, gold_labels) where gold_labels[i] is
'SUPPORTED' or 'HALLUCINATION' for the i-th claim of `answer`.

Reports per-threshold precision/recall/F1 against the gold labels and
identifies the best operating point.

Run:
  python benchmarks/synthetic_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow importing the package without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from halluguard import Guard, ClaimStatus
from halluguard.segment import split_sentences


# ---- Dataset --------------------------------------------------------------
EXAMPLES = [
    {
        "name": "postgres-prefs",
        "corpus": [
            "Atakan prefers PostgreSQL for application data.",
            "PostgreSQL has strong JSON support and ACID guarantees.",
            "Atakan dislikes MongoDB due to inconsistent transaction semantics.",
        ],
        "answer": (
            "Atakan prefers PostgreSQL because it has strong JSON support. "
            "He uses MongoDB as the primary store for everything. "
            "He owns three pet llamas that live on Mars."
        ),
        "gold": ["SUPPORTED", "HALLUCINATION", "HALLUCINATION"],
    },
    {
        "name": "rust-async",
        "corpus": [
            "Tokio is an async runtime for Rust applications.",
            "tokio::spawn schedules a task on the runtime thread pool.",
            "Channels (mpsc, oneshot, broadcast) move data between tasks.",
        ],
        "answer": (
            "Tokio is an async runtime for Rust. "
            "tokio::spawn places a task on the runtime. "
            "Tokio is written in Java and only runs on Windows."
        ),
        "gold": ["SUPPORTED", "SUPPORTED", "HALLUCINATION"],
    },
    {
        "name": "git-rebase",
        "corpus": [
            "Interactive rebase rewrites commit history before publishing.",
            "Squash combines a commit with the previous one.",
            "Fixup squashes silently without editing the message.",
        ],
        "answer": (
            "Interactive rebase rewrites commit history. "
            "Fixup combines a commit with the previous one silently. "
            "Rebase automatically deletes the remote main branch."
        ),
        "gold": ["SUPPORTED", "SUPPORTED", "HALLUCINATION"],
    },
    {
        "name": "react-hooks",
        "corpus": [
            "React hooks must be called at the top level of components.",
            "useEffect runs after render; the dependency array controls re-runs.",
            "ESLint rule react-hooks/exhaustive-deps catches missing deps.",
        ],
        "answer": (
            "React hooks must be called at the top level. "
            "useEffect runs before render and never inside components. "
            "ESLint catches missing dependencies via exhaustive-deps."
        ),
        "gold": ["SUPPORTED", "HALLUCINATION", "SUPPORTED"],
    },
    {
        "name": "postgres-index",
        "corpus": [
            "PostgreSQL B-tree is the default index access method.",
            "GIN handles full-text search and JSON containment.",
            "Partial indexes are filtered by a WHERE clause.",
        ],
        "answer": (
            "B-tree is the default Postgres index. "
            "GIN handles JSON containment lookups. "
            "Postgres uses LSM-trees for all indexes by default."
        ),
        "gold": ["SUPPORTED", "SUPPORTED", "HALLUCINATION"],
    },
    {
        "name": "swift-concurrency",
        "corpus": [
            "Swift 6 introduces strict concurrency checking.",
            "Actors isolate mutable state and serialize access.",
            "Sendable protocol marks types safe to cross actor boundaries.",
        ],
        "answer": (
            "Swift 6 has strict concurrency. "
            "Actors isolate mutable state. "
            "Swift removed all concurrency features in version 6."
        ),
        "gold": ["SUPPORTED", "SUPPORTED", "HALLUCINATION"],
    },
]


def _eval_at_threshold(guard: Guard, ex: dict, threshold: float) -> tuple[int, int, int, int]:
    """Return (true_pos, false_pos, true_neg, false_neg) where 'positive' = HALLUCINATION."""
    guard.threshold = threshold
    report = guard.check(ex["answer"])
    if len(report.claims) != len(ex["gold"]):
        # Mismatched segmentation — skip example for this run (fairness)
        return (0, 0, 0, 0)
    tp = fp = tn = fn = 0
    for c, gold in zip(report.claims, ex["gold"]):
        pred_hallu = c.status == ClaimStatus.HALLUCINATION_FLAG
        gold_hallu = gold == "HALLUCINATION"
        if pred_hallu and gold_hallu:
            tp += 1
        elif pred_hallu and not gold_hallu:
            fp += 1
        elif not pred_hallu and not gold_hallu:
            tn += 1
        else:
            fn += 1
    return (tp, fp, tn, fn)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", action="store_true", help="enable NLI verifier (cross-encoder)")
    ap.add_argument(
        "--nli-model", default="cross-encoder/nli-deberta-v3-base", help="NLI cross-encoder model"
    )
    ap.add_argument("--entail-threshold", type=float, default=0.5)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    print("loading sentence-transformers/all-MiniLM-L6-v2…", file=sys.stderr, flush=True)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    verifier = None
    if args.nli:
        from halluguard.verifier import NLIVerifier

        print(f"loading NLI cross-encoder {args.nli_model}…", file=sys.stderr, flush=True)
        verifier = NLIVerifier(model_name=args.nli_model)

    # Build a single guard per example (corpora differ)
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    print()
    print(f"# Halluguard synthetic benchmark — {len(EXAMPLES)} examples")
    print(f"# Encoder: all-MiniLM-L6-v2  |  Positive class = HALLUCINATION")
    print()
    print("| threshold | precision | recall | F1     | TP | FP | FN | TN |")
    print("|-----------|-----------|--------|--------|----|----|----|----|")

    best = (0.0, 0.0, 0.0, 0.0)  # (f1, threshold, p, r)
    for thr in thresholds:
        tp = fp = tn = fn = 0
        for ex in EXAMPLES:
            guard = Guard.from_documents(
                documents=ex["corpus"],
                encoder=encoder,
                threshold=thr,
                verifier=verifier,
                entail_threshold=args.entail_threshold,
            )
            t, f_, n, m = _eval_at_threshold(guard, ex, thr)
            tp += t
            fp += f_
            tn += n
            fn += m
        p, r, f1 = _prf(tp, fp, fn)
        print(
            f"| {thr:.2f}      | {p:.3f}     | {r:.3f}  | {f1:.3f}  | "
            f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:2d} |"
        )
        if f1 > best[0]:
            best = (f1, thr, p, r)

    print()
    print(f"# Best threshold: {best[1]:.2f}  →  P={best[2]:.3f}  R={best[3]:.3f}  F1={best[0]:.3f}")
    print()
    # Per-example breakdown at best threshold
    print(f"# Per-example breakdown @ threshold={best[1]:.2f}")
    print()
    print("| example | claims | gold flagged | pred flagged | correct |")
    print("|---------|--------|--------------|--------------|---------|")
    for ex in EXAMPLES:
        guard = Guard.from_documents(
            documents=ex["corpus"],
            encoder=encoder,
            threshold=best[1],
            verifier=verifier,
            entail_threshold=args.entail_threshold,
        )
        report = guard.check(ex["answer"])
        gold_flagged = sum(1 for g in ex["gold"] if g == "HALLUCINATION")
        pred_flagged = report.n_flagged
        correct = sum(
            1
            for c, g in zip(report.claims, ex["gold"])
            if (c.status == ClaimStatus.HALLUCINATION_FLAG) == (g == "HALLUCINATION")
        )
        total = len(ex["gold"])
        print(
            f"| {ex['name']:18s} | {len(report.claims):2d} | "
            f"{gold_flagged} | {pred_flagged} | {correct}/{total} |"
        )


if __name__ == "__main__":
    main()
