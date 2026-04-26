"""Real hallucination benchmark for halluguard — HaluEval QA.

We attempted RAGTruth public mirrors first (wandb/RAGTruth, flagrant/RAGTruth,
ParticleMedia/RAGTruth, TIGER-Lab/RAGTruth) — none are accessible without auth
or simply don't exist on HF Hub as of 2026-04. Falling back to
`pminervini/HaluEval` (`qa` config), which is the next-best public benchmark
with the right shape: knowledge (context), question, right_answer (gold-true),
hallucinated_answer (gold-false).

Each HaluEval item is expanded into two test cases:
  - (knowledge, right_answer)        → gold = NOT hallucinated
  - (knowledge, hallucinated_answer) → gold = HALLUCINATED

We run `Guard.check` on each; the prediction is "hallucinated" iff *any* claim
in the answer is flagged HALLUCINATION_FLAG. This is item-level evaluation,
matching HaluEval's design (HaluEval has no per-span labels — RAGTruth does,
but we couldn't reach it). Per-task-type reporting is not available for QA
since HaluEval QA is single-task; we leave the hook in for future RAGTruth
runs.

Run:
  python benchmarks/ragtruth_eval.py --n-examples 100
  python benchmarks/ragtruth_eval.py --n-examples 100 --nli
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow importing the package without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from halluguard import Guard, ClaimStatus


DATASET_NAME = "pminervini/HaluEval"
DATASET_CONFIG = "qa"
DATASET_SPLIT = "data"


def load_examples(n_examples: int) -> list[dict]:
    """Load HaluEval QA and expand each item into 2 cases (truthful + hallucinated).

    Returns a list of dicts: {context, answer, gold_hallucinated, task_type}.
    Total cases = 2 * n_examples (balanced).
    """
    from datasets import load_dataset

    print(
        f"loading {DATASET_NAME} [{DATASET_CONFIG}] split={DATASET_SPLIT} ...",
        file=sys.stderr,
        flush=True,
    )
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    cases: list[dict] = []
    for i in range(min(n_examples, len(ds))):
        row = ds[i]
        ctx = row["knowledge"]
        q = row["question"]
        # Frame answers as natural-language statements. HaluEval answers are
        # often short noun phrases ("Arthur's Magazine") so we prepend the
        # question for context to make the segmenter happy and to give the
        # bi-encoder a self-contained claim.
        truthful = f"{q} {row['right_answer']}"
        hallu = f"{q} {row['hallucinated_answer']}"
        cases.append(
            {
                "context": ctx,
                "question": q,
                "answer": truthful,
                "gold_hallucinated": False,
                "task_type": "qa",
                "id": f"{i}-true",
            }
        )
        cases.append(
            {
                "context": ctx,
                "question": q,
                "answer": hallu,
                "gold_hallucinated": True,
                "task_type": "qa",
                "id": f"{i}-hallu",
            }
        )
    return cases


def predict(guard: Guard, case: dict) -> bool:
    """Return True if the guard predicts the answer is hallucinated."""
    report = guard.check(case["answer"], question=case.get("question"))
    if not report.claims:
        # Empty segmentation — treat as hallucinated (defensive)
        return True
    return any(c.status == ClaimStatus.HALLUCINATION_FLAG for c in report.claims)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate(
    cases: list[dict],
    encoder,
    threshold: float,
    verifier,
    entail_threshold: float,
) -> tuple[int, int, int, int, dict[str, tuple[int, int, int, int]]]:
    """Run guard on all cases at the given threshold.

    Returns (tp, fp, tn, fn, per_task) where positive class = HALLUCINATED.
    per_task[task] = (tp, fp, tn, fn).
    """
    tp = fp = tn = fn = 0
    per_task: dict[str, list[int]] = {}
    for case in cases:
        guard = Guard.from_documents(
            documents=[case["context"]],
            encoder=encoder,
            threshold=threshold,
            verifier=verifier,
            entail_threshold=entail_threshold,
        )
        pred_hallu = predict(guard, case)
        gold_hallu = case["gold_hallucinated"]
        bucket = per_task.setdefault(case["task_type"], [0, 0, 0, 0])
        if pred_hallu and gold_hallu:
            tp += 1
            bucket[0] += 1
        elif pred_hallu and not gold_hallu:
            fp += 1
            bucket[1] += 1
        elif not pred_hallu and not gold_hallu:
            tn += 1
            bucket[2] += 1
        else:
            fn += 1
            bucket[3] += 1
    per_task_t = {k: tuple(v) for k, v in per_task.items()}
    return tp, fp, tn, fn, per_task_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-examples",
        type=int,
        default=100,
        help="Number of HaluEval items (each yields 2 cases — 1 truthful, 1 hallucinated)",
    )
    ap.add_argument("--nli", action="store_true", help="enable NLI verifier (cross-encoder)")
    ap.add_argument(
        "--nli-model",
        default="cross-encoder/nli-deberta-v3-base",
        help="NLI cross-encoder model",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, only evaluate at this single threshold (no sweep).",
    )
    ap.add_argument("--entail-threshold", type=float, default=0.5)
    ap.add_argument(
        "--encoder",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers encoder",
    )
    args = ap.parse_args()

    cases = load_examples(args.n_examples)
    n_total = len(cases)
    n_pos = sum(1 for c in cases if c["gold_hallucinated"])
    print(
        f"# loaded {n_total} cases ({n_pos} hallucinated, {n_total - n_pos} truthful) "
        f"from {DATASET_NAME}/{DATASET_CONFIG}",
        flush=True,
    )

    from sentence_transformers import SentenceTransformer

    print(f"loading encoder {args.encoder} ...", file=sys.stderr, flush=True)
    encoder = SentenceTransformer(args.encoder)

    verifier = None
    if args.nli:
        from halluguard.verifier import NLIVerifier

        print(f"loading NLI cross-encoder {args.nli_model} ...", file=sys.stderr, flush=True)
        verifier = NLIVerifier(model_name=args.nli_model)
        # Force model load now so timing in the sweep is fair
        verifier._ensure_model()

    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    print()
    print(
        f"# Halluguard real benchmark — {DATASET_NAME}/{DATASET_CONFIG} "
        f"({n_total} cases)"
    )
    print(
        f"# Encoder: {args.encoder}  |  NLI: {'on (' + args.nli_model + ')' if args.nli else 'off'}  "
        f"|  entail_threshold: {args.entail_threshold}"
    )
    print(f"# Positive class = HALLUCINATED")
    print()
    print("| threshold | precision | recall | F1     |  TP |  FP |  FN |  TN |")
    print("|-----------|-----------|--------|--------|-----|-----|-----|-----|")

    best = (0.0, 0.0, 0.0, 0.0, None)  # (f1, threshold, p, r, per_task)
    for thr in thresholds:
        t0 = time.time()
        tp, fp, tn, fn, per_task = evaluate(
            cases, encoder, thr, verifier, args.entail_threshold
        )
        p, r, f1 = prf(tp, fp, fn)
        dt = time.time() - t0
        print(
            f"| {thr:.2f}      | {p:.3f}     | {r:.3f}  | {f1:.3f}  | "
            f"{tp:3d} | {fp:3d} | {fn:3d} | {tn:3d} |   ({dt:.1f}s)"
        )
        if f1 > best[0]:
            best = (f1, thr, p, r, per_task)

    print()
    f1, thr, p, r, per_task = best
    print(f"# Best threshold: {thr:.2f}  ->  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    if per_task and len(per_task) > 1:
        print()
        print(f"# Per-task-type breakdown @ threshold={thr:.2f}")
        print()
        print("| task | precision | recall | F1     |  TP |  FP |  FN |  TN |")
        print("|------|-----------|--------|--------|-----|-----|-----|-----|")
        for task, (ttp, tfp, ttn, tfn) in sorted(per_task.items()):
            tp_p, tp_r, tp_f1 = prf(ttp, tfp, tfn)
            print(
                f"| {task:4s} | {tp_p:.3f}     | {tp_r:.3f}  | {tp_f1:.3f}  | "
                f"{ttp:3d} | {tfp:3d} | {tfn:3d} | {ttn:3d} |"
            )


if __name__ == "__main__":
    main()
