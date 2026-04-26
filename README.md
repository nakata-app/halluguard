# halluguard

**Reverse RAG hallucination detector.** No LLM judge. Vendor-neutral. Real-time.

## The problem

Every LLM hallucinates. The standard mitigation is RAG: retrieve relevant documents, feed them as context, ask the LLM to answer "based on context." This **reduces** hallucination but does not **detect** when one slips through. After generation, you don't know which sentences are supported by the context and which the model invented.

The current detection options all have downsides:

- **LLM-as-judge** — recursive, slow, expensive, and the judge can hallucinate too.
- **Constrained generation** — fragile, breaks on any prompt change.
- **Post-hoc fact-checking** — manual, batch, doesn't scale.
- **Vendor-locked tools** — tied to a specific provider's safety stack.

## The idea

**Reverse RAG.** Standard RAG goes `query → retrieve → answer`. Halluguard goes `answer → retrieve → flag`.

After the LLM produces an answer:

1. Split the answer into atomic claims (sentence-level by default, semantic-unit option later).
2. For each claim, retrieve the top-k chunks from the source corpus that **could** support it (same encoder you use for normal RAG).
3. Score the alignment between the claim and each retrieved chunk (cosine + lightweight cross-encoder verification).
4. Below threshold → `HALLUCINATION_FLAG`. Above threshold → `SUPPORTED` with citation IDs.

No LLM is invoked at any stage. The retriever is the same model your RAG pipeline already runs.

## Why this is new

Most "hallucination detection" libraries either:

- Run an LLM-as-judge (expensive recursion, judge itself hallucinates), or
- Compute embedding similarity of the **whole answer** to the **whole context** (loses sentence-level granularity), or
- Require fine-tuning a custom classifier per domain.

Halluguard works at sentence granularity, uses the retrieval stack you already have, and adds zero LLM cost per check.

## Architecture (planned)

```
halluguard/
  __init__.py
  segment.py        # split answer into atomic claims
  retriever.py      # bi-encoder retrieval over corpus chunks
  verifier.py       # claim ↔ chunk alignment scoring
  guard.py          # high-level Guard.check(answer, context) -> Report
  report.py         # SupportReport dataclass + to_html / to_markdown
  cli.py            # halluguard check answer.txt --corpus docs/
benchmarks/
  truthfulqa_eval.py     # measure flag precision/recall on known datasets
  ragtruth_eval.py
tests/
  test_segment.py
  test_retriever.py
  test_verifier.py
  test_end_to_end.py
README.md
LICENSE
pyproject.toml
.github/workflows/
```

## High-level API (target)

```python
from halluguard import Guard

guard = Guard.from_corpus(documents=["..."])  # build index once
report = guard.check(answer="The user prefers PostgreSQL because it has better JSON support.")

for claim in report.claims:
    print(claim.text, claim.status, claim.support_score)
# "The user prefers PostgreSQL"   SUPPORTED  0.91 (chunk #m12)
# "...because it has better JSON support"   HALLUCINATION_FLAG  0.34
```

## What halluguard is NOT

- Not an LLM proxy. It does not generate or rewrite text.
- Not a fact-checker against the open web. The "ground truth" is **your corpus**.
- Not a guarantee. Some hallucinations align lexically with unrelated chunks; halluguard reduces but does not eliminate false negatives. The threshold is tunable per use case.

## Benchmarks

### Synthetic suite (`benchmarks/synthetic_eval.py`)

6 examples, 18 claims, 7 hallucinations covering polarity flips ("after render" → "before render"), false attribution (LSM-trees in Postgres), and unsupported additions (pet llamas on Mars).

| Pipeline | Precision | Recall | F1 |
|---|---|---|---|
| Bi-encoder only (cosine threshold) | 0.80 | 0.57 | 0.667 |
| **Bi-encoder + NLI verifier** | **0.78** | **1.00** | **0.875** |

Synthetic data is curated to be diagnostic — F1=0.875 is a "the pipeline isn't broken" signal, not a generalisation claim.

### Real benchmark — HaluEval QA (`benchmarks/ragtruth_eval.py`)

We tried the public RAGTruth mirrors first (`wandb/RAGTruth`, `flagrant/RAGTruth`, `ParticleMedia/RAGTruth`, `TIGER-Lab/RAGTruth`) — none are accessible on HF Hub. Fell back to **`pminervini/HaluEval`** (`qa` config, 10 000 items). Each item is expanded into two cases: `(knowledge, right_answer)` → gold-truthful, `(knowledge, hallucinated_answer)` → gold-hallucinated. Prediction is "hallucinated" iff any claim in the answer is flagged. Item-level evaluation (HaluEval QA has no per-span labels — RAGTruth would, but it's locked).

| Pipeline | n cases | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Bi-encoder only | 200 | 0.70 | 0.451 | 0.790 | 0.575 |
| Bi-encoder + NLI | 200 | 0.70 | 0.490 | 0.960 | **0.649** |
| Bi-encoder + NLI + question-aware premise | 100 | 0.70 | 0.489 | 0.920 | 0.639 |

Latest run is `benchmarks/results_ragtruth_q_v1.json` — same Guard, but the NLI premise is now `f"Question: {q}\nContext: {chunk}"` instead of `chunk` alone. F1 is within noise of the older 200-case run (0.639 vs 0.649; ±1 case ≈ ±1pt on a 100-case sample). Recall held above 0.90; precision did not move much. Question-aware framing **did not collapse the false-positive ceiling on its own** — the bi-encoder still surfaces enough lexically-similar but semantically-irrelevant chunks that NLI rules them in.

The honest takeaway: **halluguard catches almost every hallucination on a real benchmark, but at a high false-positive rate**. For real-world deployment you want it as a *flag-for-review* layer, not an auto-reject gate. The new `min_entail_votes` knob (require N of top-K to entail, not just one) is the lever for trading recall down for precision — that ablation isn't measured here yet.

Default NLI model: `cross-encoder/nli-deberta-v3-base` (~440MB, lazy-loaded). Replaceable with any HuggingFace cross-encoder NLI checkpoint.

Run yourself:
```bash
pip install -e ".[bench]"
# Apple silicon: pass --device cpu to bypass the MPS deadlock that
# silently exits long NLI sweeps.
python benchmarks/ragtruth_eval.py --n-examples 50 --nli --device cpu
```

Larger benchmarks (full RAGTruth once a public mirror appears, FActScore) coming next.

## Status

`v0.2` — bi-encoder + NLI verifier wired up, synthetic bench passing, real HaluEval QA bench wired in with honest numbers. RAGTruth (when reachable) and CI workflows next.

## License

MIT.
