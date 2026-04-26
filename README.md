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

## Initial benchmark

Synthetic suite — 6 examples, 18 claims, 7 hallucinations covering polarity flips ("after render" → "before render"), false attribution (LSM-trees in Postgres), and unsupported additions (pet llamas on Mars).

| Pipeline | Precision | Recall | F1 |
|---|---|---|---|
| Bi-encoder only (cosine threshold) | 0.80 | 0.57 | 0.667 |
| **Bi-encoder + NLI verifier** | **0.78** | **1.00** | **0.875** |

The NLI stage catches polarity flips and unsupported additions that share lexical context with the corpus but reverse or extend its meaning. Bi-encoder cosine treats those as near-identical (high word overlap). NLI sees the logical relationship.

Default NLI model: `cross-encoder/nli-deberta-v3-base` (~440MB, lazy-loaded). Replaceable with any HuggingFace cross-encoder NLI checkpoint.

Larger benchmarks (RAGTruth, FActScore, HaluEval) coming next.

## Status

`v0.2` — bi-encoder + NLI verifier wired up, synthetic bench passing. RAGTruth integration and CI workflows next.

## License

MIT.
