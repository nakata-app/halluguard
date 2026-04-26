# halluguard roadmap

Status — `v0.2-ext` (April 26-27 2026): NLI verifier + question-aware premise + multi-evidence vote policy + Claim metadata + full CLI surface (`--question`, `--nli`, `--min-votes`, `--corpus-text`, `--corpus-file`) + `Guard.from_adaptmem` bridge + py.typed + GitHub Actions CI + 40 passing tests + four committed bench JSONs (synthetic, q-aware, FT-300 ablation, vote ablation). All `[Unreleased]` items in `CHANGELOG.md`.

The path below picks up from there. Each milestone has a concrete exit criterion.

---

## v0.3 — real benchmarks + vote ablation honesty (target: 1-2 weeks)

**Goal:** measure the multi-evidence vote policy on a corpus shape that actually exercises it, plus a second public benchmark to validate the approach beyond HaluEval QA.

- [ ] **RAGTruth public mirror.** Four mirrors (`wandb/RAGTruth`, `flagrant/RAGTruth`, `ParticleMedia/RAGTruth`, `TIGER-Lab/RAGTruth`) all unreachable as of 2026-04. Re-check periodically; when one lands, span-level evaluation unlocks (HaluEval is item-level only).
- [ ] **Vote ablation re-run on multi-chunk corpora.** The HaluEval QA run (`results_ragtruth_vote_v1.json`) collapsed into a flag-everything classifier because every case has a 1-chunk corpus. RAGTruth's per-case multi-chunk shape is the right test bed for `min_entail_votes ∈ {1, 2, 3}` to actually trade recall for precision.
- [ ] **FActScore bench.** Different shape (atomic facts, retrieval over Wikipedia). Adds out-of-domain validation. Public via `wcclark/FActScore` HF Hub.
- [ ] **Sentence-level span labels.** RAGTruth provides per-span gold; current item-level eval (`{predicted_hallucinated: bool}`) hides where exactly the Guard flagged. Span-level lets us measure precision/recall at the claim granularity the API already exposes.
- [ ] **Per-task-type breakdown.** RAGTruth distinguishes QA / dialog / summarisation / data-to-text. Mirror the per-type table that `longmemeval_eval.py` already produces (commit `2ce0132` in adaptmem).

**Exit:** three benchmark JSONs in `benchmarks/` (HaluEval QA + RAGTruth + FActScore), README table updated with the multi-bench numbers, vote ablation re-run shows a non-degenerate trade-off curve.

---

## v0.4 — production-ready surface (target: 1 week)

**Goal:** make halluguard something a stranger can drop into a CI step or a request middleware without reading source code.

- [x] CI matrix on GitHub Actions (Python 3.10/3.11/3.12) — `4298373`.
- [x] py.typed (PEP 561) for downstream type-checkers — `6c48739`.
- [x] CLI subprocess smoke tests — `49c266e`.
- [x] `--corpus-text` / `--corpus-file` alternatives to `--corpus` directory — `b26f870`.
- [x] `--device` flag for Apple-silicon MPS-deadlock workaround — `57c1889`.
- [ ] **Trust score.** Response-level scalar `report.trust_score` (mean support_score across claims) so middleware can route on a single number.
- [ ] **`SupportReport.to_dict()`** + `--format json` CLI flag for programmatic consumers. Markdown / plain output already exist.
- [ ] **Streaming `Guard.check_stream(answer_chunks)`.** Token-by-token feed; flag claims as soon as a sentence boundary is hit. Useful for live LLM contexts where waiting for the whole response is too slow.
- [ ] **PyPI release.** Wheel build + `pypi-publish` job on tag. Needs maintainer-controlled API token.
- [ ] **Strict-mode mypy clean.** `py.typed` was the marker; the actual `mypy --strict halluguard` pass is the next quality gate.

**Exit:** `pip install halluguard`, the JSON output integrates into a CI pipeline, the streaming API ships in a tagged release.

---

## v0.5 — outreach + ecosystem fit (target: 2 weeks)

**Goal:** position halluguard as a peer alternative to the LLM-as-judge category (Patronus AI, Galileo, CleanLab) — different design tradeoff, different cost profile.

- [ ] **Comparison table** in README: halluguard vs LLM-as-judge across F1, latency-per-claim, cost-per-claim, vendor lock-in.
- [ ] **Position paper / blog post.** "Reverse RAG: hallucination detection without LLM judgement." Public artefact for outreach.
- [ ] **`Guard.from_adaptmem` validation on a multi-chunk bench.** The HaluEval QA run was a null result because of corpus shape; RAGTruth multi-chunk should reveal whether domain-tuned encoders lift the cosine gate (the original Fikir-1 hypothesis).
- [ ] **Integration recipe** with one popular RAG framework (LangChain or LlamaIndex). Drop-in middleware example.
- [ ] **Open a GitHub Discussion** on a relevant LLM-safety repo (e.g. `guardrails-ai/guardrails`) framing halluguard as the "no-LLM-judge" branch of the design space.

**Exit:** README has a side-by-side comparison with the closed-source category, the streaming integration is documented end-to-end with at least one external reference.

---

## Non-goals (until further notice)

- **LLM-as-judge fallback.** The whole point of halluguard is "no LLM in the loop." Adding an optional LLM judge would dilute the positioning; keep it as a separate tool if it ever becomes useful.
- **Custom fine-tuned NLI model.** A bigger NLI head would help precision but balloons install size and breaks the "swap any HF cross-encoder" contract. Defer until a benchmark gap demands it.
- **Open-web fact-checking.** Halluguard's "ground truth" is the supplied corpus, by design. Web-scale verification is a different product.

---

## What needs Atakan's hand

- **PyPI release** requires a maintainer-controlled API token.
- **GitHub Discussion** in upstream LLM-safety repos is author-attribution-sensitive.
- **RAGTruth public mirror availability** is upstream-dependent (HF Hub or paper authors).

Everything else above can be done locally in normal sessions.
