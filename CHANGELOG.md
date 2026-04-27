# Changelog

All notable changes to halluguard are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v0.2-prod

### Added
- **`Guard.from_daemon(documents, daemon_url)`** — build a Guard whose
  encoder is a long-lived `adaptmem serve` process. The retriever, NLI
  verifier, and segmenter all stay local; only the encoder hop crosses
  HTTP. Saves the per-process model load cost when claimcheck +
  halluguard + a third service would otherwise each load the same
  MiniLM. Calls `/healthz` first so misconfig fails loudly.
- **`halluguard.daemon.DaemonEncoder`** — drop-in `encoder` for
  `Guard.from_documents` / `CorpusIndex` that POSTs `/embed` batches
  to the daemon. Implements the SentenceTransformer encode subset that
  retriever needs, with local L2 re-normalisation as a defence.
- **`tests/test_daemon.py`** — 5 cases via stdlib HTTPServer fixture
  (no real daemon required). DaemonEncoder unit tests + Guard.from_daemon
  end-to-end.
- **`benchmarks/timing_bench.py`** — measures `Guard.check` p50/p90/p99
  ms across grounded / mixed / hallucinated workloads. Mac-deadlocks
  on first NLI predict; runs to completion on Linux/CI.
- **`SupportReport.trust_score`** — response-level scalar (mean per-claim
  support score) so middleware can route on a single number.
- **`SupportReport.to_dict()` + `--format json` CLI flag** — programmatic
  consumers consume the same structure markdown reports surface.
- **`Guard.check_stream(answer_chunks, question)`** — sentence-by-
  sentence verifier; flags a hallucinated sentence the moment it lands.
  Useful for live LLM responses where waiting for the full answer is
  too slow.
- **`Guard.from_adaptmem(am)` factory** — bridges adaptmem (domain-
  tuned retrieval) and halluguard. Tuned encoder + already-encoded
  corpus reused via `CorpusIndex.from_precomputed`.
- **mypy --strict pass.** `report.py` `to_dict() -> dict[str, Any]`,
  `verifier.py` `_ensure_model() -> None`, post-condition assert on the
  optional model field.
- **`release.yml`** — wheel build + sdist + tag-gated PyPI publish step
  (skipped via shell guard when `PYPI_API_TOKEN` is absent so initial
  tags don't fail).

### Changed
- README — new "Daemon mode (`Guard.from_daemon`)" section and link to
  the [adaptmem ADR](https://github.com/nakata-app/adaptmem/blob/master/docs/metis_integration.md)
  documenting the cross-repo integration.

## [Unreleased] — v0.2-ext

### Added
- **Question-aware NLI premise.** `NLIVerifier.verify()` accepts an
  optional `question` kwarg; when set, the NLI premise becomes
  `f"Question: {q}\nContext: {chunk}"`. RAG-derived answers often
  restate the question implicitly; without it, NLI can fail to align
  an otherwise correct answer-from-context.
- **Multi-evidence vote policy.** `Guard.min_entail_votes` (default 1,
  legacy behaviour preserved). A claim is now SUPPORTED iff
  `(max_entailment ≥ entail_threshold) AND (entail_votes ≥
  min_entail_votes)`. Raise to 2+ for a "agreement of N chunks"
  flag policy that trades recall for precision.
- **Claim metadata on the report.** `Claim.entail_votes` and
  `Claim.entail_chunks` populated when the verifier runs (None
  otherwise). New `vote_str` property. Added as a column in
  `to_markdown()`.
- **CLI surface for the new knobs.** `--question`, `--nli`,
  `--nli-model`, `--entail-threshold`, `--min-votes`. Plus
  `--corpus-text` and `--corpus-file` as alternatives to the
  `--corpus` directory glob (mutually-exclusive group).
- **`--device` flag on `benchmarks/ragtruth_eval.py`.** Forwards
  PyTorch device (`cpu` / `cuda` / `mps`) to both encoder and NLI.
  Apple-silicon MPS-deadlock workaround.
- **CLI subprocess smoke tests.** 7 tests covering --help flag listing,
  missing corpus source, mutex violation, missing corpus file,
  argparse type errors.
- **`py.typed`** (PEP 561) marker.
- **GitHub Actions CI matrix** on Python 3.10 / 3.11 / 3.12.
- **`benchmarks/results_ragtruth_q_v1.json`** — full 9-threshold sweep
  on HaluEval QA (50 items = 100 cases) with question-aware NLI and
  default `min_entail_votes=1`. Best at T=0.70: P=0.489, R=0.920,
  F1=0.639.

### Fixed
- `tests/test_guard.py` collection error. `from tests.test_retriever
  import FakeEncoder` failed with `ModuleNotFoundError: No module
  named 'tests'` because the project root was not on pytest's
  `sys.path`. Added `pythonpath = ["."]` to
  `[tool.pytest.ini_options]` and an empty `tests/__init__.py`.

### Changed
- `Guard.check(answer)` → `Guard.check(answer, question=None)` —
  forward-compatible (kwarg defaults preserve every existing call).
- README RAGTruth table: new row for question-aware NLI; updated
  reproduce snippet to use `--device cpu`. Honest takeaway revised:
  question-aware did not collapse the FP ceiling alone; vote-based
  policy ablation is the next lever.

### Open (carried into v0.3)
- **Vote ablation unmeasured.** Code shipped, tests pass, but no JSON
  for `min_entail_votes ∈ {2, 3}`. First v0.3 task.
- RAGTruth public mirror still unreachable (HaluEval QA fallback in
  use). When a mirror lands, span-level evaluation unlocks.
- FActScore bench — different shape, out-of-domain validation.

## [0.2.0] — pre-session baseline (commit `400d789`)

### Added
- Real HaluEval QA benchmark wired in (`benchmarks/ragtruth_eval.py`);
  README updated with honest numbers (Bi-encoder + NLI: F1=0.649 at
  T=0.70 on 200 cases).
- NLI cross-encoder verifier (`halluguard.verifier.NLIVerifier`,
  default `cross-encoder/nli-deberta-v3-base`).

### Earlier
- v0.1 — bi-encoder retrieval, claim segmentation, Guard high-level
  API, synthetic bench harness with diagnostic data.
