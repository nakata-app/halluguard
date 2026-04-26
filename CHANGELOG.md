# Changelog

All notable changes to halluguard are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
