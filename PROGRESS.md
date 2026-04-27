# halluguard — progress & resume notes

**Last updated:** 2026-04-27 (public-push session).

This file is the resume contract: open the repo, read this, you know the
state of play. Updated at the end of each working session.

**Public:** https://github.com/nakata-app/halluguard (master, MIT, CI green).

## Where we are

```
v0.1 skeleton           ████████████  done  (segment + retriever + report)
v0.2 NLI verifier       ████████████  done  (NLI + RAGTruth bench + honest numbers)
v0.2-ext                ████████████  done  (question-aware + vote + Claim metadata
                                             + CLI surface + smoke tests + CI + py.typed)
v0.2-prod               ████████████  done  (trust_score + JSON CLI + check_stream
                                             + from_adaptmem + release.yml + mypy --strict
                                             + timing bench + Guard.from_daemon + DaemonEncoder)
v0.3 ablation + real    ████░░░░░░░░  ~30%  (vote ablation done = honest null on HaluEval
                                             QA; RAGTruth mirror still dead, FActScore
                                             dataset state messy, sentence-level labels
                                             pending)
```

## Bench results — what's actually committed

| Pipeline | n cases | Threshold | Precision | Recall | F1 | JSON |
|---|---|---|---|---|---|---|
| Bi-encoder only | 200 | 0.70 | 0.451 | 0.790 | 0.575 | (older run, README only) |
| Bi-encoder + NLI | 200 | 0.70 | 0.490 | 0.960 | **0.649** | (older run, README only) |
| Bi-encoder + NLI + question-aware | 100 | 0.70 | 0.489 | 0.920 | **0.639** | `benchmarks/results_ragtruth_q_v1.json` (`2f42a71`) |

**Key observation:** question-aware F1 is within noise of plain NLI
(0.639 vs 0.649; ±1 case ≈ ±1pt on 100-case sample). **Question-aware
alone did not collapse the FP ceiling.** Recall held above 0.90;
precision didn't move.

The lever that *should* trade recall for precision is `min_entail_votes`
— that ablation is **not yet measured.**

## What's solid vs what's claimed but unmeasured

**Solid (committed JSONs / passing tests):**
- 34/34 tests pass (verifier, guard, CLI, retriever, segment).
- Question-aware NLI premise wired end-to-end (Verifier → Guard →
  CLI → bench).
- Vote policy code + 3 unit tests.
- Claim carries `entail_votes / entail_chunks` metadata, surfaced in
  markdown report.
- CLI: `--question`, `--nli`, `--entail-threshold`, `--min-votes`,
  `--corpus-text`, `--corpus-file`.
- CI matrix on GitHub Actions.
- py.typed shipped.

**Claimed in code but unmeasured:**
- `min_entail_votes=2` should lift precision. Hypothesised, not
  benchmarked. → v0.3 first item.
- `min_entail_votes=3` even more so.

## Open items

### v0.3 — vote ablation + real benchmarks
- [ ] **Vote ablation sweep:** rerun `ragtruth_eval.py` at the best
  threshold (0.70) with `min_entail_votes ∈ {1, 2, 3}` and write
  `results_ragtruth_vote_v1.json`. Expected: precision rises, recall
  drops; the F1 sweet spot shifts. This is the v0.3 hero number.
  - Bench script does not yet expose `--min-votes`. Add it first
    (small edit to `ragtruth_eval.py`).
- [ ] RAGTruth public mirror — periodically check HF Hub. As of
  2026-04, all known mirrors (`wandb/RAGTruth`, `flagrant/RAGTruth`,
  `ParticleMedia/RAGTruth`, `TIGER-Lab/RAGTruth`) are unreachable. If
  it lands, span-level evaluation becomes possible (HaluEval is
  item-level only).
- [ ] FActScore bench — different shape (atomic facts, retrieval over
  Wikipedia). Adds out-of-domain validation.
- [ ] Per-task-type breakdown — currently only QA. RAGTruth would
  unlock dialog / summarisation / data-to-text buckets.

### Production-ready follow-ups (analogous to adaptmem v0.4)
- [x] py.typed (`6c48739`)
- [x] CI (`4298373`)
- [x] CLI subprocess smoke tests (`49c266e`)
- [x] `--corpus-text` / `--corpus-file` CLI options (`b26f870`)
- [ ] PyPI release.

## How to resume (next session)

1. Read this file + `README.md` + `ROADMAP.md`.
2. Vote ablation done (committed result_ragtruth_vote_v1.json) — **honest
   null result** on HaluEval QA: vote=2/3 collapses into a flag-everything
   classifier because every HaluEval QA case has a 1-chunk corpus. The
   right test bed is RAGTruth multi-chunk, but all 4 mirrors are still
   dead as of 2026-04. Periodic `huggingface_hub` probe is the pragmatic
   answer.
3. FActScore dataset state on HF Hub is messy — original `shmsw25/FActScore`
   paper repo, only third-party copies on Hub (`dskar/FActScore`, etc).
   Treat like RAGTruth: blocked on upstream.
4. Open work that is NOT data-blocked:
   - Sentence-level span labels (RAGTruth shape) — still data-blocked.
   - Per-task-type breakdown — still data-blocked.
   - **PyPI release** (Atakan token).

## Toolchain

- Same shared venv as adaptmem + claimcheck:
  `~/Projects/metis-pair/benchmarks/.venv`.
- Tests: `cd ~/Projects/halluguard && ../metis-pair/benchmarks/.venv/bin/pytest -q`
- Current suite: **52/52 pass** (47 + 5 daemon), lint clean, mypy --strict clean.
- Timing bench: `python benchmarks/timing_bench.py --n 30 --out benchmarks/results_timing.json`
- Daemon mode (no per-process model load):
  ```bash
  pip install "adaptmem[server]"
  adaptmem serve --port 7800
  python -c "from halluguard import Guard; g = Guard.from_daemon(['doc'], 'http://127.0.0.1:7800')"
  ```

## Timeline

- **2026-04-26** — v0.2-ext shipped (question-aware NLI, vote policy,
  Claim metadata, full CLI surface, CI matrix, py.typed, 34 tests).
- **2026-04-27 morning** — v0.2-prod (trust_score + JSON CLI +
  check_stream + Guard.from_adaptmem + release.yml + mypy --strict +
  timing bench, 47 tests).
- **2026-04-27 noon** — v0.2-prod-daemon (Guard.from_daemon +
  DaemonEncoder, 52 tests).
- **2026-04-27 afternoon** — `nakata-app/halluguard` public on GitHub,
  `v0.3.1` shipped on PyPI (`pip install halluguard`), `[daemon]`
  optional dep gated `requests`.

## Open / next session

1. **PyPI token rotate** — first session token was pasted in chat, low
   risk but rotate per best practice. Need atakan to create a new
   token in PyPI; the secret update is a one-liner.
2. **RAGTruth public mirror** — periodic HF Hub probe; when one lands,
   span-level eval and vote-ablation v2 unlock.
3. **FActScore bench** — original `shmsw25/FActScore` repo only;
   third-party HF Hub mirrors are unverified. Treat like RAGTruth.
4. **Cross-repo integration test** — pytest fixture spawning
   `adaptmem serve` as a subprocess, hitting `Guard.from_daemon`
   end-to-end. Linux-only (Mac sentence-transformers + uvicorn
   deadlock).
