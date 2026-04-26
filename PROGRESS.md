# halluguard — progress & resume notes

**Last updated:** 2026-04-26 (post-Claude-session, before reboot).

This file is the resume contract: open the repo, read this, you know the
state of play. Updated at the end of each working session.

## Where we are

```
v0.1 skeleton           ████████████  done  (segment + retriever + report)
v0.2 NLI verifier       ████████████  done  (NLI + RAGTruth bench + honest numbers)
v0.2-ext (this session) ████████████  done  (question-aware + vote + Claim metadata
                                             + CLI surface + smoke tests + CI + py.typed)
v0.3 ablation + real    ███░░░░░░░░░  ~25%  (vote ablation, RAGTruth public mirror,
                                             FActScore, sentence-level labels)
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

1. Read this file + `README.md`.
2. First v0.3 task = vote ablation:
   - Add `--min-votes` flag to `benchmarks/ragtruth_eval.py` (forwards
     to `Guard(min_entail_votes=...)`). Single arg + propagation, ~10
     line change.
   - Run sweep:
     ```
     python benchmarks/ragtruth_eval.py --n-examples 50 --nli \
         --device cpu --threshold 0.70 --min-votes 1
     # repeat with --min-votes 2 and 3
     ```
   - Write `benchmarks/results_ragtruth_vote_v1.json` with the three
     rows.
   - Update README table with the precision lift (expected: P=0.49 →
     0.55+ at votes=2, F1 trade-off TBD).
3. After ablation, decide whether to:
   - Push for FActScore (broader validation), OR
   - Wait on RAGTruth public mirror, OR
   - Cut a v0.2 → v0.3 release (PyPI).

## Toolchain

- Same shared venv as adaptmem:
  `~/Projects/metis-pair/benchmarks/.venv`.
- Tests: `cd ~/Projects/halluguard && ../metis-pair/benchmarks/.venv/bin/pytest -q`
- Current suite: **34/34 pass**.

## Commit log highlights (this session)

```
2f42a71 README: add question-aware RAGTruth row + commit results JSON
b26f870 cli: --corpus-text and --corpus-file alternatives to --corpus dir
57c1889 bench: --device flag for ragtruth_eval (CPU forcing)
49c266e test: CLI subprocess smoke tests
4298373 ci: GitHub Actions matrix (py 3.10/3.11/3.12)
6c48739 package: ship py.typed marker (PEP 561)
66b77c0 report: surface entail_votes/entail_chunks on Claim + markdown
10acbf6 cli: surface --nli, --entail-threshold, --min-votes
06ff16f guard: vote-based policy on top of max-entailment gate
b6b51aa cli/bench: thread question through CLI + RAGTruth bench
5a76603 guard: forward question to verifier in Guard.check()
6ce41a6 test: fix tests/test_guard.py collection error
69cf032 verifier: question-aware NLI premise + multi-evidence vote count + tests
```

13 commits on top of `400d789` (the previous tip). All shipped to
`master` locally.
