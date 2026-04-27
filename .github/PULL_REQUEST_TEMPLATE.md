## What this changes

<!-- one-paragraph summary; link to a tracking issue if there is one -->

## How it was tested

<!-- pytest output, a manual repro, or a benchmark JSON -->

## Checklist

- [ ] `ruff check halluguard tests` is clean
- [ ] `mypy --strict halluguard` is clean
- [ ] `pytest -q` passes locally
- [ ] CHANGELOG entry added (under `[Unreleased]`)
- [ ] If this changes the public API: README + ROADMAP updated
- [ ] If this adds a benchmark: `benchmarks/results_*.json` committed +
      README table updated (honest numbers, null results welcome)
- [ ] No LLM-as-judge coupling introduced (or documented why it's
      worth the dilution)
