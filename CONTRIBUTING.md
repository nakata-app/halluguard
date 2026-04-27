# Contributing to halluguard

Thanks for considering a contribution. halluguard is a small, focused
package — "no LLM in the loop" is the core design constraint. Keep
changes aligned with that and the review will be quick.

## Quickstart for a local dev loop

```bash
git clone https://github.com/nakata-app/halluguard.git
cd halluguard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install requests        # for daemon-mode tests
pre-commit install
```

## What we run before every commit

```bash
ruff check halluguard tests
mypy --strict halluguard
pytest -q
```

CI runs the same three on Python 3.10 / 3.11 / 3.12.

## What lands easily

- Bug fixes with a regression test (failing before / passing after).
- New benchmarks. We track honest numbers — null results are valuable.
  See `benchmarks/results_ragtruth_q_v1.json` for shape.
- New verifier backends as long as they keep the no-LLM-judge
  positioning (e.g. a different NLI cross-encoder, a rule-based gate).
- Daemon-mode improvements (`Guard.from_daemon`, `DaemonEncoder`).
- Streaming / async API additions that don't break the existing
  `Guard.check` contract.

## What needs a discussion first

- **LLM-as-judge backends.** The whole point is "no LLM in the
  inference loop." If you have a use case that demands one, open an
  issue first — we'll usually suggest you wire one yourself in a
  thin downstream wrapper rather than coupling halluguard to it.
- Custom fine-tuned NLI heads. A bigger NLI model would help
  precision but balloons install size and breaks the
  "swap any HuggingFace cross-encoder" contract. Defer until a
  benchmark gap demands it.
- Open-web fact-checking. halluguard's "ground truth" is the supplied
  corpus, by design.

## Style

- Match the existing code. Type hints on public surfaces; no
  speculative abstractions; comments only for non-obvious WHY.
- One commit per logical change.

## Reporting bugs

GitHub Issues. Include:
- Python version + OS.
- A minimum reproduction (claim + corpus + expected vs actual).
- Which threshold / vote / NLI model you ran with.

## Reporting security issues

See [`SECURITY.md`](SECURITY.md). Don't open a public issue for an
unpatched vulnerability.
