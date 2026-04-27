# halluguard examples

Three runnable scripts. Each is self-contained — no API keys needed.

| File | What it shows | Extra deps |
|---|---|---|
| `quickstart.py` | Build a Guard, score grounded vs mixed answers | none |
| `streaming.py` | `check_stream` — flag a hallucinated sentence the moment it lands | none |
| `daemon_mode.py` | `Guard.from_daemon` — encoder lives in a long-lived `adaptmem serve` process | `adaptmem[server]`, `requests` |

## Prerequisites

```bash
pip install -e ".[dev]"
```

The first run downloads MiniLM (≈90MB) and the NLI cross-encoder (≈700MB)
on demand. CPU-only, no GPU required.

For `daemon_mode.py`, also start a daemon in another terminal:

```bash
pip install "adaptmem[server]"
adaptmem serve --port 7800 --base-model all-MiniLM-L6-v2
```

## Mental model

```
documents ─→ encoder.encode ─→ CorpusIndex
                                    │
                                    ▼
answer ─→ segment(claims) ─→ retrieve(top-k) ─→ NLI verify ─→ Claim status
```

Three knobs decide what counts as SUPPORTED:
- `threshold` — cosine similarity gate (default 0.55).
- `entail_threshold` — NLI entailment gate (default 0.5).
- `min_entail_votes` — how many of the top-K chunks must clear
  `entail_threshold` (default 1; raise to 2-3 for stricter agreement).

## Choosing a threshold

`SupportReport.trust_score` is the mean per-claim entailment in [0, 1]:

- `< 0.4` — most claims unsupported. Block by default.
- `0.4 – 0.7` — mixed; warn or surface the flagged subset.
- `≥ 0.7` — supported.

Calibrate on a held-out set of labelled examples — the right cutoff
depends on your tolerance for false flags vs missed hallucinations.
