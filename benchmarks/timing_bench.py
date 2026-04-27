"""Timing bench: how long does Guard.check take.

Standalone halluguard (no adaptmem). Same pattern as claimcheck's
timing_bench but with the no-frills `Guard.check` path — the numbers
a verification-only deployment would actually see.

Run from the repo root:

    pip install -e ".[dev]"
    python benchmarks/timing_bench.py --n 30 --out benchmarks/results_timing.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from halluguard import Guard
from halluguard.verifier import NLIVerifier


CORPUS = [
    "PostgreSQL added native JSON in 9.4 and JSONB shortly after.",
    "MySQL gained a JSON column type in version 5.7.7 (2015).",
    "SQLite has no native JSON type; the JSON1 extension queries TEXT.",
    "MongoDB stores documents as BSON, a binary JSON superset.",
    "Redis stores JSON via the RedisJSON module, not a native type.",
    "ChromaDB is an embedding-first vector database with a Python client.",
    "Pinecone is a managed vector database serving billion-scale indexes.",
    "Qdrant is an open-source vector database written in Rust.",
    "FAISS is a similarity search library from Meta AI for dense vectors.",
    "Annoy is Spotify's approximate nearest-neighbor library.",
]

WORKLOAD = [
    ("Which databases have native JSON?",
     "PostgreSQL has native JSON since 9.4. MySQL added JSON in 5.7.7."),
    ("Vector databases?",
     "Pinecone is a managed vector database. ChromaDB is embedding-first. "
     "Cassandra is a vector-native key-value store."),
    ("Where is JSON stored as TEXT?",
     "Postgres stores all JSON as TEXT. MySQL has no JSON support whatsoever."),
    ("Similarity search libraries?",
     "FAISS is Meta AI's library. Annoy is from Spotify."),
    ("Vector databases?",
     "Qdrant is written in Rust. Weaviate ships its own LLM internally."),
]


def run(n: int, warmup: int = 2) -> dict:
    print("building Guard (cpu, MiniLM + NLI cross-encoder)…", flush=True)
    t_build = time.perf_counter()
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    guard = Guard.from_documents(CORPUS, encoder=encoder, verifier=NLIVerifier())
    build_s = time.perf_counter() - t_build
    print(f"  built in {build_s:.1f}s", flush=True)

    for _ in range(warmup):
        guard.check(WORKLOAD[0][1], question=WORKLOAD[0][0])

    timings_ms: list[float] = []
    for i in range(n):
        question, answer = WORKLOAD[i % len(WORKLOAD)]
        t0 = time.perf_counter()
        guard.check(answer, question=question)
        ms = (time.perf_counter() - t0) * 1000.0
        timings_ms.append(ms)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}  last={ms:.0f}ms", flush=True)

    timings_ms.sort()
    return {
        "n_calls": n,
        "ms_per_call_p50": round(statistics.median(timings_ms), 2),
        "ms_per_call_p90": round(timings_ms[int(0.90 * len(timings_ms))], 2),
        "ms_per_call_p99": round(timings_ms[min(int(0.99 * len(timings_ms)), len(timings_ms) - 1)], 2),
        "ms_per_call_mean": round(statistics.fmean(timings_ms), 2),
        "ms_per_call_max": round(max(timings_ms), 2),
        "build_time_s": round(build_s, 2),
        "device": "cpu",
        "encoder": "all-MiniLM-L6-v2 (raw, no fine-tune)",
        "verifier": "cross-encoder/nli-deberta-v3-base",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="benchmarks/results_timing.json")
    args = ap.parse_args()

    result = run(args.n, args.warmup)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\n→ wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
