"""Daemon mode: encoder lives in a long-lived `adaptmem serve` process.

Useful when claimcheck + halluguard + a third service would otherwise
each load their own MiniLM. With the daemon they share one model in
memory.

Prerequisites:

    pip install "adaptmem[server]" requests
    adaptmem serve --port 7800 --base-model all-MiniLM-L6-v2

Then in another terminal:

    pip install -e ".[dev]"
    python examples/daemon_mode.py
"""
from halluguard import Guard
from halluguard.verifier import NLIVerifier


def main() -> None:
    documents = [
        "PostgreSQL ships native JSON since version 9.4.",
        "Redis stores JSON via the RedisJSON module, not a native type.",
        "MongoDB stores documents in BSON, a binary JSON superset.",
    ]

    # `from_daemon` calls /healthz first — fail loudly here if the URL
    # is wrong, not deep inside the first .check().
    guard = Guard.from_daemon(
        documents=documents,
        daemon_url="http://127.0.0.1:7800",
        verifier=NLIVerifier(),  # NLI verifier still runs in-process
    )

    report = guard.check(
        "PostgreSQL has native JSON since 9.4. Redis stores JSON natively too.",  # second sentence wrong
        question="Which databases have native JSON?",
    )
    print(f"trust_score={report.trust_score:.3f}")
    for c in report.claims:
        marker = "ok  " if c.status.value == "SUPPORTED" else "FLAG"
        print(f"  {marker}  score={c.support_score:.2f}  {c.text!r}")


if __name__ == "__main__":
    main()
