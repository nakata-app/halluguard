"""Cross-repo end-to-end test: spawn `adaptmem serve` and exercise it
through `Guard.from_daemon`.

Skipped on macOS where the sentence-transformers + uvicorn cluster
deadlocks during model encode (documented in adaptmem PROGRESS.md).
Runs in CI on Linux.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import socket
import subprocess
import sys
import time

import pytest


# Skip predicates — keep the file importable everywhere, skip the body
# when the environment isn't suitable.
if importlib.util.find_spec("requests") is None:
    pytest.skip("requests not installed", allow_module_level=True)
if importlib.util.find_spec("adaptmem") is None or importlib.util.find_spec("fastapi") is None:
    pytest.skip("adaptmem[server] not installed", allow_module_level=True)
if platform.system() == "Darwin" and os.environ.get("HG_E2E_FORCE") != "1":
    pytest.skip(
        "Mac/Py3.14 deadlock in sentence-transformers + uvicorn thread pool. "
        "Set HG_E2E_FORCE=1 to override.",
        allow_module_level=True,
    )


def _free_port() -> int:
    """Find a random free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_healthz(url: str, timeout_s: float = 90.0) -> None:
    """Poll /healthz until the daemon is up or timeout (model load can be slow)."""
    import requests

    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/healthz", timeout=2)
            if r.ok:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(0.5)
    raise TimeoutError(f"daemon at {url} did not become healthy in {timeout_s}s: {last_err}")


@pytest.fixture(scope="module")
def daemon():
    """Spawn `adaptmem serve` on a random port, tear it down at module exit."""
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "adaptmem.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--device",
            "cpu",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_healthz(url)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_guard_from_daemon_reports_supported_for_grounded_claim(daemon):
    """End-to-end: real daemon, real encoder, halluguard verdict."""
    from halluguard import Guard
    from halluguard.verifier import NLIVerifier

    docs = [
        "PostgreSQL has native JSON and JSONB types since version 9.4.",
        "Redis stores JSON only via the RedisJSON module, not natively.",
    ]
    guard = Guard.from_daemon(
        documents=docs,
        daemon_url=daemon,
        verifier=NLIVerifier(),
    )
    report = guard.check(
        "PostgreSQL has had native JSON support since 9.4.",
        question="Which databases support JSON natively?",
    )
    assert len(report.claims) >= 1
    # Trust score should be high when the claim is supported by the corpus.
    assert report.trust_score >= 0.5


def test_guard_from_daemon_flags_hallucinated_claim(daemon):
    from halluguard import Guard
    from halluguard.verifier import NLIVerifier

    docs = [
        "PostgreSQL has native JSON and JSONB types since version 9.4.",
        "Redis stores JSON only via the RedisJSON module, not natively.",
    ]
    guard = Guard.from_daemon(
        documents=docs,
        daemon_url=daemon,
        verifier=NLIVerifier(),
    )
    # A claim that contradicts the corpus.
    report = guard.check(
        "Redis ships a native JSON column type out of the box.",
        question="Does Redis have native JSON?",
    )
    assert any(
        c.status.value == "HALLUCINATION_FLAG" for c in report.claims
    ), f"expected at least one flagged claim, got {[(c.text, c.status.value) for c in report.claims]}"
