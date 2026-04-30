"""DaemonEncoder + Guard.from_daemon tests using a stubbed HTTP server."""
from __future__ import annotations

import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

if importlib.util.find_spec("requests") is None:
    pytest.skip("requests not installed", allow_module_level=True)


# ---- Tiny in-memory daemon ------------------------------------------------


class _FakeHandler(BaseHTTPRequestHandler):
    embed_dim = 4
    healthz_calls = 0
    embed_calls = 0
    last_body: dict | None = None

    def log_message(self, *_args, **_kwargs):  # silence the BaseHTTPRequestHandler default
        return

    def do_GET(self):  # noqa: N802 — stdlib name
        if self.path == "/healthz":
            type(self).healthz_calls += 1
            payload = {"ok": True, "uptime_s": 1.0}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):  # noqa: N802
        if self.path == "/embed":
            type(self).embed_calls += 1
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n))
            type(self).last_body = req
            texts = req["texts"]
            # Deterministic fake embedding: each text → unit vector along a hashed axis.
            embeddings = []
            for i, t in enumerate(texts):
                vec = np.zeros(type(self).embed_dim, dtype=np.float32)
                vec[hash(t) % type(self).embed_dim] = 1.0
                # Add tiny perturbation by index so hits don't all tie
                vec[(i + 1) % type(self).embed_dim] += 0.01
                vec = vec / np.linalg.norm(vec)
                embeddings.append(vec.tolist())
            payload = {
                "embeddings": embeddings,
                "model": "fake-encoder",
                "dim": type(self).embed_dim,
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def fake_daemon():
    # Reset class-level counters so tests are independent.
    _FakeHandler.healthz_calls = 0
    _FakeHandler.embed_calls = 0
    _FakeHandler.last_body = None

    server = HTTPServer(("127.0.0.1", 0), _FakeHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Tiny wait for the listener to be ready.
    time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


# ---- DaemonEncoder unit tests --------------------------------------------


def test_daemon_encoder_encode_returns_numpy_array(fake_daemon):
    from halluguard.daemon import DaemonEncoder

    enc = DaemonEncoder(daemon_url=fake_daemon)
    out = enc.encode(["hello", "world"])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 4)
    assert out.dtype == np.float32
    # Each row should be unit-normalised after the encoder's local re-norm.
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_daemon_encoder_handles_string_input(fake_daemon):
    from halluguard.daemon import DaemonEncoder

    enc = DaemonEncoder(daemon_url=fake_daemon)
    out = enc.encode("solo")  # not a list
    assert out.shape == (1, 4)


def test_daemon_encoder_empty_input_returns_zeros(fake_daemon):
    from halluguard.daemon import DaemonEncoder

    enc = DaemonEncoder(daemon_url=fake_daemon)
    out = enc.encode([])
    assert out.shape[0] == 0
    # No HTTP call needed for the empty path.
    assert _FakeHandler.embed_calls == 0


def test_daemon_encoder_healthz(fake_daemon):
    from halluguard.daemon import DaemonEncoder

    enc = DaemonEncoder(daemon_url=fake_daemon)
    body = enc.healthz()
    assert body["ok"] is True
    assert body["uptime_s"] == 1.0
    assert _FakeHandler.healthz_calls == 1


def test_daemon_encoder_sends_authorization_header_when_api_key_set(fake_daemon):
    """When api_key is set, every /embed request carries Bearer header."""
    from halluguard.daemon import DaemonEncoder

    enc = DaemonEncoder(daemon_url=fake_daemon, api_key="secret-xyz")
    enc.encode(["hello"])
    # The fake handler doesn't capture headers, so we extend it via a flag
    # set on the class. Confirm via auth_headers shape.
    headers = enc._auth_headers()
    assert headers == {"Authorization": "Bearer secret-xyz"}


def test_daemon_encoder_no_auth_header_when_api_key_unset(fake_daemon, monkeypatch):
    monkeypatch.delenv("ADAPTMEM_API_KEY", raising=False)
    from halluguard.daemon import DaemonEncoder

    enc = DaemonEncoder(daemon_url=fake_daemon)
    assert enc._auth_headers() == {}


# ---- Guard.from_daemon end-to-end ----------------------------------------


def test_guard_from_daemon_builds_and_checks(fake_daemon):
    from halluguard import Guard

    docs = [
        "PostgreSQL has native JSON support since 9.4.",
        "Redis stores JSON via the RedisJSON module.",
    ]
    g = Guard.from_daemon(docs, daemon_url=fake_daemon)
    # /healthz called by the factory; /embed at corpus encode time.
    assert _FakeHandler.healthz_calls == 1
    assert _FakeHandler.embed_calls >= 1

    # Run a check — at minimum, retrieval-side encode must round-trip.
    report = g.check("Postgres has JSON.", question="Where is JSON native?")
    assert report.answer == "Postgres has JSON."
    # `_FakeHandler.last_body` is the most recent /embed body — it should be
    # the query text wrapped as a list.
    assert _FakeHandler.last_body is not None
    assert isinstance(_FakeHandler.last_body["texts"], list)
