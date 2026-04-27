"""Daemon-backed encoder shim.

Lets `Guard` (and any other consumer with the same `encode(texts, ...)`
contract) talk to a long-lived `adaptmem serve` process instead of
loading its own `SentenceTransformer`. Useful for:
- agent loops (metis) that want one model in memory across many calls;
- production middleware where two services would otherwise each load a
  copy of the same encoder.

The retriever-side abstraction is still local — only the encoder hop
goes through HTTP. Cosine search, NLI verification, segmentation all
stay in-process.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np


class DaemonEncoder:
    """Drop-in `encoder` for `Guard.from_documents(...)` / `CorpusIndex`.

    Implements the subset of the SentenceTransformer encode contract
    that halluguard's retriever uses:
        encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False, batch_size=64) -> np.ndarray

    The daemon already L2-normalises, so `normalize_embeddings` is
    advisory (we re-normalise locally just in case).
    """

    def __init__(self, daemon_url: str = "http://127.0.0.1:7800", timeout_s: float = 10.0) -> None:
        if importlib.util.find_spec("requests") is None:
            raise SystemExit(
                "DaemonEncoder requires `requests`. Install with `pip install requests`."
            )
        self.daemon_url = daemon_url.rstrip("/")
        self.timeout_s = timeout_s
        self._dim: int | None = None

    def encode(
        self,
        texts: list[str] | str,
        *,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 64,
        **_: Any,
    ) -> np.ndarray[Any, Any]:
        """POST /embed and return an `(n, dim)` numpy array."""
        import requests  # type: ignore[import-untyped]

        # Match SentenceTransformer's input flexibility.
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)
        if not texts_list:
            dim = self._dim or 1
            return np.zeros((0, dim), dtype=np.float32)

        try:
            resp = requests.post(
                f"{self.daemon_url}/embed",
                json={"texts": texts_list},
                timeout=self.timeout_s,
            )
        except requests.exceptions.RequestException as e:  # pragma: no cover
            raise RuntimeError(
                f"adaptmem daemon at {self.daemon_url} unreachable: {e}. "
                f"Start it with `adaptmem serve` (see adaptmem docs/metis_integration.md)."
            ) from e
        if not resp.ok:
            raise RuntimeError(
                f"adaptmem daemon /embed returned {resp.status_code}: {resp.text[:200]}"
            )

        body = resp.json()
        embeddings: np.ndarray[Any, Any] = np.asarray(body["embeddings"], dtype=np.float32)
        self._dim = int(body["dim"])

        if normalize_embeddings and embeddings.size:
            # Daemon already normalises, but defend against a future change.
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            embeddings = (embeddings / norms).astype(np.float32)

        return embeddings

    def healthz(self) -> dict[str, Any]:
        """Best-effort check before passing to a Guard."""
        import requests

        resp = requests.get(f"{self.daemon_url}/healthz", timeout=self.timeout_s)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result
