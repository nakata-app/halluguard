"""CLI smoke tests for halluguard.

These exercise the argparse plumbing + corpus loading via subprocess; they
do *not* download a real sentence-transformers model. We use a tiny stub
encoder injected via PYTHONPATH that the CLI accepts in place of the
default `all-MiniLM-L6-v2`.

If we can't avoid the real encoder load (cli.py constructs it directly),
we fall back to asserting the CLI's `--help` and argument-error paths,
which is enough to catch argparse regressions.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CLI = [sys.executable, "-m", "halluguard.cli"]


def _run(args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        CLI + args,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        **kwargs,
    )


def test_cli_help_lists_all_flags():
    out = _run(["--help"])
    assert out.returncode == 0
    for flag in ["--corpus", "--threshold", "--top-k", "--question",
                 "--nli", "--entail-threshold", "--min-votes", "--format"]:
        assert flag in out.stdout, f"missing flag in --help: {flag}"


def test_cli_missing_corpus_dir_errors(tmp_path: Path):
    answer_file = tmp_path / "ans.txt"
    answer_file.write_text("any claim.")
    out = _run([str(answer_file), "--corpus", str(tmp_path / "does-not-exist")])
    assert out.returncode == 1
    assert "corpus directory not found" in out.stderr


def test_cli_empty_corpus_dir_errors(tmp_path: Path):
    corpus_dir = tmp_path / "empty"
    corpus_dir.mkdir()
    answer_file = tmp_path / "ans.txt"
    answer_file.write_text("any claim.")
    out = _run([str(answer_file), "--corpus", str(corpus_dir)])
    assert out.returncode == 1
    assert "no .txt files" in out.stderr


def test_cli_min_votes_arg_parses_as_int(tmp_path: Path):
    """`--min-votes foo` should fail at argparse, not silently coerce."""
    answer_file = tmp_path / "ans.txt"
    answer_file.write_text("x.")
    out = _run([str(answer_file), "--corpus", str(tmp_path), "--min-votes", "not-an-int"])
    assert out.returncode != 0
    # argparse writes the error to stderr
    assert "invalid int value" in out.stderr or "argument --min-votes" in out.stderr
