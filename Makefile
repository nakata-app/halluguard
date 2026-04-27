PYTHON ?= python

.PHONY: install dev test lint typecheck verify clean help

help:
	@echo "halluguard — common dev targets:"
	@echo "  make install    — pip install -e .[dev,daemon]"
	@echo "  make dev        — install + pre-commit hooks"
	@echo "  make test       — pytest -q"
	@echo "  make lint       — ruff check"
	@echo "  make typecheck  — mypy --strict"
	@echo "  make verify     — lint + typecheck + test (CI-equivalent)"
	@echo "  make clean      — strip caches and build artefacts"

install:
	$(PYTHON) -m pip install -e ".[dev,daemon]"

dev: install
	$(PYTHON) -m pip install pre-commit
	pre-commit install

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check halluguard tests

typecheck:
	$(PYTHON) -m mypy --strict halluguard

verify: lint typecheck test
	@echo "all gates green"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
