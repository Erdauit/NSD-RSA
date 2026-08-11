.PHONY: help setup check clean-cache lint test s0-estimate s1-download s1-sanity s2-stimuli s2-activations s2-check s3-rsa s4-analysis s5-encoding figures

PY := uv run python

help:
	@echo "NSD-RSA — stage runner"
	@echo ""
	@echo "  make setup          Install uv (if missing), create py3.11 venv, install deps"
	@echo "  make check          Verify environment: python, torch, key packages, AWS creds"
	@echo "  make test           Run unit tests"
	@echo "  make lint           Ruff lint + format check"
	@echo ""
	@echo "  make s0-estimate    S0: estimate NSD download size BEFORE downloading"
	@echo "  make s1-download    S1: fetch shared1000 betas from NSD (resumable)"
	@echo "  make s1-sanity      S1: sanity checks + figures"
	@echo "  make s2-stimuli     S2: fetch the 1000 shared images (542 MB)"
	@echo "  make s2-activations S2: extract + cache model activations"
	@echo "  make s2-check       S2: sanity-check activations + layer-similarity figure"
	@echo "  make s3-rsa         S3: build RDMs, RSA scores, noise-ceiling normalisation"
	@echo "  make s4-analysis    S4: main analysis figures"
	@echo "  make s5-encoding    S5: ridge encoding model"
	@echo ""
	@echo "  make clean-cache    Delete cache/ (activations, RDMs). Keeps data/."

setup:
	@command -v uv >/dev/null 2>&1 || { \
		echo ">>> uv not found — installing to ~/.local/bin"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo ">>> Creating Python 3.11 environment"
	uv python install 3.11
	uv venv --python 3.11
	@echo ">>> Installing dependencies (this pulls torch, ~2-3 GB, takes a few minutes)"
	uv sync --extra dev
	@echo ""
	@echo "=== setup complete ==="
	@$(MAKE) --no-print-directory check

check:
	$(PY) scripts/check_env.py

lint:
	uv run ruff check src scripts tests
	uv run ruff format --check src scripts tests

test:
	uv run pytest

clean-cache:
	rm -rf cache/*
	@echo "cache/ cleared (data/ untouched)"

s0-estimate:
	$(PY) scripts/s0_estimate_download.py --config configs/data.yaml

s1-download:
	$(PY) scripts/s1_download_betas.py --config configs/data.yaml

s1-sanity:
	$(PY) scripts/s1_sanity_checks.py --config configs/data.yaml

s2-stimuli:
	$(PY) scripts/s2_fetch_stimuli.py --config configs/data.yaml

s2-activations:
	$(PY) scripts/s2_extract_activations.py --config configs/models.yaml

s2-check:
	$(PY) scripts/s2_check_activations.py --config configs/models.yaml

s3-rsa:
	$(PY) scripts/s3_rsa.py --config configs/rsa.yaml

s4-analysis:
	$(PY) scripts/s4_analysis.py --config configs/rsa.yaml

s5-encoding:
	$(PY) scripts/s5_encoding.py --config configs/encoding.yaml
