# Makefile for the Stillwar reproducibility pipeline.
#
# `make all` regenerates every table and figure from the raw snapshot. Expensive
# stages are content-addressed (see src/stillwar/cache.py), so a second `make all`
# on an unchanged tree only re-renders tables and figures. Use `make all FORCE=1`
# to recompute everything.
.PHONY: all help setup verify ingest features train baselines simulate experiments figures tables trials test lint clean

PYTHON ?= uv run python
PYTEST ?= uv run pytest
RUFF   ?= uv run ruff
FORCE  ?= 0
FORCEARG = $(if $(filter 1,$(FORCE)),--force,)

help:
	@echo "Stillwar reproducibility pipeline:"
	@echo "  make all         - verify, features, train, baselines, simulate, experiments, figures, tables, test"
	@echo "  make setup       - install the package and dev dependencies"
	@echo "  make verify      - verify the SHA-256 data manifest"
	@echo "  make ingest      - download any missing raw CSVs from Yahoo Finance"
	@echo "  make features    - build the order-2 log-signature tensor"
	@echo "  make train       - train LogSig, LogSig-Inc, MLP and LSTM across all seeds"
	@echo "  make baselines   - build the Hist-MV and GMV benchmark forecasts"
	@echo "  make simulate    - produce Tables 1-3 and the Phase-4 diagnostics"
	@echo "  make experiments - run the structural extensions of Section 3.7"
	@echo "  make trials      - evaluate the 1,225 Deflated-Sharpe trials (about 20 min, once)"
	@echo "  make figures     - render the manuscript figures"
	@echo "  make tables      - regenerate the manuscript tables and sync docs/JOURNAL.md"
	@echo "  make test        - run the test suite"
	@echo "  make lint        - run ruff over src and tests"
	@echo "  make clean       - remove caches and generated artifacts"
	@echo ""
	@echo "  FORCE=1 recomputes every stage regardless of the cache."

all: verify features train baselines simulate experiments figures tables test

setup:
	uv pip install -e ".[dev]"

verify:
	$(PYTHON) -m stillwar verify $(FORCEARG)

ingest:
	$(PYTHON) -m stillwar ingest $(FORCEARG)

features:
	$(PYTHON) -m stillwar features $(FORCEARG)

train:
	$(PYTHON) -m stillwar train $(FORCEARG)

baselines:
	$(PYTHON) -m stillwar baselines $(FORCEARG)

simulate:
	$(PYTHON) -m stillwar simulate $(FORCEARG)

experiments:
	$(PYTHON) -m stillwar experiments $(FORCEARG)

trials:
	$(PYTHON) -m stillwar trials $(FORCEARG)

figures:
	$(PYTHON) -m stillwar figures

tables:
	$(PYTHON) -m stillwar tables

lint:
	$(RUFF) check src tests

test:
	$(PYTEST) tests/ -q

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.cache')]]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('data/processed').glob('*.json')]"
	@echo "Cleaned .cache, __pycache__ and data/processed/*.json"
