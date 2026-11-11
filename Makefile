# Makefile for the Stillwar reproducibility pipeline.
#
# `make all` regenerates every table and figure from the raw snapshot. Expensive
# stages are content-addressed (see src/stillwar/provenance/cache.py), so a second
# `make all` on an unchanged tree only re-renders tables and figures. Use
# `make all FORCE=1` to recompute everything.
.PHONY: all help setup verify ingest features train baselines simulate experiments figures tables trials test lint format typecheck manuscript check clean

PYTHON ?= uv run python
PYTEST ?= uv run pytest
RUFF   ?= uv run ruff
MYPY   ?= uv run mypy
FORCE  ?= 0
FORCEARG = $(if $(filter 1,$(FORCE)),--force,)
CONFIG ?= configs/base.yaml
SET    ?=
SETARG = $(if $(SET),--set $(SET),)

# Make the source tree importable even when the package has not been installed
# (useful for a fresh checkout and for CI jobs that only need the CLI).
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))

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
	@echo "  make format      - format src and tests with ruff"
	@echo "  make typecheck   - run mypy over the package"
	@echo "  make manuscript  - structural check of docs/JOURNAL.md"
	@echo "  make check       - lint + test + manuscript check"
	@echo "  make clean       - remove caches and generated binary arrays"
	@echo ""
	@echo "  FORCE=1 recomputes every stage regardless of the cache."
	@echo "  CONFIG=configs/local.yaml selects another configuration."
	@echo "  SET=workers__backtest=1 overrides one config value (repeatable)."

all: verify features train baselines simulate experiments figures tables test

setup:
	uv sync --extra dev

verify:
	$(PYTHON) -m stillwar verify $(FORCEARG) --config $(CONFIG) $(SETARG)

ingest:
	$(PYTHON) -m stillwar ingest $(FORCEARG) --config $(CONFIG) $(SETARG)

features:
	$(PYTHON) -m stillwar features $(FORCEARG) --config $(CONFIG) $(SETARG)

train:
	$(PYTHON) -m stillwar train $(FORCEARG) --config $(CONFIG) $(SETARG)

baselines:
	$(PYTHON) -m stillwar baselines $(FORCEARG) --config $(CONFIG) $(SETARG)

simulate:
	$(PYTHON) -m stillwar simulate $(FORCEARG) --config $(CONFIG) $(SETARG)

experiments:
	$(PYTHON) -m stillwar experiments $(FORCEARG) --config $(CONFIG) $(SETARG)

trials:
	$(PYTHON) -m stillwar trials $(FORCEARG) --config $(CONFIG) $(SETARG)

figures:
	$(PYTHON) -m stillwar figures --config $(CONFIG) $(SETARG)

tables:
	$(PYTHON) -m stillwar tables --config $(CONFIG) $(SETARG)

lint:
	$(RUFF) check src tests

format:
	$(RUFF) format src tests

typecheck:
	$(MYPY) src/stillwar

manuscript:
	$(PYTHON) tools/check_manuscript.py

test:
	$(PYTEST) tests/ -q

check: lint test manuscript

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.cache'), pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache'), pathlib.Path('.mypy_cache')]]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('data/processed').glob('*.npz')]; [p.unlink() for p in pathlib.Path('data/processed').glob('*_cache.json')]"
	@echo "Cleaned caches and generated binary arrays (tracked JSON artifacts kept)."
