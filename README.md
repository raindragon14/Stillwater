# When the Optimizer Refuses to Trade

Reproducible code and manuscript for **“When the Optimizer Refuses to Trade: A KKT
No-Trade Zone and Lévy-Area Signal Activation under Weak Signals in LQ45 Equities”**.

Muhammad Reihan Pandanarang — Mathematics Department,
Universitas Negeri Semarang.

The repository contains both the empirical pipeline and the paper:

| Path | Purpose |
|:--|:--|
| [`src/stillwar/`](src/stillwar/) | Modular Python package (data, features, models, optimization, backtest, reporting) |
| [`docs/JOURNAL.md`](docs/JOURNAL.md) | Complete manuscript; Tables 1–3 are generated from the artifacts |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Audit of the original draft and the corrections applied |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Package map, dependency order and invariants |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | What is guaranteed, what changed, and what is outstanding |
| [`configs/base.yaml`](configs/base.yaml) | Single source of truth for every number |
| [`data/raw/`](data/raw/) | Archived Yahoo Finance snapshot + SHA-256 manifest |
| [`data/processed/`](data/processed/) | Provenance-stamped result artifacts (Tables 1–3, KKT log, trials) |
| [`reports/figures/`](reports/figures/) | Publication figures at 300 DPI |

## Quickstart

```bash
uv sync --extra dev          # Python 3.14, CPU torch, test/lint tools
uv run python -m stillwar verify
make all                     # verify -> features -> train -> baselines -> simulate -> experiments -> figures -> tables -> test
```

`make all` is content-addressed: a second run on an unchanged tree is fast. Use
`make all FORCE=1` to recompute every stage. The 1,225-trial Deflated-Sharpe grid is
separate and expensive:

```bash
make trials                  # about 20 minutes, once
```

> `make` is a convenience wrapper. On Windows without GNU make, run the same stages
> directly with `uv run python -m stillwar <stage>`; every stage accepts `--config`,
> `--set SECTION__KEY=VALUE` and `--force`.

Quality gates:

```bash
make check                   # ruff + pytest + manuscript structural check
make lint
make test
```

## Pipeline stages

| Stage | What it does |
|:--|:--|
| `verify` | Check every raw CSV against `data/raw/manifest.sha256` |
| `features` | Build the 45 order-2 log-signature features and purge/embargo splits |
| `train` | Train LogSig, LogSig-Inc, MLP and LSTM across five pinned seeds |
| `baselines` | Build the Hist-MV and GMV forecast arrays |
| `simulate` | Produce Tables 1–3, the KKT log and the Phase-4 diagnostics |
| `experiments` | Run the structural extensions (cap-drift buffer, shrinkage, volatility-scaled penalties) |
| `trials` | Evaluate the 1,225 Deflated-Sharpe configurations (about 20 min) |
| `figures` | Render the manuscript figures |
| `tables` | Regenerate Tables 1–3 and sync `docs/JOURNAL.md` |

`python -m stillwar <stage>` is the low-level entry point; `make` is the convenience
wrapper.

## Modular layout

```text
src/stillwar/
├── config/        # frozen Pydantic schema, YAML loader, BacktestParams
├── data/          # universe, Yahoo ingestion, manifest, adjusted panel
├── features/      # log-signature, Levy areas, standardised dataset, splits
├── models/        # MSGN/LogSig-Inc/MLP/LSTM, loss, deterministic trainer
├── optimization/  # Ledoit-Wolf, Theorem 1 KKT test, SLSQP QP
├── backtest/      # execution engine, costs, drift, metrics, statistics
├── experiments/   # table runs, sensitivity grids, DSR trials
├── reporting/     # JSON -> Markdown tables, JSON -> figures, manuscript sync
├── provenance/    # metadata, forecast contracts, content-addressed cache
├── cli.py         # stage dispatcher
└── __main__.py    # `python -m stillwar` wrapper
```

The package is the public API; `stillwar.cli` orchestrates the stages and
`stillwar.__main__` is the `python -m stillwar` wrapper. There are no top-level
compatibility modules left.

## Reproducibility

- **Data**: the raw snapshot is committed with a SHA-256 manifest and never patched;
  missing prices are forward-filled only.
- **Config**: every number lives in `configs/base.yaml`; `Config` is frozen and rejects
  unknown keys.
- **Seeds**: five pinned seeds, deterministic CPU kernels and one torch thread.
- **Cache**: content-addressed stage and backtest cache; `--force` bypasses it.
- **Artifacts**: result JSONs contain numbers, never formatted strings; the manuscript
  tables are rendered from them and compared cell-by-cell. Figures are rendered from the
  same artifacts; PNG bytes can differ across matplotlib/font builds even when the
  plotted data are identical.
- **Environment**: `uv.lock` pins the Python environment; `Dockerfile` pins the OS-level
  image; `.devcontainer/` provides the same environment in VS Code.
- **CI**: GitHub Actions runs lint, tests, the manifest check, a feature build and the
  manuscript-sync check on a clean machine.
- **Pipeline**: `dvc.yaml` describes the same stages with DVC lineage for reviewers who
  prefer `dvc repro`.

## Manuscript

The manuscript is Markdown, not Word-first. `docs/JOURNAL.docx` is a generated artifact
and is ignored by Git; regenerate it with the target journal's template after the
journal and author metadata are confirmed. Tables 1–3 sit between
`<!-- BEGIN TABLE n -->` / `<!-- END TABLE n -->` markers and are rewritten by
`make tables`.

Open author decisions are listed in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Optional extras

| Extra | Adds |
|:--|:--|
| `dev` | pytest, hypothesis, pytest-regressions, ruff, mypy, pre-commit |
| `download` | `yfinance` for `make ingest` (the archived snapshot is the default) |
| `tracking` | MLflow for experiment tracking |
| `config` | Hydra/OmegaConf for config composition and sweeps |
| `validation` | pandera DataFrame schemas |
| `qp` | cvxpy as an independent QP cross-check |
| `pipeline` | DVC for `dvc repro` lineage |

## Citation

See [`CITATION.cff`](CITATION.cff). The code release is archived at
[`raindragon14/Stillwar`](https://github.com/raindragon14/Stillwar).

## License

MIT. See [`LICENSE`](LICENSE).
