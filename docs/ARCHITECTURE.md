# Architecture

The package is organised by **responsibility**, not by file type. A module may import
only from packages below it in the dependency order. `stillwater.cli` is the only
orchestrator; no package imports the CLI.

## Dependency order

```text
config  ->  data  ->  features  ->  models
                                   |
                         optimization  ->  backtest
                                   |
                            experiments  ->  reporting
                                   |
                              provenance
```

`provenance` is deliberately low-level: it records hashes and metadata. The one
exception is `provenance/cache.py`, which also hosts the backtest cache and therefore
imports `backtest.engine`; keeping the cache key and the engine in one place is worth
that single edge. `reporting` reads artifacts; it never re-simulates.

## Package map

| Package | Responsibility | Key modules |
|:--|:--|:--|
| `stillwater.config` | Frozen Pydantic schema, YAML loading, backtest parameters | `schema.py`, `loader.py`, `params.py` |
| `stillwater.data` | Universe, Yahoo ingestion, SHA-256 manifest, aligned adjusted panel | `universe.py`, `ingest.py`, `panel.py` |
| `stillwater.features` | Order-2 log-signature, Levy areas, standardised dataset, purged splits | `logsig.py`, `dataset.py` |
| `stillwater.models` | MSGN/LogSig-Inc/MLP/LSTM, composite loss, deterministic trainer | `architectures.py`, `loss.py`, `trainer.py` |
| `stillwater.optimization` | Ledoit-Wolf covariance, Theorem 1 KKT test, SLSQP QP | `covariance.py`, `kkt.py`, `qp.py` |
| `stillwater.backtest` | Execution loop, costs, drift, metrics, statistics | `engine.py`, `metrics.py`, `stats.py` |
| `stillwater.experiments` | Table 1-3 runs, sensitivity grids, 1,225-trial DSR grid | `simulate.py`, `sensitivity.py`, `trials.py` |
| `stillwater.reporting` | JSON -> Markdown tables, JSON -> figures, manuscript sync | `tables.py`, `figures.py` |
| `stillwater.provenance` | Artifact metadata, forecast contracts, content-addressed cache | `metadata.py`, `artifacts.py`, `cache.py` |
| `stillwater.cli` | Stage dispatcher (`python -m stillwater <stage>`) | `cli.py` |

## Pipeline stages

```text
verify -> features -> train -> baselines -> simulate -> experiments -> figures -> tables
                                                                  \
                                                                   trials (expensive, once)
```

Each stage is idempotent. Expensive stages check the content-addressed cache in
`provenance/cache.py`; `--force` bypasses it. The `tables` stage rewrites only the
regions between `<!-- BEGIN TABLE n -->` and `<!-- END TABLE n -->` in
`docs/JOURNAL.md`, so prose is never touched.

## Invariants

1. **Raw data is immutable.** `data/raw/manifest.sha256` is verified before any panel
   is built; missing values are forward-filled only.
2. **Configuration is frozen.** `Config` rejects unknown keys and every number lives in
   `configs/base.yaml`; the CLI never carries a silent default.
3. **No leakage.** Feature standardisation uses training stock-days only; the split
   purge and embargo are tested.
4. **Determinism.** Seeds are pinned; torch is pinned to one thread; the cache key
   includes config, code and input digests.
5. **Artifacts are numbers.** Result JSONs contain floats, never pre-formatted strings;
   formatting lives in `reporting/`.
6. **The manuscript is generated.** Tables 1-3 are rendered from `data/processed/*.json`
   and compared cell-by-cell in `tests/test_provenance.py`.

## Adding a stage

1. Add the pure logic to the relevant package (not to `cli.py`).
2. Add a `stage_*` function to `cli.py` that loads the config, checks the cache, calls
   the logic, and writes a provenance-stamped artifact.
3. Add a `Makefile` target and a `dvc.yaml` stage if the output is a durable artifact.
4. Add a test under `tests/` and, if the stage feeds the manuscript, a table renderer
   and a golden hash.
