"""Command-line interface: ``python -m stillwar <stage>``.

Each stage is independently invocable and idempotent. Stages that are expensive check
:func:`stillwar.cache.is_fresh` first, so ``make all`` on an unchanged tree only
re-renders tables and figures. Pass ``--force`` to bypass every cache check.

Stage order and dependencies::

    verify -> features -> train -> baselines -> simulate -> experiments
                                                              |
                                          tables <------------+
                                          figures <-----------+
    trials   (separate, expensive, cached; produces the DSR trial distribution)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from stillwar import cache
from stillwar.artifacts import MODEL_KEYS, load_all, save_baselines, save_forecasts
from stillwar.config import Config, load_config
from stillwar.data import download, verify_manifest
from stillwar.experiments import (
    run_shrinkage_experiment,
    run_unconstrained_and_tolerance_experiment,
    run_volatility_penalty_experiment,
)
from stillwar.features import load_dataset
from stillwar.figures import plot_all
from stillwar.provenance import artifact_metadata, write_json
from stillwar.simulate import run_all
from stillwar.tables import JOURNAL, bless_golden, load_blocks, sync_manuscript
from stillwar.train import gate_activations, train_all_models
from stillwar.trials import run_trial_grid

PROCESSED = Path("data/processed")
CHECKPOINTS = Path("models/checkpoints")

_FEATURE_DEPS = ("config", "data", "features", "universe")
_TRAIN_DEPS = ("config", "models", "train", "features", "data", "stats")
_SIMULATE_DEPS = ("config", "params", "portfolio", "cache", "metrics", "features", "data", "stats", "simulate")
_EXPERIMENT_DEPS = ("config", "params", "portfolio", "cache", "metrics", "features", "data", "experiments")
_TRIAL_DEPS = ("config", "params", "portfolio", "cache", "features", "data", "trials")


def _skip(stage: str) -> None:
    """Report that a stage was skipped because its cache record is current."""
    print(f"[skip] {stage}: up to date")


def _forecast_inputs() -> dict[str, str]:
    """Content digests of every forecast artifact the simulation stages consume.

    A stage is fresh only if these match. Keying on the configuration alone would let
    a retrain leave ``simulate`` marked fresh, so the tables would describe the
    previous forecasts.
    """
    digests = {f"forecasts_{key}": cache.file_digest(PROCESSED / f"forecasts_{key}.npz") for key in MODEL_KEYS}
    digests["forecasts_baselines"] = cache.file_digest(PROCESSED / "forecasts_baselines.npz")
    return digests


def stage_verify(cfg: Config) -> int:
    """Check every raw CSV against the SHA-256 manifest."""
    count = verify_manifest(cfg.data.raw_dir, cfg.data.manifest)
    print(f"Verified SHA-256 manifest across {count} constituent files.")
    return 0


def stage_ingest(cfg: Config) -> int:
    """Download any missing raw CSVs and rewrite the manifest."""
    download(cfg)
    return 0


def stage_features(cfg: Config) -> int:
    """Build and persist the log-signature feature tensor."""
    if cache.is_fresh("features", cfg, _FEATURE_DEPS):
        _skip("features")
        return 0
    ds = load_dataset(cfg)
    print(f"Features extracted: {ds.x.shape} across {len(ds.dates)} decision dates.")
    cache.mark_done("features", cfg, _FEATURE_DEPS)
    return 0


def stage_train(cfg: Config) -> int:
    """Train all four models across all seeds and export their forecasts."""
    if cache.is_fresh("train", cfg, _TRAIN_DEPS):
        _skip("train")
        return 0
    ds = load_dataset(cfg)
    trained = train_all_models(cfg, ds, CHECKPOINTS)
    for key in MODEL_KEYS:
        save_forecasts(PROCESSED / f"forecasts_{key}.npz", trained[key]["panel_a"], trained[key]["panel_b"], ds, cfg.seeds)
        print(f"Trained {key}: lr={trained[key]['lr']}")

    gates = gate_activations("logsig", ds, cfg, trained["logsig"]["fits"][cfg.seeds[0]])
    if gates.size:
        from stillwar.stats import gate_weights

        write_json(
            PROCESSED / "gate_weights.json",
            artifact_metadata(cfg, _TRAIN_DEPS),
            gate_weights(gates),
        )
    cache.mark_done("train", cfg, _TRAIN_DEPS)
    return 0


def stage_baselines(cfg: Config) -> int:
    """Produce the non-learned benchmark forecasts (Hist-MV and GMV)."""
    if cache.is_fresh("baselines", cfg, _FEATURE_DEPS):
        _skip("baselines")
        return 0
    ds = load_dataset(cfg)
    baselines = {
        "hist_mv": ds.historical_forecast(cfg.trading.alpha)[ds.idx["test_b"]],
        "gmv": np.zeros((len(ds.idx["test_b"]), len(ds.tickers))),
    }
    save_baselines(PROCESSED / "forecasts_baselines.npz", baselines, ds)
    print(f"Baseline forecasts written: hist_mv {baselines['hist_mv'].shape}, gmv {baselines['gmv'].shape}.")
    cache.mark_done("baselines", cfg, _FEATURE_DEPS)
    return 0


def stage_simulate(cfg: Config) -> int:
    """Produce Tables 1-3 and the Phase-4 diagnostics."""
    inputs = _forecast_inputs()
    if cache.is_fresh("simulate", cfg, _SIMULATE_DEPS, inputs):
        _skip("simulate")
        return 0
    forecasts, baselines = load_all(cfg, PROCESSED)
    result = run_all(cfg, forecasts, baselines, PROCESSED)
    print(f"Tables written: {', '.join(sorted(result))}.")
    cache.mark_done("simulate", cfg, _SIMULATE_DEPS, inputs)
    return 0


def stage_experiments(cfg: Config) -> int:
    """Run the three structural extensions of Section 3.7."""
    inputs = _forecast_inputs()
    if cache.is_fresh("experiments", cfg, _EXPERIMENT_DEPS, inputs):
        _skip("experiments")
        return 0
    forecasts, _ = load_all(cfg, PROCESSED)
    logsig = forecasts["logsig"]
    run_unconstrained_and_tolerance_experiment(cfg, logsig)
    run_shrinkage_experiment(cfg, logsig)
    run_volatility_penalty_experiment(cfg, logsig)
    print("Structural extensions written.")
    cache.mark_done("experiments", cfg, _EXPERIMENT_DEPS, inputs)
    return 0


def stage_trials(cfg: Config) -> int:
    """Evaluate the 1,225 configurations behind the Deflated Sharpe ratio."""
    inputs = _forecast_inputs()
    if cache.is_fresh("trials", cfg, _TRIAL_DEPS, inputs):
        _skip("trials")
        return 0
    forecasts, _ = load_all(cfg, PROCESSED)
    payload = run_trial_grid(cfg, forecasts, PROCESSED / "trials.json")
    print(f"Trial grid written: {payload['n_trials']} configurations.")
    cache.mark_done("trials", cfg, _TRIAL_DEPS, inputs)
    return 0


def stage_figures(cfg: Config) -> int:
    """Render every manuscript figure from the JSON artifacts."""
    written = plot_all(PROCESSED)
    print(f"Figures written: {', '.join(p.name for p in written)}.")
    return 0


def stage_tables(cfg: Config) -> int:
    """Regenerate the manuscript tables and sync ``docs/JOURNAL.md``."""
    changed = sync_manuscript(load_blocks(PROCESSED), JOURNAL)
    print(f"Manuscript tables {'updated' if changed else 'already in sync'} ({JOURNAL}).")
    return 0


STAGES = {
    "verify": stage_verify,
    "ingest": stage_ingest,
    "features": stage_features,
    "train": stage_train,
    "baselines": stage_baselines,
    "simulate": stage_simulate,
    "experiments": stage_experiments,
    "trials": stage_trials,
    "figures": stage_figures,
    "tables": stage_tables,
}

ALL_STAGES = (
    "verify",
    "features",
    "train",
    "baselines",
    "simulate",
    "experiments",
    "figures",
    "tables",
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m stillwar``."""
    parser = argparse.ArgumentParser(prog="stillwar", description="Stillwar reproducibility pipeline")
    parser.add_argument("stage", choices=[*STAGES, "all"], nargs="?", default="verify")
    parser.add_argument("--config", default="configs/base.yaml", help="path to the YAML configuration")
    parser.add_argument("--force", action="store_true", help="bypass every cache check")
    parser.add_argument(
        "--bless",
        action="store_true",
        help="with the tables stage, write tests/golden.json from the current artifacts",
    )
    args = parser.parse_args(argv)

    if args.bless and args.stage != "tables":
        parser.error("--bless requires the tables stage")

    cache.set_force(args.force)
    cfg = load_config(args.config)

    stages = ALL_STAGES if args.stage == "all" else (args.stage,)
    for name in stages:
        STAGES[name](cfg)

    if args.bless:
        hashes = bless_golden(PROCESSED, Path("tests/golden.json"))
        print(f"Golden hashes blessed: {len(hashes)} artifacts (tests/golden.json).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
