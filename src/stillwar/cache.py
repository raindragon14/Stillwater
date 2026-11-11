"""Compatibility shim for the old ``stillwar.cache`` import path.

New code should import from :mod:`stillwar.provenance.cache`.
"""

from stillwar.provenance.cache import (
    BACKTEST_DIR,
    CACHE_DIR,
    DATASET_DIR,
    STAGE_DIR,
    backtest_key,
    cached_backtest,
    clear,
    dataset_cache_path,
    file_digest,
    force,
    forecasts_digest,
    is_fresh,
    mark_done,
    prices_digest,
    run_backtests,
    set_force,
    stats,
)

__all__ = [
    "BACKTEST_DIR",
    "CACHE_DIR",
    "DATASET_DIR",
    "STAGE_DIR",
    "backtest_key",
    "cached_backtest",
    "clear",
    "dataset_cache_path",
    "file_digest",
    "force",
    "forecasts_digest",
    "is_fresh",
    "mark_done",
    "prices_digest",
    "run_backtests",
    "set_force",
    "stats",
]
