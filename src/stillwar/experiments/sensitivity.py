"""Sensitivity grids and structural extensions of JOURNAL Sections 3.6, 3.7 and Table S1.

Each experiment is a list of :class:`GridCell` values crossed with the seed set and run
through one shared runner, so the three experiments no longer repeat the same
load / pool / slice / aggregate boilerplate. Every result is written as raw numbers
with a provenance block; formatting happens in :mod:`stillwar.tables`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stillwar.backtest.engine import BacktestResult, slice_panel_a
from stillwar.backtest.metrics import backtest_metrics, seed_summary, trigger_concentration
from stillwar.config import BacktestParams, Config
from stillwar.data import load_panel
from stillwar.features import load_dataset
from stillwar.provenance import cache
from stillwar.provenance.metadata import artifact_metadata, write_json

__all__ = [
    "run_shrinkage_experiment",
    "run_unconstrained_and_tolerance_experiment",
    "run_volatility_penalty_experiment",
]

_DEPENDS_ON = ("config", "params", "optimization", "backtest", "provenance", "features", "data")


@dataclass(frozen=True)
class GridCell:
    """One configuration in a sensitivity grid.

    Attributes
    ----------
    label
        Stable identifier used as the JSON key and the table row label.
    params
        Backtest parameters for this cell.
    """

    label: str
    params: BacktestParams


def _run_grid(
    cells: Sequence[GridCell],
    forecasts: np.ndarray,
    panel: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    n_cycles_a: int,
    rebalance_freq: int,
    workers: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[BacktestResult]]]:
    """Run every ``(cell, seed)`` pair and return per-cell metrics and raw results.

    Parameters
    ----------
    cells
        Configurations to evaluate.
    forecasts
        ``(n_seeds, n_dates, n_assets)`` forecast array.
    panel
        Adjusted close panel.
    dates
        Panel B decision dates.
    n_cycles_a
        Number of rebalance cycles in Panel A.
    rebalance_freq
        Trading days per cycle.
    workers
        Maximum concurrent processes.

    Returns
    -------
    tuple of (dict, dict)
        Per-cell seed metrics ``{label: [{"panel_a", "panel_b", "kkt"}, ...]}`` and the
        raw :class:`BacktestResult` objects in the same order.
    """
    tasks = [
        (panel, dates, cell.params, forecasts[seed_idx])
        for cell in cells
        for seed_idx in range(len(forecasts))
    ]
    results = cache.run_backtests(tasks, workers)

    metrics: dict[str, list[dict[str, Any]]] = {cell.label: [] for cell in cells}
    raw: dict[str, list[BacktestResult]] = {cell.label: [] for cell in cells}
    n_seeds = len(forecasts)
    for cell_idx, cell in enumerate(cells):
        for seed_idx in range(n_seeds):
            result = results[cell_idx * n_seeds + seed_idx]
            metrics[cell.label].append(
                {
                    "panel_b": backtest_metrics(result),
                    "panel_a": backtest_metrics(slice_panel_a(result, n_cycles_a, rebalance_freq)),
                    "kkt": _kkt_counts(result),
                }
            )
            raw[cell.label].append(result)
    return metrics, raw


def _kkt_counts(result: BacktestResult) -> dict[str, float]:
    """Decision-regime counts from a backtest's KKT log."""
    logs = result["kkt_logs"]
    return {
        "cap_drift": float(sum(1 for k in logs if k["is_cap_drift"])),
        "signal_trade": float(
            sum(1 for k in logs if k["kkt_should_trade"] and not k["is_cap_drift"])
        ),
        "optimal_hold": float(
            sum(1 for k in logs if not k["kkt_should_trade"] and not k["is_cap_drift"])
        ),
        "total": float(len(logs)),
    }


def _mean_of(runs: Sequence[dict[str, Any]], path: tuple[str, ...]) -> float:
    """Mean of a nested metric across seeds."""
    values = []
    for run in runs:
        node: Any = run
        for key in path:
            node = node[key]
        values.append(float(node))
    return float(np.mean(values)) if values else 0.0


def run_unconstrained_and_tolerance_experiment(
    cfg: Config,
    forecasts: np.ndarray,
    out_path: str | Path = "data/processed/exp_unconstrained_tolerance.json",
) -> dict[str, Any]:
    """Table S1 extensions (i) and the unconstrained-cap diagnostic (Section 3.6).

    Two parts: a single unconstrained cell (``w_max = 1.0``) that documents Michaud
    error maximisation, and a ``(w_max, epsilon)`` grid that measures how the cap-drift
    buffer trades forced rebalances against interior KKT holds.

    Parameters
    ----------
    cfg
        Configuration.
    forecasts
        ``(n_seeds, n_dates, n_assets)`` LogSig forecasts.
    out_path
        Destination JSON.

    Returns
    -------
    dict
        ``unconstrained`` and ``tolerance_grid`` result blocks.
    """
    ds = load_dataset(cfg)
    panel = load_panel(cfg).close
    dates_b = list(ds.dates[ds.idx["test_b"]])
    n_cycles_a = cfg.reporting.panel_a_rebalances
    base = BacktestParams.from_config(cfg)
    workers = min(len(cfg.seeds), cfg.workers.backtest)

    unconstrained_metrics, unconstrained_raw = _run_grid(
        [GridCell("w_max_1.00", base.replace(w_max=1.0))],
        forecasts, panel, dates_b, n_cycles_a, base.rebalance_freq, workers,
    )
    unconstrained = unconstrained_metrics["w_max_1.00"]

    weights = []
    for result in unconstrained_raw["w_max_1.00"]:
        w = np.array(result["weights_history"])
        weights.append(
            {
                "max_weight_mean": float(np.max(w, axis=1).mean()),
                "hhi_mean": float(np.mean(np.sum(w**2, axis=1))),
            }
        )

    grid_cells = [
        GridCell(f"w_max_{wm:.2f}_eps_{eps:.3f}", base.replace(w_max=wm, cap_drift_buffer=eps))
        for wm in (0.15, 0.20)
        for eps in (0.0, 0.005, 0.010, 0.020, 0.050)
    ]
    grid, _ = _run_grid(
        grid_cells, forecasts, panel, dates_b, n_cycles_a, base.rebalance_freq, workers
    )

    tolerance = {}
    for cell in grid_cells:
        runs = grid[cell.label]
        n_dates = float(runs[0]["kkt"]["total"])
        tolerance[cell.label] = {
            "w_max": cell.params.w_max,
            "epsilon": cell.params.cap_drift_buffer,
            "cap_drift_mean": _mean_of(runs, ("kkt", "cap_drift")),
            "signal_trade_mean": _mean_of(runs, ("kkt", "signal_trade")),
            "optimal_hold_mean": _mean_of(runs, ("kkt", "optimal_hold")),
            "cap_drift_share": _mean_of(runs, ("kkt", "cap_drift")) / n_dates,
            "signal_trade_share": _mean_of(runs, ("kkt", "signal_trade")) / n_dates,
            "optimal_hold_share": _mean_of(runs, ("kkt", "optimal_hold")) / n_dates,
            "panel_b": {
                key: _mean_of(runs, ("panel_b", key))
                for key in ("cagr", "sharpe", "mdd", "turnover")
            },
        }

    results = {
        "unconstrained": {
            "panel_b": {
                key: seed_summary([r["panel_b"][key] for r in unconstrained])
                for key in ("cagr", "sharpe", "mdd", "turnover")
            },
            "max_weight_mean": float(np.mean([w["max_weight_mean"] for w in weights])),
            "hhi_mean": float(np.mean([w["hhi_mean"] for w in weights])),
        },
        "tolerance_grid": tolerance,
    }
    write_json(out_path, artifact_metadata(cfg, _DEPENDS_ON), results)
    return results


def run_shrinkage_experiment(
    cfg: Config,
    forecasts: np.ndarray,
    out_path: str | Path = "data/processed/exp_shrinkage.json",
) -> dict[str, Any]:
    """Table S1 extension (ii): forecast shrinkage ``alpha`` (Section 3.7).

    ``alpha = 0`` keeps the raw forecast, ``alpha = 1`` collapses it to the
    cross-sectional mean and therefore to the GMV limit, so the grid traces the
    inverted-U Sharpe profile reported in the manuscript.

    Parameters
    ----------
    cfg
        Configuration.
    forecasts
        ``(n_seeds, n_dates, n_assets)`` LogSig forecasts.
    out_path
        Destination JSON.

    Returns
    -------
    dict
        ``{alpha_label: {"alpha": float, "panel_a": {...}, "panel_b": {...}}}``.
    """
    ds = load_dataset(cfg)
    panel = load_panel(cfg).close
    dates_b = list(ds.dates[ds.idx["test_b"]])
    base = BacktestParams.from_config(cfg)
    workers = min(len(cfg.seeds), cfg.workers.backtest)

    alphas = (0.0, 0.25, 0.50, 0.75, 1.0)
    cells = [GridCell(f"{a:.2f}", base.replace(alpha=a)) for a in alphas]
    grid, _ = _run_grid(
        cells, forecasts, panel, dates_b, cfg.reporting.panel_a_rebalances, base.rebalance_freq, workers
    )

    results = {}
    for cell, alpha in zip(cells, alphas, strict=True):
        runs = grid[cell.label]
        results[cell.label] = {
            "alpha": alpha,
            "panel_a": {k: _mean_of(runs, ("panel_a", k)) for k in ("cagr", "sharpe", "mdd", "turnover")},
            "panel_b": {k: _mean_of(runs, ("panel_b", k)) for k in ("cagr", "sharpe", "mdd", "turnover")},
        }
    write_json(out_path, artifact_metadata(cfg, _DEPENDS_ON), results)
    return results


def run_volatility_penalty_experiment(
    cfg: Config,
    forecasts: np.ndarray,
    out_path: str | Path = "data/processed/exp_volatility_penalty.json",
) -> dict[str, Any]:
    """Table S1 extension (iii): volatility-scaled penalties (Sections 3.5 and 3.7).

    Replaces the uniform band with ``eta_i proportional to sigma_i / mean(sigma)`` and
    measures whether trigger concentration falls. Trigger statistics are reported for
    the primary seed, matching the manuscript's Figure 3.

    Parameters
    ----------
    cfg
        Configuration.
    forecasts
        ``(n_seeds, n_dates, n_assets)`` LogSig forecasts.
    out_path
        Destination JSON.

    Returns
    -------
    dict
        ``uniform`` and ``volatility_scaled`` blocks, each with panel metrics and
        trigger concentration.
    """
    ds = load_dataset(cfg)
    loaded = load_panel(cfg)
    panel, tickers = loaded.close, loaded.tickers
    dates_b = list(ds.dates[ds.idx["test_b"]])
    base = BacktestParams.from_config(cfg)
    workers = min(len(cfg.seeds), cfg.workers.backtest)

    cells = [
        GridCell("uniform", base),
        GridCell("volatility_scaled", base.replace(volatility_scaled=True)),
    ]
    grid, raw = _run_grid(
        cells, forecasts, panel, dates_b, cfg.reporting.panel_a_rebalances, base.rebalance_freq, workers
    )

    results = {}
    for cell in cells:
        runs = grid[cell.label]
        results[cell.label] = {
            "panel_b": {k: _mean_of(runs, ("panel_b", k)) for k in ("cagr", "sharpe", "mdd", "turnover")},
            "triggers": trigger_concentration(raw[cell.label][0]["kkt_logs"], tickers),
        }
    write_json(out_path, artifact_metadata(cfg, _DEPENDS_ON), results)
    return results
