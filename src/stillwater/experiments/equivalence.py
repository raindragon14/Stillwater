"""Short-circuit equivalence check for the Theorem 1 diagnostic (JOURNAL Section 2.6).

The production backtest skips the QP on interior hold dates for speed: when Theorem 1
says the incumbent is optimal, :func:`stillwater.backtest.engine.run_backtest` sets
``solver_traded = False`` without asking the solver. That shortcut is what makes the
128-date run cheap, but it also means the hold column of Table 3 is assigned by
construction rather than measured.

This experiment re-runs every optimizing strategy with ``short_circuit = False`` and
records, per run, whether the shortcut changed any trade decision. The manuscript's
claim that the agreement of Table 3 does not depend on skipping the QP on holds
(Section 2.6, Appendix A.4) is therefore backed by an artifact instead of by prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stillwater.backtest.engine import BacktestResult
from stillwater.config import BacktestParams, Config
from stillwater.data import load_panel
from stillwater.experiments.simulate import LEARNED_MODELS
from stillwater.features import load_dataset
from stillwater.provenance import cache
from stillwater.provenance.metadata import artifact_metadata, write_json

__all__ = ["run_short_circuit_equivalence_experiment"]

_DEPENDS_ON = ("config", "optimization", "backtest", "provenance", "features", "data")


def _runs(
    cfg: Config,
    forecasts: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    base: BacktestParams,
) -> list[tuple[str, np.ndarray, BacktestParams]]:
    """Every optimizing strategy at the primary configuration, one entry per seed.

    Equal weight is excluded because ``forecasts = None`` bypasses the optimizer
    entirely and produces no KKT log to compare.
    """
    runs = [
        (f"{name}|s{seed}", forecasts[key][seed], base)
        for name, key in LEARNED_MODELS
        for seed in range(len(cfg.seeds))
    ]
    runs += [
        ("Hist-MV, penalized", baselines["hist_mv"], base),
        ("Hist-MV, unpenalized", baselines["hist_mv"], base.replace(kappa=0.0)),
        ("GMV", baselines["gmv"], base),
        ("GMV, unpenalized", baselines["gmv"], base.replace(kappa=0.0)),
    ]
    return runs


def _compare(label: str, short: BacktestResult, full: BacktestResult) -> dict[str, Any]:
    """Agreement between one short-circuited run and its full-solve twin."""
    short_decisions = [row["solver_traded"] for row in short["kkt_logs"]]
    full_decisions = [row["solver_traded"] for row in full["kkt_logs"]]
    agree = [a == b for a, b in zip(short_decisions, full_decisions, strict=True)]
    valid = [row for row in full["kkt_logs"] if not np.isnan(row["delta_g"])]
    return {
        "label": label,
        "n_dates": len(short_decisions),
        "n_agree": int(sum(agree)),
        "n_disagree": int(len(agree) - sum(agree)),
        "agreement_rate": float(np.mean(agree)) if agree else 0.0,
        "n_traded_short_circuit": int(sum(short_decisions)),
        "n_traded_full_solve": int(sum(full_decisions)),
        "n_interior_trades": int(
            sum(1 for row in full["kkt_logs"] if row["kkt_should_trade"] and not row["is_cap_drift"])
        ),
        "n_interior_holds": int(
            sum(1 for row in full["kkt_logs"] if not row["kkt_should_trade"] and not row["is_cap_drift"])
        ),
        "n_cap_drift": int(sum(1 for row in full["kkt_logs"] if row["is_cap_drift"])),
        "n_boundary": int(sum(1 for row in full["kkt_logs"] if row["is_boundary"])),
        "kkt_solver_agreement_full_solve": (
            float(np.mean([row["kkt_should_trade"] == row["solver_traded"] for row in valid]))
            if valid
            else 0.0
        ),
        "max_abs_daily_return_diff": (
            float(np.max(np.abs(short["daily_returns"] - full["daily_returns"])))
            if len(short["daily_returns"])
            else 0.0
        ),
        "max_abs_turnover_diff": (
            float(
                np.max(
                    np.abs(
                        np.asarray(short["executed_turnovers"])
                        - np.asarray(full["executed_turnovers"])
                    )
                )
            )
            if short["executed_turnovers"]
            else 0.0
        ),
        "disagreement_cycles": [
            row["cycle"]
            for row, a, b in zip(short["kkt_logs"], short_decisions, full_decisions, strict=True)
            if a != b
        ],
    }


def run_short_circuit_equivalence_experiment(
    cfg: Config,
    forecasts: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    out_path: str | Path = "data/processed/exp_short_circuit_equivalence.json",
) -> dict[str, Any]:
    """Re-run every strategy with the hold shortcut disabled and record agreement.

    Parameters
    ----------
    cfg
        Configuration.
    forecasts
        ``{model: (n_seeds, n_dates, n_assets)}`` arrays for the four learned models.
    baselines
        ``{"hist_mv": (n_dates, n_assets), "gmv": (n_dates, n_assets)}``.
    out_path
        Destination JSON.

    Returns
    -------
    dict
        ``primary`` (the Table 3 run, ``LogSig|s0``), ``all_runs`` (one row per run),
        ``n_runs`` and ``all_agree``.

    Notes
    -----
    Both runs are cached individually, so the short-circuited half is a cache hit
    whenever ``simulate`` already ran. The full-solve half costs 128 QP solves per run
    and is the only new work.
    """
    ds = load_dataset(cfg)
    panel = load_panel(cfg).close
    dates_b = list(ds.dates[ds.idx["test_b"]])
    base = BacktestParams.from_config(cfg)
    workers = min(len(cfg.seeds), cfg.workers.backtest)

    runs = _runs(cfg, forecasts, baselines, base)
    short_results = cache.run_backtests(
        [(panel, dates_b, params, fc) for _label, fc, params in runs], workers
    )
    full_results = cache.run_backtests(
        [(panel, dates_b, params.replace(short_circuit=False), fc) for _label, fc, params in runs],
        workers,
    )

    rows = [
        _compare(label, short, full)
        for (label, _fc, _params), short, full in zip(runs, short_results, full_results, strict=True)
    ]
    results = {
        "primary": rows[0],
        "all_runs": rows,
        "n_runs": len(rows),
        "all_agree": bool(all(row["n_disagree"] == 0 for row in rows)),
    }
    write_json(out_path, artifact_metadata(cfg, _DEPENDS_ON), results)
    return results
