"""Produce Tables 1-3, Table S1 and the Phase-4 diagnostics as raw numbers.

This module replaces the computation half of the old ``reporting.py``. It writes JSON
containing floats only; :mod:`stillwater.reporting.tables` formats them and :mod:`stillwater.reporting.figures`
draws them. Keeping the three apart is what makes a manuscript table verifiable against
its artifact.

Every backtest goes through :func:`stillwater.provenance.cache.cached_backtest`, so the primary
LogSig configuration is computed once even though Table 1, Table 3, the Phase-4
diagnostics and both figures all need it.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stillwater.backtest.engine import slice_panel_a
from stillwater.backtest.metrics import (
    backtest_metrics,
    declined_trade_test,
    rank_ic_summary,
    seed_summary,
    trigger_concentration,
)
from stillwater.backtest.stats import deflated_sharpe, newey_west_alpha
from stillwater.config import BacktestParams, Config
from stillwater.data import load_panel
from stillwater.experiments.trials import load_trials
from stillwater.features import load_dataset
from stillwater.optimization.kkt import tabulate_kkt_agreement
from stillwater.provenance import cache
from stillwater.provenance.metadata import artifact_metadata, write_json

__all__ = ["run_all", "run_phase4", "run_table1", "run_table2", "run_table3"]

_DEPENDS_ON = ("config", "optimization", "backtest", "provenance", "features", "data")

LEARNED_MODELS: tuple[tuple[str, str], ...] = (
    ("LogSig", "logsig"),
    ("LogSig-Inc", "logsig_inc"),
    ("MLP", "mlp"),
    ("LSTM", "lstm"),
)


def _load_trials_or_none() -> np.ndarray | None:
    """Trial Sharpe ratios, or ``None`` when ``make trials`` has not been run yet."""
    try:
        return load_trials()
    except FileNotFoundError:
        return None


def _summarise(per_seed: Sequence[dict[str, float]], key: str) -> dict[str, float]:
    """Mean and standard deviation of one metric across seeds."""
    return seed_summary([m[key] for m in per_seed])


def _panel_block(
    per_seed: Sequence[dict[str, float]],
    rank_ic: dict[str, Any] | None,
    nw_alpha: dict[str, float] | None,
    dsr: float | None,
) -> dict[str, Any]:
    """One panel of one Table 1 row."""
    return {
        "cagr": _summarise(per_seed, "cagr"),
        "sharpe": _summarise(per_seed, "sharpe"),
        "mdd": _summarise(per_seed, "mdd"),
        "turnover": _summarise(per_seed, "turnover"),
        "rank_ic": rank_ic,
        "nw_alpha": nw_alpha,
        "dsr": dsr,
    }


def _deflated(
    per_seed: Sequence[dict[str, float]],
    trials: np.ndarray | None,
    returns: np.ndarray,
) -> float | None:
    """Deflated Sharpe ratio of the seed-averaged strategy, or ``None`` without trials."""
    if trials is None:
        return None
    return deflated_sharpe(float(np.mean([m["sharpe"] for m in per_seed])), trials, returns)


def run_table1(
    cfg: Config,
    forecasts: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    out_path: str | Path = "data/processed/table1.json",
) -> dict[str, Any]:
    """Build Table 1: out-of-sample performance for every strategy, Panels A and B.

    Parameters
    ----------
    cfg
        Configuration.
    forecasts
        ``{model_key: (n_seeds, n_dates, n_assets)}`` for the four learned models.
    baselines
        ``{"hist_mv": (n_dates, n_assets), "gmv": (n_dates, n_assets)}``.
    out_path
        Destination JSON.

    Returns
    -------
    dict
        ``{strategy: {"panel_a": {...}, "panel_b": {...}}}`` with floats only.
    """
    ds = load_dataset(cfg)
    panel = load_panel(cfg).close
    dates_b = list(ds.dates[ds.idx["test_b"]])
    n_cycles_a = cfg.reporting.panel_a_rebalances
    base = BacktestParams.from_config(cfg)
    workers = min(len(cfg.seeds), cfg.workers.backtest)
    trials = _load_trials_or_none()

    ew_b = cache.cached_backtest(panel, dates_b, base, None)
    ew_a = slice_panel_a(ew_b, n_cycles_a, base.rebalance_freq)
    ew_rets_b, ew_rets_a = ew_b["daily_returns"], ew_a["daily_returns"]
    y_b = ds.y[ds.idx["test_b"]]
    y_a = ds.y[ds.idx["test_a"]]
    n_dates_a = len(ds.idx["test_a"])

    learned_tasks = [
        (panel, dates_b, base, forecasts[key][seed])
        for _, key in LEARNED_MODELS
        for seed in range(len(cfg.seeds))
    ]
    learned_results = cache.run_backtests(learned_tasks, workers)

    table: dict[str, Any] = {}
    cursor = 0
    for name, key in LEARNED_MODELS:
        fc = forecasts[key]
        chunk = learned_results[cursor : cursor + len(cfg.seeds)]
        cursor += len(cfg.seeds)

        seeds_b = [backtest_metrics(r) for r in chunk]
        seeds_a = [backtest_metrics(slice_panel_a(r, n_cycles_a, base.rebalance_freq)) for r in chunk]
        rets_a = np.mean(
            [slice_panel_a(r, n_cycles_a, base.rebalance_freq)["daily_returns"] for r in chunk], axis=0
        )
        rets_b = np.mean([r["daily_returns"] for r in chunk], axis=0)

        rank_ic_b = {
            "weekly": seed_summary(
                [
                    rank_ic_summary(
                        fc[s], y_b, base.rebalance_freq, cfg.reporting.effective_ratio_weekly
                    )["mean"]
                    for s in range(len(cfg.seeds))
                ]
            ),
            "ensemble_weekly": rank_ic_summary(
                fc.mean(axis=0), y_b, base.rebalance_freq, cfg.reporting.effective_ratio_weekly
            ),
            "ensemble_daily": rank_ic_summary(
                fc.mean(axis=0), y_b, 1, cfg.reporting.effective_ratio_daily
            ),
        }
        rank_ic_a = {
            "weekly": seed_summary(
                [
                    rank_ic_summary(
                        fc[s][:n_dates_a],
                        y_a,
                        base.rebalance_freq,
                        cfg.reporting.effective_ratio_weekly,
                    )["mean"]
                    for s in range(len(cfg.seeds))
                ]
            )
        }
        table[name] = {
            "panel_a": _panel_block(
                seeds_a,
                rank_ic_a,
                newey_west_alpha(rets_a, ew_rets_a[: len(rets_a)]),
                _deflated(seeds_a, trials, rets_a),
            ),
            "panel_b": _panel_block(
                seeds_b,
                rank_ic_b,
                newey_west_alpha(rets_b, ew_rets_b[: len(rets_b)]),
                _deflated(seeds_b, trials, rets_b),
            ),
        }

    classical = {
        "Hist-MV, penalized": (baselines["hist_mv"], base),
        "Hist-MV, unpenalized": (baselines["hist_mv"], base.replace(kappa=0.0)),
        "GMV": (baselines["gmv"], base),
        "GMV, unpenalized (supplementary)": (baselines["gmv"], base.replace(kappa=0.0)),
    }
    classical_results = cache.run_backtests(
        [(panel, dates_b, params, fc) for fc, params in classical.values()], workers
    )

    for (name, (fc, _params)), result in zip(classical.items(), classical_results, strict=True):
        sl_a = slice_panel_a(result, n_cycles_a, base.rebalance_freq)
        has_signal = bool(np.any(fc != 0))

        def _classical_ic(
            forecast: np.ndarray,
            target: np.ndarray,
            *,
            signal: bool = has_signal,
        ) -> dict[str, Any] | None:
            if not signal:
                return None
            return {
                "weekly": seed_summary(
                    [
                        rank_ic_summary(
                            forecast, target, base.rebalance_freq, cfg.reporting.effective_ratio_weekly
                        )["mean"]
                    ]
                )
            }

        seeds_b = [backtest_metrics(result)]
        seeds_a = [backtest_metrics(sl_a)]
        table[name] = {
            "panel_a": _panel_block(
                seeds_a,
                _classical_ic(fc[:n_dates_a], y_a),
                newey_west_alpha(sl_a["daily_returns"], ew_rets_a[: len(sl_a["daily_returns"])]),
                _deflated(seeds_a, trials, sl_a["daily_returns"]),
            ),
            "panel_b": _panel_block(
                seeds_b,
                _classical_ic(fc, y_b),
                newey_west_alpha(result["daily_returns"], ew_rets_b[: len(result["daily_returns"])]),
                _deflated(seeds_b, trials, result["daily_returns"]),
            ),
        }

    seeds_b = [backtest_metrics(ew_b)]
    seeds_a = [backtest_metrics(ew_a)]
    table["EW"] = {
        "panel_a": _panel_block(seeds_a, None, None, _deflated(seeds_a, trials, ew_rets_a)),
        "panel_b": _panel_block(seeds_b, None, None, _deflated(seeds_b, trials, ew_rets_b)),
    }

    write_json(out_path, artifact_metadata(cfg, _DEPENDS_ON), table)
    return table


def run_table2(
    cfg: Config,
    forecasts: np.ndarray,
    baselines: dict[str, np.ndarray],
    out_path: str | Path = "data/processed/table2.json",
) -> list[dict[str, Any]]:
    """Build Table 2: one-at-a-time sensitivity of LogSig to ``kappa``, ``gamma`` and cost.

    Also computes the risk-only activation rate ``psi_kappa^0`` from the GMV benchmark
    at each penalty scale, which is the forecast-collapse indicator of Section 2.7.

    Parameters
    ----------
    cfg
        Configuration.
    forecasts
        ``(n_seeds, n_dates, n_assets)`` LogSig forecasts.
    baselines
        ``{"gmv": (n_dates, n_assets)}``.
    out_path
        Destination JSON.

    Returns
    -------
    list of dict
        One row per setting, with Panel A metrics and both activation rates.
    """
    ds = load_dataset(cfg)
    panel = load_panel(cfg).close
    dates_b = list(ds.dates[ds.idx["test_b"]])
    n_cycles_a = cfg.reporting.panel_a_rebalances
    base = BacktestParams.from_config(cfg)
    workers = min(len(cfg.seeds), cfg.workers.backtest)

    settings: list[tuple[str, BacktestParams, bool]] = [
        ("kappa_0", base.replace(kappa=0.0), True),
        ("kappa_1", base.replace(kappa=1.0), True),
        ("kappa_2.5", base.replace(kappa=2.5), True),
        ("kappa_5", base, True),
        ("kappa_10", base.replace(kappa=10.0), True),
        ("gamma_1", base.replace(gamma=1.0), False),
        ("gamma_5", base.replace(gamma=5.0), False),
        ("gamma_10", base.replace(gamma=10.0), False),
        ("m_0.5", base.replace(c_b=base.c_b * 0.5, c_s=base.c_s * 0.5), False),
        ("m_2", base.replace(c_b=base.c_b * 2.0, c_s=base.c_s * 2.0), False),
        ("non_inclusive", base.replace(c_b=0.001933, c_s=0.002933), False),
    ]

    psi0_settings = [(label, params) for label, params, with_psi0 in settings if with_psi0]
    psi0_results = cache.run_backtests(
        [(panel, dates_b, params, baselines["gmv"]) for _, params in psi0_settings], workers
    )
    psi0 = {
        label: slice_panel_a(result, n_cycles_a, base.rebalance_freq)["psi_kappa"]
        for (label, _), result in zip(psi0_settings, psi0_results, strict=True)
    }

    tasks = [
        (panel, dates_b, params, forecasts[seed])
        for _, params, _ in settings
        for seed in range(len(cfg.seeds))
    ]
    results = cache.run_backtests(tasks, workers)

    n_seeds = len(cfg.seeds)
    rows = []
    for idx, (label, params, with_psi0) in enumerate(settings):
        chunk = results[idx * n_seeds : (idx + 1) * n_seeds]
        sl = [slice_panel_a(r, n_cycles_a, base.rebalance_freq) for r in chunk]
        rows.append(
            {
                "label": label,
                "kappa": params.kappa,
                "gamma": params.gamma,
                "c_b": params.c_b,
                "c_s": params.c_s,
                "cagr": float(np.mean([s["net_cagr"] for s in sl])),
                "sharpe": float(np.mean([s["sharpe"] for s in sl])),
                "mdd": float(np.mean([s["mdd"] for s in sl])),
                "turnover": float(np.mean([s["turnover"] for s in sl])),
                "psi_kappa": float(np.mean([s["psi_kappa"] for s in sl])),
                "psi_kappa_0": psi0.get(label) if with_psi0 else None,
            }
        )

    write_json(out_path, artifact_metadata(cfg, _DEPENDS_ON), rows)
    return rows


def run_table3(
    cfg: Config,
    forecasts: np.ndarray,
    out_path: str | Path = "data/processed/table3.json",
) -> dict[str, int]:
    """Build Table 3: Theorem 1 verdicts cross-tabulated against the solver.

    Parameters
    ----------
    cfg
        Configuration.
    forecasts
        ``(n_seeds, n_dates, n_assets)`` LogSig forecasts; the primary seed is used.
    out_path
        Destination JSON.

    Returns
    -------
    dict
        Counts from :func:`stillwater.optimization.kkt.tabulate_kkt_agreement`.
    """
    ds = load_dataset(cfg)
    panel = load_panel(cfg).close
    dates_b = list(ds.dates[ds.idx["test_b"]])
    result = cache.cached_backtest(panel, dates_b, BacktestParams.from_config(cfg), forecasts[0])
    counts = tabulate_kkt_agreement(result["kkt_logs"])
    write_json(out_path, artifact_metadata(cfg, _DEPENDS_ON), counts)
    write_json(
        Path(out_path).with_name("kkt_log.json"),
        artifact_metadata(cfg, _DEPENDS_ON),
        result["kkt_logs"],
    )
    return counts


def run_phase4(
    cfg: Config,
    forecasts: np.ndarray,
    out_path: str | Path = "data/processed/phase4.json",
) -> dict[str, Any]:
    """Price the declined rebalances and measure trigger concentration (Section 3.3).

    Parameters
    ----------
    cfg
        Configuration.
    forecasts
        ``(n_seeds, n_dates, n_assets)`` LogSig forecasts; the primary seed is used.
    out_path
        Destination JSON.

    Returns
    -------
    dict
        Declined-trade statistics, the trigger Herfindahl index, and the decision-regime
        counts that Figure 2 annotates.
    """
    ds = load_dataset(cfg)
    loaded = load_panel(cfg)
    panel = loaded.close
    dates_b = list(ds.dates[ds.idx["test_b"]])
    base = BacktestParams.from_config(cfg)

    penalized = cache.cached_backtest(panel, dates_b, base, forecasts[0])
    unpenalized = cache.cached_backtest(panel, dates_b, base.replace(kappa=0.0), forecasts[0])

    logs = penalized["kkt_logs"]
    results = {
        "declined": declined_trade_test(penalized, unpenalized, base.rebalance_freq),
        "triggers": trigger_concentration(logs, loaded.tickers),
        "cap_drift_weeks": float(sum(1 for k in logs if k["is_cap_drift"])),
        "interior_weeks": float(sum(1 for k in logs if not k["is_cap_drift"])),
        "interior_trades": float(
            sum(1 for k in logs if k["kkt_should_trade"] and not k["is_cap_drift"])
        ),
        "interior_holds": float(
            sum(1 for k in logs if not k["kkt_should_trade"] and not k["is_cap_drift"])
        ),
        "boundary_cases": float(sum(1 for k in logs if k["is_boundary"])),
        "total_dates": float(len(logs)),
        "no_trade_band": base.no_trade_band,
    }
    write_json(out_path, artifact_metadata(cfg, _DEPENDS_ON), results)
    return results


def run_all(
    cfg: Config,
    forecasts: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    out_dir: str | Path = "data/processed",
) -> dict[str, Any]:
    """Produce every table artifact in one pass.

    Parameters
    ----------
    cfg
        Configuration.
    forecasts
        ``{model_key: (n_seeds, n_dates, n_assets)}`` for the four learned models.
    baselines
        ``{"hist_mv": ..., "gmv": ...}``.
    out_dir
        Directory for the JSON artifacts.

    Returns
    -------
    dict
        ``{"table1": ..., "table2": ..., "table3": ..., "phase4": ...}``.
    """
    out = Path(out_dir)
    return {
        "table1": run_table1(cfg, forecasts, baselines, out / "table1.json"),
        "table2": run_table2(cfg, forecasts["logsig"], baselines, out / "table2.json"),
        "table3": run_table3(cfg, forecasts["logsig"], out / "table3.json"),
        "phase4": run_phase4(cfg, forecasts["logsig"], out / "phase4.json"),
    }
