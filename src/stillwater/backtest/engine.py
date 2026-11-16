"""Execution loop and portfolio accounting (JOURNAL Section 2.9)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from stillwater.config import BacktestParams
from stillwater.optimization.covariance import ledoit_wolf_covariance
from stillwater.optimization.kkt import KKTLog, evaluate_kkt
from stillwater.optimization.qp import solve_qp

__all__ = ["BacktestResult", "compute_summary", "run_backtest", "slice_panel_a"]


class BacktestResult(TypedDict):
    """Output of :func:`run_backtest`; consumed by ``metrics``, ``tables`` and ``figures``."""

    net_cagr: float
    sharpe: float
    mdd: float
    turnover: float
    psi_kappa: float
    kkt_agreement_rate: float
    cum_wealth: np.ndarray
    daily_returns: np.ndarray
    executed_turnovers: list[float]
    kkt_logs: list[KKTLog]
    weights_history: list[np.ndarray]


def compute_summary(
    daily_returns: np.ndarray,
    rf_annual: float = 0.06,
) -> tuple[float, float, float, np.ndarray]:
    """Annualised net CAGR, excess Sharpe, maximum drawdown and the wealth path.

    Parameters
    ----------
    daily_returns
        Net daily portfolio returns, costs already deducted.
    rf_annual
        Constant annual risk-free rate used for the excess Sharpe ratio.

    Returns
    -------
    tuple
        ``(net_cagr, sharpe, mdd, cum_wealth)``; all zeros for an empty series.
    """
    n_days = len(daily_returns)
    if n_days == 0:
        return 0.0, 0.0, 0.0, np.array([], dtype=float)

    cum_wealth = np.cumprod(1.0 + daily_returns)
    net_cagr = float(cum_wealth[-1] ** (252.0 / n_days) - 1.0)

    excess = daily_returns - rf_annual / 252.0
    sd = float(np.std(excess, ddof=1)) if n_days > 1 else 0.0
    sharpe = float((np.mean(excess) / sd) * np.sqrt(252.0)) if sd > 1e-8 else 0.0

    peaks = np.maximum.accumulate(cum_wealth)
    mdd = float(np.abs(np.min((cum_wealth - peaks) / peaks)))
    return net_cagr, sharpe, mdd, cum_wealth


def slice_panel_a(
    result: BacktestResult,
    n_cycles: int,
    rebalance_freq: int,
    rf_annual: float = 0.06,
) -> dict[str, Any]:
    """Slice Panel A out of a continuous Panel B run.

    Panel A is the prefix of Panel B, so no separate simulation is needed. The
    previous signature mixed two units -- a decision-date count used to slice a
    daily-return array -- which left Panel A's returns three trading days short of
    its own turnover window. Slicing by cycle count keeps the two aligned.

    Parameters
    ----------
    result
        Output of :func:`run_backtest`.
    n_cycles
        Number of rebalance cycles in Panel A.
    rebalance_freq
        Trading days per cycle, used to convert cycles into daily observations.
    rf_annual
        Risk-free rate forwarded to :func:`compute_summary`.

    Returns
    -------
    dict
        ``net_cagr``, ``sharpe``, ``mdd``, ``turnover``, ``psi_kappa`` and
        ``daily_returns`` for the Panel A window.
    """
    rets_a = result["daily_returns"][: n_cycles * rebalance_freq]
    cagr_a, sharpe_a, mdd_a, _ = compute_summary(rets_a, rf_annual=rf_annual)
    tos_a = result["executed_turnovers"][:n_cycles]
    valid_kkt = [k for k in result["kkt_logs"][:n_cycles] if not np.isnan(k["delta_g"])]
    return {
        "net_cagr": cagr_a,
        "sharpe": sharpe_a,
        "mdd": mdd_a,
        "turnover": float(np.mean(tos_a)) if tos_a else 0.0,
        "psi_kappa": float(np.mean([k["kkt_should_trade"] for k in valid_kkt])) if valid_kkt else 0.0,
        "kkt_agreement_rate": (
            float(np.mean([k["kkt_should_trade"] == k["solver_traded"] for k in valid_kkt]))
            if valid_kkt
            else 0.0
        ),
        "daily_returns": rets_a,
    }


def _drift(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Propagate weights through one day of returns (Eq. 8)."""
    gross = 1.0 + float(np.dot(weights, returns))
    if gross <= 1e-8:
        return weights.copy()
    return (weights * (1.0 + returns)) / gross


def run_backtest(
    prices: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    params: BacktestParams,
    forecasts: np.ndarray | None = None,
) -> BacktestResult:
    """Simulate one continuous portfolio run with a full KKT decision log.

    Walks the decision dates in order. At each rebalance date ``t`` the optimizer
    sees the holdings drifted to that close, evaluates Theorem 1, and either holds
    (short-circuit) or solves Eq. (4)-(5). The chosen target executes at the close of
    ``t + 1`` against the holdings drifted through that day, so the cost of Eq. (9)
    is charged on the weights actually held and is booked on the first holding day
    ``t + 2``. Between rebalances, weights drift by Eq. (8).

    Parameters
    ----------
    prices
        Adjusted close panel indexed by trading date; columns identify the assets.
    dates
        Decision dates in ascending order. Every ``params.rebalance_freq``-th date
        is a rebalance; the intervening dates are holding days.
    params
        Frozen backtest configuration. Its digest is part of the cache key.
    forecasts
        ``(len(dates), n_assets)`` forecast array, or ``None`` for equal weight.

    Returns
    -------
    BacktestResult
        Daily net returns, per-cycle executed turnover, the KKT log, and summary
        statistics.

    Raises
    ------
    KeyError
        If a decision date is absent from ``prices.index``.
    RuntimeError
        If the QP solver fails on a rebalance date.

    Notes
    -----
    The KKT log is the single source of truth for Table 3 and Figure 1. A hold date
    sends no order: ``C_{t+1} = 0`` and holdings keep drifting through the holding
    window. A cap-drift date lies outside Theorem 1's assumptions and always solves.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a DataFrame indexed by trading date")

    all_dates = list(prices.index)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_assets = prices.shape[1]
    p_mat = prices.values
    r_mat = np.zeros_like(p_mat)
    r_mat[1:] = (p_mat[1:] / p_mat[:-1]) - 1.0

    reb_indices = list(range(0, len(dates), params.rebalance_freq))
    missing = [d for d in dates if pd.Timestamp(d) not in date_to_idx]
    if missing:
        raise KeyError(f"{len(missing)} decision dates absent from prices.index, e.g. {missing[0]}")

    is_equal_weight = forecasts is None
    w_holding = np.full(n_assets, 1.0 / n_assets)
    kkt_logs: list[KKTLog] = []
    executed_turnovers: list[float] = []
    daily_port_returns: list[float] = []
    weights_history: list[np.ndarray] = []

    for cycle_i, r_idx in enumerate(reb_indices):
        t_global = date_to_idx[pd.Timestamp(dates[r_idx])]

        if is_equal_weight:
            w_opt = np.full(n_assets, 1.0 / n_assets)
            solver_traded = True
            solver_to = 0.0
        else:
            ret_window = r_mat[max(1, t_global - params.cov_lookback + 1) : t_global + 1]
            sigma_mat = ledoit_wolf_covariance(ret_window, scale_to_horizon=5.0)

            vol_scale = None
            if params.volatility_scaled:
                sig_t = np.std(ret_window, axis=0, ddof=1)
                mean_sig = float(np.mean(sig_t))
                vol_scale = sig_t / mean_sig if mean_sig > 1e-8 else np.ones(n_assets)
            eta_b_t, eta_s_t = params.penalty_vectors(n_assets, vol_scale)

            assert forecasts is not None
            y_hat = forecasts[r_idx]
            mu_raw = np.expm1(y_hat) if params.log_forecast else np.array(y_hat, dtype=float)
            mu_shrunk = (1.0 - params.alpha) * mu_raw + params.alpha * float(np.mean(mu_raw))

            w_minus = w_holding.copy()
            kkt = evaluate_kkt(
                -mu_shrunk + params.gamma * (sigma_mat @ w_minus),
                w_minus,
                eta_b_t,
                eta_s_t,
                w_max=params.w_max,
                eps=params.cap_drift_buffer,
                boundary_tol=params.boundary_tol,
            )

            if params.short_circuit and not (
                kkt["should_trade"] or kkt["is_boundary"] or kkt["is_cap_drift"]
            ):
                w_opt = w_minus.copy()
                solver_to = 0.0
                solver_traded = False
            else:
                w_opt, solver_to = solve_qp(
                    mu_shrunk,
                    sigma_mat,
                    w_minus,
                    params.gamma,
                    eta_b_t,
                    eta_s_t,
                    w_max=params.w_max,
                )
                solver_traded = bool(solver_to > params.turnover_tol)

            kkt_logs.append(
                {
                    "cycle": cycle_i,
                    "date": pd.Timestamp(dates[r_idx]).strftime("%Y-%m-%d"),
                    "delta_g": float(kkt["delta_g"]),
                    "threshold": float(kkt["threshold"]),
                    "margin": float(kkt["margin"]),
                    "kkt_should_trade": bool(kkt["should_trade"]),
                    "solver_traded": solver_traded,
                    "is_boundary": bool(kkt["is_boundary"]),
                    "is_cap_drift": bool(kkt["is_cap_drift"]),
                    "trigger_sell_idx": int(kkt["trigger_sell_idx"]),
                    "trigger_buy_idx": int(kkt["trigger_buy_idx"]),
                }
            )

        weights_history.append(w_opt.copy())

        t1 = t_global + 1
        if t1 < len(all_dates):
            r_t1 = r_mat[t1]
            daily_port_returns.append(float(np.dot(w_holding, r_t1)))
            w_t1_drifted = _drift(w_holding, r_t1)
        else:
            w_t1_drifted = w_holding.copy()

        if solver_traded:
            diff = w_opt - w_t1_drifted
            cost_t1 = float(
                params.c_b * np.sum(np.maximum(diff, 0.0))
                + params.c_s * np.sum(np.maximum(-diff, 0.0))
            )
            exec_turnover = float(np.sum(np.abs(diff)))
            w_holding = w_opt.copy()
        else:
            cost_t1 = 0.0
            exec_turnover = 0.0
            w_holding = w_t1_drifted.copy()

        executed_turnovers.append(exec_turnover)

        next_t = (
            date_to_idx[pd.Timestamp(dates[reb_indices[cycle_i + 1]])]
            if cycle_i + 1 < len(reb_indices)
            else min(len(all_dates) - 1, t_global + params.rebalance_freq)
        )
        for day in range(t_global + 2, next_t + 1):
            if day >= len(all_dates):
                break
            gross = float(np.dot(w_holding, r_mat[day]))
            daily_port_returns.append(gross - cost_t1 if day == t_global + 2 else gross)
            w_holding = _drift(w_holding, r_mat[day])

    daily_rets = np.array(daily_port_returns, dtype=float)
    cagr, sharpe, mdd, cum_wealth = compute_summary(daily_rets, rf_annual=params.rf_annual)
    valid_kkt = [k for k in kkt_logs if not np.isnan(k["delta_g"])]

    return {
        "net_cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "turnover": float(np.mean(executed_turnovers)) if executed_turnovers else 0.0,
        "psi_kappa": float(np.mean([k["kkt_should_trade"] for k in valid_kkt])) if valid_kkt else 0.0,
        "kkt_agreement_rate": (
            float(np.mean([k["kkt_should_trade"] == k["solver_traded"] for k in valid_kkt]))
            if valid_kkt
            else 0.0
        ),
        "cum_wealth": cum_wealth,
        "daily_returns": daily_rets,
        "executed_turnovers": executed_turnovers,
        "kkt_logs": kkt_logs,
        "weights_history": weights_history,
    }
