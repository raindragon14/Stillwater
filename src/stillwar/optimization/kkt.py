"""Multiplier-free KKT no-trade test (Theorem 1)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypedDict

import numpy as np

__all__ = ["KKTLog", "evaluate_kkt", "tabulate_kkt_agreement"]


class KKTLog(TypedDict):
    """One row of the pre-trade diagnostic log, one per rebalance date."""

    cycle: int
    date: str
    delta_g: float
    threshold: float
    margin: float
    kkt_should_trade: bool
    solver_traded: bool
    is_boundary: bool
    is_cap_drift: bool
    trigger_sell_idx: int
    trigger_buy_idx: int


def _as_vector(value: float | np.ndarray, n_assets: int) -> np.ndarray:
    """Broadcast a scalar penalty to a length-``n_assets`` vector, or validate a vector."""
    if np.isscalar(value):
        return np.full(n_assets, float(value), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n_assets,):
        raise ValueError(f"penalty vector must have shape ({n_assets},), got {arr.shape}")
    return arr


def evaluate_kkt(
    g: np.ndarray,
    w_minus: np.ndarray,
    eta_b: float | np.ndarray,
    eta_s: float | np.ndarray,
    w_max: float = 0.15,
    eps: float = 0.0,
    boundary_tol: float = 1e-6,
) -> dict[str, Any]:
    """Evaluate Theorem 1: decide whether holding the incumbent weights is optimal.

    The test is the multiplier-free condition of Eq. (6). It compares the largest
    gradient among assets that can be sold against the smallest among assets that
    can be bought, and declares a hold when the spread does not exceed the round-trip
    penalty band. The budget multiplier cancels because both sides of the pair trade
    move the same amount of weight (Appendix A.2).

    Parameters
    ----------
    g
        Gradient of the smooth objective at ``w_minus``, ``-mu + gamma * Sigma @ w_minus``.
    w_minus
        Incumbent weights, the holdings drifted to the decision close.
    eta_b, eta_s
        Buy and sell penalties. Scalars give the uniform band; arrays of shape
        ``(n_assets,)`` give the volatility-scaled generalisation of Section 3.7.
    w_max
        Box cap.
    eps
        Cap-drift buffer: pre-trade drift up to ``w_max + eps`` is still routed
        through Theorem 1 (Section 3.7, extension i).
    boundary_tol
        Absolute margin below which the verdict is reported as a boundary case
        rather than a trade or a hold.

    Returns
    -------
    dict
        ``delta_g`` (the spread at the penalised-optimal pair), ``threshold``
        (``eta_b[i*] + eta_s[j*]``), ``margin`` (``delta_g - threshold``),
        ``should_trade``, ``is_boundary``, ``is_cap_drift``, the trigger indices and
        the raw extreme gradients.

    Raises
    ------
    ValueError
        If the sellable or buyable set is empty, which can only happen if the
        budget or box constraints are violated.

    Notes
    -----
    When ``w_minus`` breaches the cap the incumbent is infeasible for Eq. (5) and
    Theorem 1 does not apply; the function reports ``is_cap_drift`` and
    ``delta_g = nan`` so the caller can force a rebalance and exclude the date from
    the interior statistics.
    """
    n_assets = len(g)
    eta_b_arr = _as_vector(eta_b, n_assets)
    eta_s_arr = _as_vector(eta_s, n_assets)

    is_cap_drift = bool(np.any(w_minus > (w_max + eps) + 1e-7))
    if is_cap_drift:
        return {
            "delta_g": np.nan,
            "threshold": float(np.mean(eta_b_arr) + np.mean(eta_s_arr)),
            "margin": np.nan,
            "should_trade": True,
            "is_boundary": False,
            "is_cap_drift": True,
            "trigger_sell_idx": -1,
            "trigger_buy_idx": -1,
            "max_g_J": np.nan,
            "min_g_I": np.nan,
        }

    sellable_idx = np.where(w_minus > 1e-8)[0]
    buyable_idx = np.where(w_minus < w_max - 1e-8)[0]
    if len(sellable_idx) == 0 or len(buyable_idx) == 0:
        raise ValueError("Budget and box constraints violated: empty J or I sets.")

    sell_scores = g[sellable_idx] - eta_s_arr[sellable_idx]
    buy_scores = g[buyable_idx] + eta_b_arr[buyable_idx]
    margin_mat = sell_scores[:, None] - buy_scores[None, :]
    margin_mat[sellable_idx[:, None] == buyable_idx[None, :]] = -np.inf

    opt_j, opt_i = np.unravel_index(np.argmax(margin_mat), margin_mat.shape)
    j_star = int(sellable_idx[opt_j])
    i_star = int(buyable_idx[opt_i])

    margin = float(margin_mat[opt_j, opt_i])
    return {
        "delta_g": float(g[j_star] - g[i_star]),
        "threshold": float(eta_b_arr[i_star] + eta_s_arr[j_star]),
        "margin": margin,
        "should_trade": bool(margin > 0.0),
        "is_boundary": bool(abs(margin) < boundary_tol),
        "is_cap_drift": False,
        "trigger_sell_idx": j_star,
        "trigger_buy_idx": i_star,
        "max_g_J": float(g[j_star]),
        "min_g_I": float(g[i_star]),
    }


def tabulate_kkt_agreement(kkt_logs: Sequence[KKTLog]) -> dict[str, int]:
    """Cross-tabulate Theorem 1 verdicts against the solver's trade decision (Table 3).

    Boundary cases and cap-drift dates are counted separately because Theorem 1's
    assumptions do not cover them: a boundary case depends on solver tolerance, and
    a cap-drift date has an infeasible incumbent.

    Parameters
    ----------
    kkt_logs
        The ``kkt_logs`` field of a :class:`BacktestResult`.

    Returns
    -------
    dict
        Counts for each verdict/decision cell plus row and column totals.
    """
    counts = {k: 0 for k in _AGREEMENT_KEYS}

    for row in kkt_logs:
        suffix: Literal["traded", "held"] = "traded" if row["solver_traded"] else "held"
        if row["is_cap_drift"]:
            counts[f"cap_drift_{suffix}"] += 1
        elif row["is_boundary"]:
            counts[f"boundary_{suffix}"] += 1
        elif row["kkt_should_trade"]:
            counts[f"trade_{suffix}"] += 1
        else:
            counts[f"hold_{suffix}"] += 1

    t_tr, t_he = counts["trade_traded"], counts["trade_held"]
    h_tr, h_he = counts["hold_traded"], counts["hold_held"]
    b_tr, b_he = counts["boundary_traded"], counts["boundary_held"]
    c_tr, c_he = counts["cap_drift_traded"], counts["cap_drift_held"]

    return {
        **counts,
        "trade_total": t_tr + t_he,
        "hold_total": h_tr + h_he,
        "boundary_count": b_tr + b_he,
        "cap_drift_count": c_tr + c_he,
        "total_solver_traded": t_tr + h_tr + b_tr + c_tr,
        "total_solver_held": t_he + h_he + b_he + c_he,
        "total_dates": len(kkt_logs),
    }


_AGREEMENT_KEYS = (
    "trade_traded",
    "trade_held",
    "hold_traded",
    "hold_held",
    "boundary_traded",
    "boundary_held",
    "cap_drift_traded",
    "cap_drift_held",
)
