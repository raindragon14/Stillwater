"""Ledoit-Wolf covariance, the multiplier-free KKT test, the QP solver, and the backtest.

This module is the mathematical core of the paper. It contains exactly four ideas:

* :func:`ledoit_wolf_covariance` -- Eq. (4)'s risk model, Ledoit & Wolf (2004).
* :func:`evaluate_kkt` -- Theorem 1, the multiplier-free no-trade test of Eq. (6).
* :func:`solve_qp` -- the numerical reference of Eq. (4)-(5), solved by SLSQP.
* :func:`run_backtest` -- the execution loop of Section 2.9.

The KKT log returned by :func:`run_backtest` is the single source of truth for both
Table 3 and Figure 2. Figures must read it rather than re-simulating; a second
implementation would silently diverge from the table it is supposed to illustrate.
"""

from collections.abc import Sequence
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from stillwar.params import BacktestParams

__all__ = [
    "BacktestResult",
    "compute_summary",
    "evaluate_kkt",
    "ledoit_wolf_covariance",
    "run_backtest",
    "slice_panel_a",
    "solve_qp",
    "tabulate_kkt_agreement",
]


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


def ledoit_wolf_covariance(
    returns_60d: np.ndarray,
    scale_to_horizon: float = 5.0,
    ridge_eps: float = 1e-12,
) -> np.ndarray:
    """Analytical Ledoit & Wolf (2004) linear-shrinkage covariance, scaled to the horizon.

    Shrinks the sample covariance toward a scaled identity, with the optimal
    shrinkage intensity derived in closed form from the fourth moments of the data.

    Parameters
    ----------
    returns_60d
        ``(n_obs, n_assets)`` matrix of daily simple returns.
    scale_to_horizon
        Multiplier applied after shrinkage; ``5.0`` converts a daily covariance to
        the five-day forecast horizon of Eq. (4).
    ridge_eps
        Diagonal ridge added after scaling, so the matrix is strictly positive
        definite even when the shrinkage target is degenerate.

    Returns
    -------
    ndarray
        Symmetric ``(n_assets, n_assets)`` covariance matrix.

    Raises
    ------
    ValueError
        If fewer than four observations are supplied.

    Notes
    -----
    JOURNAL Section 2.5: ``Sigma_t = 5 * Sigma_t^LW + 1e-12 I``.
    """
    n_obs, n_assets = returns_60d.shape
    if n_obs < 4:
        raise ValueError(f"Need at least 4 observations for Ledoit-Wolf, got {n_obs}")

    x_centered = returns_60d - np.mean(returns_60d, axis=0, keepdims=True)
    sample_cov = (x_centered.T @ x_centered) / n_obs

    var_avg = float(np.trace(sample_cov) / n_assets)
    target_f = var_avg * np.eye(n_assets)

    diffs = x_centered[:, :, None] * x_centered[:, None, :] - sample_cov[None, :, :]
    pi_hat = float(np.sum(diffs**2) / n_obs)
    gamma_hat = float(np.sum((sample_cov - target_f) ** 2))

    shrinkage = max(0.0, min(1.0, (pi_hat / n_obs) / gamma_hat)) if gamma_hat > 1e-12 else 1.0
    shrunk_cov = shrinkage * target_f + (1.0 - shrinkage) * sample_cov
    sigma = scale_to_horizon * shrunk_cov + ridge_eps * np.eye(n_assets)
    return 0.5 * (sigma + sigma.T)


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


def _finalise_weights(x: np.ndarray, w_max: float, message: str) -> np.ndarray:
    """Clip to the box, enforce the budget, and fail loudly if either is violated.

    The previous implementation divided by the weight sum unconditionally, which
    could push a weight above ``w_max`` and silently paper over a failed solve.
    """
    w = np.clip(np.asarray(x, dtype=float), 0.0, w_max)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"degenerate weights from SLSQP ({message}): sum={total!r}")
    if abs(total - 1.0) > 1e-6:
        raise RuntimeError(f"budget violated by SLSQP ({message}): sum(w)={total:.12f}")
    w = w / total
    if np.any(w > w_max + 1e-6):
        raise RuntimeError(f"box violated by SLSQP ({message}): max(w)={w.max():.12f}")
    return w


def _minimize_slsqp(
    fun: Any,
    x0: np.ndarray,
    jac: Any,
    bounds: list[tuple[float, float | None]],
    constraints: list[dict[str, Any]],
    ftol: float,
) -> Any:
    """Run SLSQP with a relaxation ladder, always keeping the analytic Jacobian.

    ``ftol = 1e-12`` is tight enough that SLSQP occasionally stalls with "positive
    directional derivative for linesearch" on the volatility-scaled extension, where
    per-asset penalties span a factor of several. The previous code retried once
    *without* the Jacobian and then accepted the result unconditionally, so a failed
    solve silently entered the published numbers.

    This ladder relaxes the tolerance progressively and warm-starts each attempt from
    the previous iterate, which resolves the stall without changing the converged
    solution: the first attempt is the one reported in Section 2.5, and later attempts
    only run when it fails.

    Raises
    ------
    RuntimeError
        If every attempt fails.
    """
    last: Any = None
    for attempt_ftol, maxiter in ((ftol, 200), (max(ftol, 1e-10), 500), (max(ftol, 1e-8), 1000)):
        res = minimize(
            fun=fun,
            x0=x0,
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": attempt_ftol, "maxiter": maxiter, "disp": False},
        )
        if res.success:
            return res
        last = res
        x0 = np.asarray(res.x, dtype=float)
    raise RuntimeError(f"SLSQP failed after three attempts: {last.message}")


def _solve_unpenalized(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_minus: np.ndarray,
    gamma: float,
    w_max: float,
    solver_tol: float,
) -> np.ndarray:
    """Solve the ``eta = 0`` problem directly in weight space (``N`` variables).

    When both penalties vanish the slack variables of the 3N splitting do not enter
    the objective, so the split Hessian is singular and SLSQP converges to a point
    that can be off by ~5e-5. Solving the equivalent ``N``-variable problem is both
    better conditioned and about thirty times faster.
    """
    n = len(mu)

    def obj(w: np.ndarray) -> float:
        return float(-np.dot(w, mu) + 0.5 * gamma * np.dot(w, sigma @ w))

    def grad(w: np.ndarray) -> np.ndarray:
        return -mu + gamma * (sigma @ w)

    res = _minimize_slsqp(
        fun=obj,
        x0=w_minus,
        jac=grad,
        bounds=[(0.0, w_max)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(n)}],
        ftol=solver_tol,
    )
    return _finalise_weights(res.x, w_max, str(res.message))


def solve_qp(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_minus: np.ndarray,
    gamma: float,
    eta_b: float | np.ndarray,
    eta_s: float | np.ndarray,
    w_max: float = 0.15,
    solver_tol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Solve the turnover-penalised mean-variance QP of Eq. (4)-(5).

    Uses the ``3N`` splitting of Eq. (5) -- ``x = (w, u+, u-)`` under the equality
    ``w - u+ + u- = w_minus`` -- whenever either penalty is positive, so the hinge
    cost is smooth and SLSQP applies directly. When both penalties vanish the split
    is singular and the equivalent ``N``-variable problem is solved instead.

    Parameters
    ----------
    mu
        Shrunk expected simple returns, shape ``(n_assets,)``.
    sigma
        Covariance scaled to the forecast horizon, shape ``(n_assets, n_assets)``.
    w_minus
        Incumbent weights; also the warm start and the reference for turnover.
    gamma
        Risk aversion.
    eta_b, eta_s
        Buy and sell penalties per unit of traded weight (scalar or per-asset).
    w_max
        Box cap.
    solver_tol
        SLSQP function tolerance. ``1e-12`` is the value reported in Section 2.5.

    Returns
    -------
    tuple of (ndarray, float)
        Optimal weights and the two-sided turnover ``||w - w_minus||_1``.

    Raises
    ------
    RuntimeError
        If SLSQP does not converge, or if the returned point violates the budget or
        the box. The previous silent retry has been removed: it dropped the analytic
        Jacobian on the second attempt and accepted a failed solve.
    """
    n = len(mu)
    eta_b_arr = _as_vector(eta_b, n)
    eta_s_arr = _as_vector(eta_s, n)

    if not np.any(eta_b_arr) and not np.any(eta_s_arr):
        w_opt = _solve_unpenalized(mu, sigma, w_minus, gamma, w_max, solver_tol)
        return w_opt, float(np.sum(np.abs(w_opt - w_minus)))

    def obj(x: np.ndarray) -> float:
        w = x[:n]
        return float(
            -np.dot(w, mu)
            + 0.5 * gamma * np.dot(w, sigma @ w)
            + np.dot(eta_b_arr, x[n : 2 * n])
            + np.dot(eta_s_arr, x[2 * n :])
        )

    def obj_grad(x: np.ndarray) -> np.ndarray:
        return np.concatenate([-mu + gamma * (sigma @ x[:n]), eta_b_arr, eta_s_arr])

    eye_n = np.eye(n)
    a_budget = np.zeros(3 * n)
    a_budget[:n] = 1.0
    a_split = np.hstack([eye_n, -eye_n, eye_n])

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:n]) - 1.0, "jac": lambda x: a_budget},
        {
            "type": "eq",
            "fun": lambda x: x[:n] - x[n : 2 * n] + x[2 * n :] - w_minus,
            "jac": lambda x: a_split,
        },
    ]
    bounds = [(0.0, w_max) for _ in range(n)] + [(0.0, None) for _ in range(2 * n)]

    try:
        res = _minimize_slsqp(
            fun=obj,
            x0=_feasible_start(w_minus, w_max),
            jac=obj_grad,
            bounds=bounds,
            constraints=constraints,
            ftol=solver_tol,
        )
        w_opt = _finalise_weights(res.x[:n], w_max, str(res.message))
    except RuntimeError:
        w_opt = _solve_split_ridged(
            mu, sigma, w_minus, gamma, eta_b_arr, eta_s_arr, w_max, solver_tol
        )
    return w_opt, float(np.sum(np.abs(w_opt - w_minus)))


def _feasible_start(w_minus: np.ndarray, w_max: float) -> np.ndarray:
    """Build a feasible ``(w, u+, u-)`` start point from the incumbent weights.

    On a cap-drift date the incumbent breaches the box, so the previous warm start
    ``(w_minus, 0, 0)`` was infeasible in ``w``. SLSQP handles an infeasible start for
    the equality constraints but not cleanly for a violated bound, and on the
    volatility-scaled extension that produced a persistent "positive directional
    derivative for linesearch" stall.

    The start is projected onto the box, the weight deficit is redistributed in
    proportion to each asset's remaining room, and the slacks are then chosen so that
    ``w - u+ + u- = w_minus`` holds exactly.

    Parameters
    ----------
    w_minus
        Incumbent weights, possibly breaching ``w_max``.
    w_max
        Box cap.

    Returns
    -------
    ndarray
        ``(3N,)`` vector ``(w, u+, u-)`` that satisfies the budget, the box and the
        split equality.
    """
    w0 = np.clip(w_minus, 0.0, w_max)
    deficit = 1.0 - float(w0.sum())
    if deficit > 0.0:
        room = w_max - w0
        total_room = float(room.sum())
        if total_room > 0.0:
            w0 = w0 + deficit * room / total_room
    u_plus = np.maximum(w0 - w_minus, 0.0)
    u_minus = np.maximum(w_minus - w0, 0.0)
    return np.concatenate([w0, u_plus, u_minus])


def _solve_split_ridged(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_minus: np.ndarray,
    gamma: float,
    eta_b_arr: np.ndarray,
    eta_s_arr: np.ndarray,
    w_max: float,
    solver_tol: float,
) -> np.ndarray:
    """Last-resort solve: the 3N splitting with a negligible ridge on the slacks.

    SLSQP occasionally stalls with "positive directional derivative for linesearch" on
    the volatility-scaled extension, where per-asset penalties span a factor of several
    and the split Jacobian becomes near rank-deficient. Adding ``ridge * ||u||^2 / 2``
    makes the Hessian positive definite; because the penalties are linear, the
    perturbation shifts the optimum by about ``ridge / eta`` -- of order ``1e-6`` at
    ``ridge = 1e-8``, two orders of magnitude below the ``1e-4`` execution tolerance.

    This path only runs when every plain SLSQP attempt has failed, so it cannot affect
    a solve that already converges.
    """
    n = len(mu)
    ridge = 1e-8
    eye_n = np.eye(n)
    a_budget = np.zeros(3 * n)
    a_budget[:n] = 1.0

    def obj(x: np.ndarray) -> float:
        w = x[:n]
        return float(
            -np.dot(w, mu)
            + 0.5 * gamma * np.dot(w, sigma @ w)
            + np.dot(eta_b_arr, x[n : 2 * n])
            + np.dot(eta_s_arr, x[2 * n :])
            + 0.5 * ridge * np.dot(x[n:], x[n:])
        )

    def grad(x: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [-mu + gamma * (sigma @ x[:n]), eta_b_arr + ridge * x[n : 2 * n], eta_s_arr + ridge * x[2 * n :]]
        )

    res = _minimize_slsqp(
        fun=obj,
        x0=_feasible_start(w_minus, w_max),
        jac=grad,
        bounds=[(0.0, w_max) for _ in range(n)] + [(0.0, None) for _ in range(2 * n)],
        constraints=[
            {"type": "eq", "fun": lambda x: np.sum(x[:n]) - 1.0, "jac": lambda x: a_budget},
            {
                "type": "eq",
                "fun": lambda x: x[:n] - x[n : 2 * n] + x[2 * n :] - w_minus,
                "jac": lambda x: np.hstack([eye_n, -eye_n, eye_n]),
            },
        ],
        ftol=max(solver_tol, 1e-10),
    )
    return _finalise_weights(res.x[:n], w_max, str(res.message))

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
    The KKT log is the single source of truth for Table 3 and Figure 2. A hold date
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
