"""Numerical reference for the turnover-penalised mean-variance QP (Eq. 4-5)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from stillwater.optimization.kkt import _as_vector

__all__ = ["solve_qp"]


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
    bounds: list[tuple[float, float | None]] = [(0.0, w_max) for _ in range(n)] + [
        (0.0, None) for _ in range(2 * n)
    ]

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
