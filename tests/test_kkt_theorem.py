"""Theorem 1 equivalence against the numerical solver, and the QP formulation contract."""

import numpy as np
import pytest

from stillwar.portfolio import evaluate_kkt, solve_qp


def _psd(rng: np.random.Generator, n: int) -> np.ndarray:
    mat = rng.standard_normal((n, n))
    return mat.T @ mat / n + 0.1 * np.eye(n)


def test_kkt_no_trade_equivalence() -> None:
    """SLSQP holds inside the KKT band and trades strictly outside it."""
    n_assets, gamma = 10, 2.5
    eta_b, eta_s, w_max = 0.0075, 0.0125, 0.20
    band = eta_b + eta_s

    rng = np.random.default_rng(42)
    sigma = _psd(rng, n_assets)
    w_minus = np.full(n_assets, 1.0 / n_assets)

    mu_hold = gamma * (sigma @ w_minus)
    diag_hold = evaluate_kkt(-mu_hold + gamma * (sigma @ w_minus), w_minus, eta_b, eta_s, w_max)
    assert not diag_hold["should_trade"]
    assert diag_hold["delta_g"] < band

    w_hold, to_hold = solve_qp(mu_hold, sigma, w_minus, gamma, eta_b, eta_s, w_max)
    assert to_hold < 1e-5
    np.testing.assert_allclose(w_hold, w_minus, atol=1e-5)

    mu_trade = mu_hold.copy()
    mu_trade[0] += 0.05
    mu_trade[1] -= 0.05
    diag_trade = evaluate_kkt(-mu_trade + gamma * (sigma @ w_minus), w_minus, eta_b, eta_s, w_max)
    assert diag_trade["should_trade"]
    assert diag_trade["delta_g"] > band

    w_trade, to_trade = solve_qp(mu_trade, sigma, w_minus, gamma, eta_b, eta_s, w_max)
    assert to_trade > 1e-3
    assert w_trade[0] > w_minus[0]
    assert w_trade[1] < w_minus[1]


def test_kkt_boundary_and_cap_drift() -> None:
    """Boundary flagging and cap-drift infeasibility detection."""
    eta_b, eta_s, w_max = 0.0075, 0.0125, 0.15

    infeasible = np.full(5, 1.0 / 5.0)
    diag = evaluate_kkt(np.zeros(5), infeasible, eta_b, eta_s, w_max)
    assert diag["is_cap_drift"]
    assert np.isnan(diag["delta_g"])

    feasible = np.full(10, 0.1)
    grad = np.zeros(10)
    grad[0] = eta_b + eta_s + 5e-7
    diag_boundary = evaluate_kkt(grad, feasible, eta_b, eta_s, w_max)
    assert diag_boundary["is_boundary"]
    assert not diag_boundary["is_cap_drift"]


def test_kkt_vector_volatility_penalty() -> None:
    """Theorem 1 holds under constituent-specific volatility-scaled penalties."""
    n_assets, gamma, w_max = 10, 2.5, 0.15
    w_minus = np.full(n_assets, 1.0 / n_assets)

    rng = np.random.default_rng(123)
    sigma = _psd(rng, n_assets)

    scales = np.ones(n_assets)
    scales[0], scales[1] = 2.5, 0.4
    scales = scales / np.mean(scales)
    eta_b, eta_s = 5.0 * 0.0015 * scales, 5.0 * 0.0025 * scales
    base = gamma * (sigma @ w_minus)

    mu_hold = base.copy()
    mu_hold[0] += 0.005
    mu_hold[1] -= 0.005
    diag_hold = evaluate_kkt(-mu_hold + base, w_minus, eta_b, eta_s, w_max)
    assert not diag_hold["should_trade"]
    _, to_hold = solve_qp(mu_hold, sigma, w_minus, gamma, eta_b, eta_s, w_max)
    assert to_hold < 1e-4

    mu_trade = base.copy()
    mu_trade[1] += 0.06
    mu_trade[2] -= 0.06
    diag_trade = evaluate_kkt(-mu_trade + base, w_minus, eta_b, eta_s, w_max)
    assert diag_trade["should_trade"]
    assert diag_trade["margin"] > 0
    _, to_trade = solve_qp(mu_trade, sigma, w_minus, gamma, eta_b, eta_s, w_max)
    assert to_trade > 1e-3


def test_unpenalized_reduction_matches_split() -> None:
    """The ``eta = 0`` reduced formulation agrees with the 3N splitting.

    Both solve the same convex problem. The split form is singular when the penalties
    vanish, so the reduced form is the numerically better one; this test pins the two
    to within the tolerance at which the trade/hold verdict could flip.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(25):
        n = 40
        sigma = _psd(rng, n)
        w_minus = rng.dirichlet(np.ones(n))
        mu = rng.normal(0.0, 0.01, n)

        reduced, _ = solve_qp(mu, sigma, w_minus, 2.5, 0.0, 0.0, w_max=0.15)

        res = minimize(
            fun=lambda w: float(-w @ mu + 1.25 * w @ (sigma @ w)),
            x0=w_minus,
            jac=lambda w: -mu + 2.5 * (sigma @ w),
            method="SLSQP",
            bounds=[(0.0, 0.15)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
            options={"ftol": 1e-12, "maxiter": 200},
        )
        worst = max(worst, float(np.max(np.abs(reduced - np.clip(res.x, 0.0, 0.15)))))

    assert worst < 1e-4, f"reduced and split formulations diverge by {worst:.2e}"


def test_solve_qp_raises_on_degenerate_input() -> None:
    """A problem with no feasible point must fail loudly, not return a silent clip."""
    mu = np.zeros(4)
    sigma = np.eye(4)
    w_minus = np.full(4, 0.25)
    with pytest.raises(RuntimeError):
        solve_qp(mu, sigma, w_minus, 2.5, 0.0075, 0.0125, w_max=0.20)


def test_feasible_start_satisfies_all_constraints() -> None:
    """The warm start is feasible even when the incumbent breaches the cap.

    Warm-starting from ``(w_minus, 0, 0)`` put a violated bound into SLSQP on every
    cap-drift date, which produced a persistent line-search stall on the
    volatility-scaled extension.
    """
    from stillwar.portfolio import _feasible_start

    rng = np.random.default_rng(11)
    for _ in range(50):
        w_minus = rng.dirichlet(np.ones(40))
        w_minus[0] += 0.05  # force a cap breach
        w_minus /= w_minus.sum()
        w_max = 0.15

        x0 = _feasible_start(w_minus, w_max)
        w, u_plus, u_minus = x0[:40], x0[40:80], x0[80:]

        assert np.all(w >= -1e-12) and np.all(w <= w_max + 1e-12), "box violated"
        assert abs(w.sum() - 1.0) < 1e-12, "budget violated"
        np.testing.assert_allclose(w - u_plus + u_minus, w_minus, atol=1e-12)
        assert np.all(u_plus >= 0.0) and np.all(u_minus >= 0.0)


def test_volatility_scaled_solve_converges() -> None:
    """The volatility-scaled extension converges from a cap-breaching incumbent.

    Regression test for the stall that only appeared once training had run in the same
    process: ``OMP_NUM_THREADS`` leaked into later stages, changing the floating-point
    path enough to trip SLSQP's line search.
    """
    rng = np.random.default_rng(2024)
    n, gamma, w_max = 40, 2.5, 0.15
    sigma = _psd(rng, n)

    w_minus = np.full(n, 1.0 / n)
    w_minus[3] = w_max + 0.003  # passive drift above the cap
    w_minus /= w_minus.sum()

    scales = np.linspace(0.4, 2.5, n)
    scales /= scales.mean()
    eta_b = 5.0 * 0.0015 * scales
    eta_s = 5.0 * 0.0025 * scales

    mu = rng.normal(0.0, 0.01, n)
    w_opt, turnover = solve_qp(mu, sigma, w_minus, gamma, eta_b, eta_s, w_max)

    assert np.isfinite(w_opt).all()
    assert abs(w_opt.sum() - 1.0) < 1e-9
    assert np.all(w_opt <= w_max + 1e-6)
    assert turnover >= 0.0


def test_penalty_vectors_shapes(repo_root) -> None:
    """Scalar and per-asset penalties broadcast to the same shape."""
    from stillwar.config import load_config
    from stillwar.params import BacktestParams

    params = BacktestParams.from_config(load_config(repo_root / "configs/base.yaml"))
    eta_b, eta_s = params.penalty_vectors(40)
    assert eta_b.shape == (40,) and eta_s.shape == (40,)
    np.testing.assert_allclose(eta_b, params.eta_b)

    scaled_b, _ = params.penalty_vectors(40, np.full(40, 2.0))
    np.testing.assert_allclose(scaled_b, params.eta_b * 2.0)
