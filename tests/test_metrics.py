"""Statistical primitives and portfolio aggregation."""

import numpy as np

from stillwar.backtest.metrics import (
    backtest_metrics,
    declined_trade_test,
    seed_summary,
    trigger_concentration,
)
from stillwar.backtest.stats import (
    deflated_sharpe,
    gate_weights,
    lag_autocorr,
    newey_west_alpha,
    rank_ic,
    rank_ic_test,
)


def test_rank_ic_on_monotone_cross_sections() -> None:
    """Spearman IC is +1 and -1 on monotone cross-sections, 0 on constant ones."""
    y_true = np.tile(np.linspace(1.0, 10.0, 40), (10, 1))
    np.testing.assert_allclose(rank_ic(y_true * 2.0 + 5.0, y_true), 1.0)
    np.testing.assert_allclose(rank_ic(-y_true, y_true), -1.0)
    np.testing.assert_allclose(rank_ic(np.ones((10, 40)), y_true), 0.0)


def test_rank_ic_test_effective_ratio() -> None:
    """The overlap correction scales the degrees of freedom, not the mean."""
    rng = np.random.default_rng(42)
    ics = np.full(100, 0.05) + rng.normal(0.0, 0.02, 100)

    weekly = rank_ic_test(ics, effective_ratio=1.0)
    daily = rank_ic_test(ics, effective_ratio=0.2)

    assert weekly["n"] == 100
    assert weekly["df"] == 99.0
    assert daily["df"] == 19.0
    assert weekly["mean"] == daily["mean"]
    assert weekly["p"] < daily["p"], "overlap correction must widen the p-value"


def test_lag_autocorr_positive_on_persistent_signal() -> None:
    """An AR(1) forecast series has positive lag-1 autocorrelation on the rebalance grid."""
    rng = np.random.default_rng(42)
    series = np.zeros((100, 40))
    for j in range(40):
        value = 0.0
        for t in range(100):
            value = 0.9 * value + float(rng.normal(0.0, 0.1))
            series[t, j] = value
    assert lag_autocorr(series, step=5) > 0.0


def test_gate_weights_uniform() -> None:
    """A uniform gate averages to one third per branch."""
    gw = gate_weights(np.ones((50, 40, 3)) / 3.0)
    assert np.isclose(gw["omega_5"], 1 / 3)
    assert np.isclose(gw["omega_20"], 1 / 3)
    assert np.isclose(gw["omega_60"], 1 / 3)


def test_newey_west_recovers_known_alpha() -> None:
    """A synthetic series with a constant 10% annual excess recovers that alpha."""
    rng = np.random.default_rng(0)
    benchmark = rng.normal(0.0, 0.01, 3000)
    strategy = 0.10 / 252.0 + benchmark + rng.normal(0.0, 0.001, 3000)
    result = newey_west_alpha(strategy, benchmark)
    assert abs(result["alpha"] - 0.10) < 0.02
    assert result["t"] > 3.0


def test_newey_west_short_sample_returns_zero() -> None:
    """Too short a sample yields an explicit zero rather than an exception."""
    assert newey_west_alpha(np.zeros(3), np.zeros(3)) == {"alpha": 0.0, "t": 0.0}


def test_deflated_sharpe_penalises_many_trials() -> None:
    """More trials, or a wider trial distribution, lowers the deflated probability."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0005, 0.01, 1000)
    few = deflated_sharpe(1.5, rng.normal(0.5, 0.2, 10), returns)
    many = deflated_sharpe(1.5, rng.normal(0.5, 0.2, 1200), returns)
    wide = deflated_sharpe(1.5, rng.normal(0.5, 0.6, 1200), returns)
    assert many < few
    assert wide < many


def test_seed_summary_ddof() -> None:
    """Seed dispersion uses the sample standard deviation."""
    out = seed_summary([1.0, 2.0, 3.0])
    assert out["n"] == 3
    assert np.isclose(out["mean"], 2.0)
    assert np.isclose(out["sd"], 1.0)
    assert seed_summary([5.0])["sd"] == 0.0


def test_backtest_metrics_returns_floats() -> None:
    """Every value of a metrics block is a float, never a formatted string."""
    result = {
        "net_cagr": 0.0858,
        "sharpe": 0.21,
        "mdd": 0.3394,
        "turnover": 0.2423,
        "psi_kappa": 0.7643,
        "kkt_agreement_rate": 1.0,
    }
    metrics = backtest_metrics(result)
    assert all(isinstance(v, float) for v in metrics.values())


def _synthetic_logs(n_cycles: int, held: set[int]) -> list[dict]:
    return [
        {
            "cycle": c,
            "date": f"2024-01-{c + 1:02d}",
            "delta_g": 0.03 if c not in held else 0.0,
            "threshold": 0.02,
            "margin": 0.01 if c not in held else -0.02,
            "kkt_should_trade": c not in held,
            "solver_traded": c not in held,
            "is_boundary": False,
            "is_cap_drift": False,
            "trigger_sell_idx": 0,
            "trigger_buy_idx": 1,
        }
        for c in range(n_cycles)
    ]


def test_declined_trade_test_counts_holds() -> None:
    """Only cycles held by the penalized run and traded by the unpenalized one count."""
    penalized = {"kkt_logs": _synthetic_logs(10, held={3, 7}), "daily_returns": np.zeros(50)}
    unpenalized = {"kkt_logs": _synthetic_logs(10, held=set()), "daily_returns": np.zeros(50)}
    out = declined_trade_test(penalized, unpenalized, rebalance_freq=5)
    assert out["n_declined"] == 2.0


def test_trigger_concentration_counts_two_endpoints_per_crossing() -> None:
    """Each crossing date contributes one sell and one buy trigger."""
    tickers = [f"T{i}.JK" for i in range(5)]
    out = trigger_concentration(_synthetic_logs(10, held=set()), tickers)
    assert out["crossing_count"] == 10
    assert out["total_triggers"] == 20.0
    assert np.isclose(sum(out["sector_shares"].values()), 1.0)
