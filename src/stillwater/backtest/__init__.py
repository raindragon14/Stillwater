"""Backtest package: execution engine, metrics and statistics."""

from stillwater.backtest.engine import BacktestResult, compute_summary, run_backtest, slice_panel_a
from stillwater.backtest.metrics import (
    backtest_metrics,
    declined_trade_test,
    rank_ic_summary,
    seed_summary,
    trigger_concentration,
    trigger_endpoints,
)
from stillwater.backtest.stats import (
    deflated_sharpe,
    gate_weights,
    lag_autocorr,
    newey_west_alpha,
    rank_ic,
    rank_ic_test,
)
from stillwater.optimization.kkt import KKTLog

__all__ = [
    "BacktestResult",
    "KKTLog",
    "backtest_metrics",
    "compute_summary",
    "declined_trade_test",
    "deflated_sharpe",
    "gate_weights",
    "lag_autocorr",
    "newey_west_alpha",
    "rank_ic",
    "rank_ic_summary",
    "rank_ic_test",
    "run_backtest",
    "seed_summary",
    "slice_panel_a",
    "trigger_concentration",
    "trigger_endpoints",
]
