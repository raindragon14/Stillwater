"""Compatibility shim for the old ``stillwar.stats`` import path.

New code should import from :mod:`stillwar.backtest.stats`.
"""

from stillwar.backtest.stats import (
    deflated_sharpe,
    gate_weights,
    lag_autocorr,
    newey_west_alpha,
    rank_ic,
    rank_ic_test,
)

__all__ = [
    "deflated_sharpe",
    "gate_weights",
    "lag_autocorr",
    "newey_west_alpha",
    "rank_ic",
    "rank_ic_test",
]
