"""Compatibility shim for the old ``stillwar.metrics`` import path.

New code should import from :mod:`stillwar.backtest.metrics`.
"""

from stillwar.backtest.metrics import (
    backtest_metrics,
    declined_trade_test,
    rank_ic_summary,
    seed_summary,
    trigger_concentration,
    trigger_endpoints,
)

__all__ = [
    "backtest_metrics",
    "declined_trade_test",
    "rank_ic_summary",
    "seed_summary",
    "trigger_concentration",
    "trigger_endpoints",
]
