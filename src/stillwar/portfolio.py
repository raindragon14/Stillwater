"""Compatibility shim for the old ``stillwar.portfolio`` import path.

New code should import from :mod:`stillwar.optimization` and
:mod:`stillwar.backtest.engine`.
"""

from stillwar.backtest.engine import BacktestResult, compute_summary, run_backtest, slice_panel_a
from stillwar.optimization.covariance import ledoit_wolf_covariance
from stillwar.optimization.kkt import KKTLog, evaluate_kkt, tabulate_kkt_agreement
from stillwar.optimization.qp import solve_qp

__all__ = [
    "BacktestResult",
    "KKTLog",
    "compute_summary",
    "evaluate_kkt",
    "ledoit_wolf_covariance",
    "run_backtest",
    "slice_panel_a",
    "solve_qp",
    "tabulate_kkt_agreement",
]
