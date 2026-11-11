"""Compatibility shim for the old ``stillwar.params`` import path.

New code should import :class:`BacktestParams` from :mod:`stillwar.config`.
"""

from stillwar.config.params import BacktestParams

__all__ = ["BacktestParams"]
