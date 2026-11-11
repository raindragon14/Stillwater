"""Compatibility shim for the old ``stillwar.experiments`` import path.

New code should import from :mod:`stillwar.experiments.sensitivity`.
"""

from stillwar.experiments.sensitivity import (
    run_shrinkage_experiment,
    run_unconstrained_and_tolerance_experiment,
    run_volatility_penalty_experiment,
)

__all__ = [
    "run_shrinkage_experiment",
    "run_unconstrained_and_tolerance_experiment",
    "run_volatility_penalty_experiment",
]
