"""Experiments package: table runs, sensitivity grids and the Deflated-Sharpe trials."""

from stillwater.experiments.sensitivity import (
    run_shrinkage_experiment,
    run_unconstrained_and_tolerance_experiment,
    run_volatility_penalty_experiment,
)
from stillwater.experiments.simulate import (
    run_all,
    run_phase4,
    run_table1,
    run_table2,
    run_table3,
)
from stillwater.experiments.trials import load_trials, run_trial_grid

__all__ = [
    "load_trials",
    "run_all",
    "run_phase4",
    "run_shrinkage_experiment",
    "run_table1",
    "run_table2",
    "run_table3",
    "run_trial_grid",
    "run_unconstrained_and_tolerance_experiment",
    "run_volatility_penalty_experiment",
]
