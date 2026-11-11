"""Compatibility shim for the old ``stillwar.trials`` import path.

New code should import from :mod:`stillwar.experiments.trials`.
"""

from stillwar.experiments.trials import load_trials, run_trial_grid

__all__ = ["load_trials", "run_trial_grid"]
