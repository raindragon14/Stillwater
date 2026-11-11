"""Compatibility shim for the old ``stillwar.train`` import path.

New code should import from :mod:`stillwar.models.trainer`.
"""

from stillwar.models.trainer import (
    MODELS,
    FitResult,
    fit_single_seed,
    gate_activations,
    select_best_lr,
    train_all_models,
)

__all__ = [
    "MODELS",
    "FitResult",
    "fit_single_seed",
    "gate_activations",
    "select_best_lr",
    "train_all_models",
]
