"""Model package: architectures, loss and deterministic trainer."""

from stillwater.models.architectures import (
    LSTM,
    MLP,
    MSGN,
    Gated,
    LogSigInc,
    LSTMBaseline,
    MLPBaseline,
    build_model,
)
from stillwater.models.loss import composite_loss
from stillwater.models.trainer import (
    MODELS,
    FitResult,
    fit_single_seed,
    gate_activations,
    select_best_lr,
    train_all_models,
)

__all__ = [
    "Gated",
    "LSTM",
    "LSTMBaseline",
    "MLP",
    "MLPBaseline",
    "MODELS",
    "MSGN",
    "FitResult",
    "LogSigInc",
    "build_model",
    "composite_loss",
    "fit_single_seed",
    "gate_activations",
    "select_best_lr",
    "train_all_models",
]
