"""Unit tests for trainable parameter counts across MSGN and baseline architectures.

Verifies exact match with ``docs/JOURNAL.md`` Sections 2.4 and 2.8:
- MSGN (LogSig): 923 parameters
- LogSig-Inc: 353 parameters
- MLPBaseline: 1,505 parameters
- LSTMBaseline: 4,513 parameters
"""

import torch

from stillwater.models import MSGN, LogSigInc, LSTMBaseline, MLPBaseline


def count_params(model: torch.nn.Module) -> int:
    """Returns total number of trainable scalar parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_msgn_parameter_count() -> None:
    """Verifies MSGN has exactly 923 trainable parameters."""
    model = MSGN(hidden_dim=16)
    assert count_params(model) == 923


def test_logsig_inc_parameter_count() -> None:
    """Verifies LogSig-Inc has exactly 353 trainable parameters."""
    model = LogSigInc(hidden_dim=16)
    assert count_params(model) == 353


def test_mlp_parameter_count() -> None:
    """Verifies single-hidden-layer MLPBaseline has exactly 1,505 parameters."""
    model = MLPBaseline(in_features=45, hidden_dim=32)
    assert count_params(model) == 1505


def test_lstm_parameter_count() -> None:
    """Verifies 32-unit LSTMBaseline has exactly 4,513 parameters."""
    model = LSTMBaseline(input_dim=1, hidden_dim=32)
    assert count_params(model) == 4513
