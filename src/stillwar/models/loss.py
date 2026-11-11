"""Composite calibration-plus-ranking loss (JOURNAL Eq. 3)."""

from __future__ import annotations

import torch

__all__ = ["composite_loss"]


def _zscore(values: torch.Tensor) -> torch.Tensor:
    """Cross-sectional z-score along the asset axis, clipped to ``[-3, 3]``."""
    return torch.clamp(
        (values - values.mean(1, keepdim=True)) / (values.std(1, keepdim=True) + 1e-8),
        -3.0,
        3.0,
    )


def composite_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    var_y: float,
    lam: float = 1.0,
) -> torch.Tensor:
    """NMSE plus a differentiable cross-sectional ranking surrogate (JOURNAL Eq. 3).

    Parameters
    ----------
    pred, y
        ``(n_dates, n_assets)`` forecasts and targets.
    var_y
        Training variance of the target, used to normalise the squared error.
    lam
        Weight on the ``1 - rho`` ranking term.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    nmse = ((pred - y) ** 2).mean() / var_y
    u, v = _zscore(pred), _zscore(y)
    u, v = u - u.mean(1, keepdim=True), v - v.mean(1, keepdim=True)
    rho = (u * v).sum(1) / (
        ((u**2).sum(1) + 1e-8).sqrt() * ((v**2).sum(1) + 1e-8).sqrt()
    )
    return nmse + lam * (1.0 - rho).mean()
