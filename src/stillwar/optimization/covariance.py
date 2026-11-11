"""Ledoit-Wolf shrinkage covariance (JOURNAL Section 2.5)."""

from __future__ import annotations

import numpy as np

__all__ = ["ledoit_wolf_covariance"]


def ledoit_wolf_covariance(
    returns_60d: np.ndarray,
    scale_to_horizon: float = 5.0,
    ridge_eps: float = 1e-12,
) -> np.ndarray:
    """Analytical Ledoit & Wolf (2004) linear-shrinkage covariance, scaled to the horizon.

    Shrinks the sample covariance toward a scaled identity, with the optimal
    shrinkage intensity derived in closed form from the fourth moments of the data.

    Parameters
    ----------
    returns_60d
        ``(n_obs, n_assets)`` matrix of daily simple returns.
    scale_to_horizon
        Multiplier applied after shrinkage; ``5.0`` converts a daily covariance to
        the five-day forecast horizon of Eq. (4).
    ridge_eps
        Diagonal ridge added after scaling, so the matrix is strictly positive
        definite even when the shrinkage target is degenerate.

    Returns
    -------
    ndarray
        Symmetric ``(n_assets, n_assets)`` covariance matrix.

    Raises
    ------
    ValueError
        If fewer than four observations are supplied.

    Notes
    -----
    JOURNAL Section 2.5: ``Sigma_t = 5 * Sigma_t^LW + 1e-12 I``.
    """
    n_obs, n_assets = returns_60d.shape
    if n_obs < 4:
        raise ValueError(f"Need at least 4 observations for Ledoit-Wolf, got {n_obs}")

    x_centered = returns_60d - np.mean(returns_60d, axis=0, keepdims=True)
    sample_cov = (x_centered.T @ x_centered) / n_obs

    var_avg = float(np.trace(sample_cov) / n_assets)
    target_f = var_avg * np.eye(n_assets)

    diffs = x_centered[:, :, None] * x_centered[:, None, :] - sample_cov[None, :, :]
    pi_hat = float(np.sum(diffs**2) / n_obs)
    gamma_hat = float(np.sum((sample_cov - target_f) ** 2))

    shrinkage = max(0.0, min(1.0, (pi_hat / n_obs) / gamma_hat)) if gamma_hat > 1e-12 else 1.0
    shrunk_cov = shrinkage * target_f + (1.0 - shrinkage) * sample_cov
    sigma = scale_to_horizon * shrunk_cov + ridge_eps * np.eye(n_assets)
    return 0.5 * (sigma + sigma.T)
