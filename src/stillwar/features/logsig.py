"""Order-2 log-signature features and Levy areas (JOURNAL Eq. 1)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "BLOCK",
    "INCREMENT_SLOTS",
    "SEQ_LEN",
    "compute_increments_and_levy_areas",
    "logsig_features",
]

SEQ_LEN = 60
BLOCK = 15  # 5 increments + 10 Levy areas per window
INCREMENT_SLOTS = tuple(range(5))


def compute_increments_and_levy_areas(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Increments and antisymmetric Levy areas for a single path of shape ``(W+1, 5)``.

    This is the scalar reference implementation of Eq. (1). It is deliberately slow
    and obvious, and exists so that :func:`logsig_features` -- the vectorised path the
    pipeline actually uses -- can be checked against it.

    Parameters
    ----------
    path
        ``(W + 1, n_channels)`` array of log-price and log-volume observations.

    Returns
    -------
    tuple of ndarray
        The ``n_channels`` increments ``X_W - X_0`` and the
        ``n_channels * (n_channels - 1) / 2`` upper-triangular Levy areas.

    Notes
    -----
    For a piecewise-linear path the area is an exact finite sum over its segments,
    ``A^{ij} = 1/2 sum_k [(X_k - X_0)^i dX_k^j - (X_k - X_0)^j dX_k^i]``.
    """
    increments = path[-1] - path[0]
    x = path - path[0]
    dx = np.diff(path, axis=0)
    m = np.sum(x[:-1, :, None] * dx[:, None, :], axis=0)
    area = 0.5 * (m - m.T)
    iu = np.triu_indices(path.shape[1], 1)
    return increments, area[iu[0], iu[1]]


def logsig_features(path: np.ndarray, windows: tuple[int, ...]) -> np.ndarray:
    """Order-2 log-signature of every trailing window (JOURNAL Eq. 1).

    Vectorised over time and assets. The ten Levy areas come from one cumulative sum
    of ``x_k^i dx_k^j`` using the identity

        ``sum_k (x_k - x_0)^i dx_k^j = dS_ij - x_0^i (x_W - x_0)^j``

    so the whole ``(T, N, 15 * len(windows))`` tensor costs a handful of array ops.

    Parameters
    ----------
    path
        ``(T, N, 5)`` array of log open, high, low, close and ``log1p(volume)``.
    windows
        Trailing window lengths in trading days, e.g. ``(5, 20, 60)``.

    Returns
    -------
    ndarray
        ``(T, N, 15 * len(windows))`` features; rows before the longest window are NaN.

    Notes
    -----
    Increments and Levy areas are both invariant to a per-channel additive constant,
    so the centring by the panel mean cannot leak information: it only improves
    float32 conditioning. Do not add a non-translation-invariant channel here without
    also replacing that centring with a training-window statistic.
    """
    x = np.nan_to_num(path - np.nanmean(path, axis=0))
    s = np.zeros((*x.shape[:2], x.shape[-1], x.shape[-1]))
    s[1:] = np.cumsum(np.einsum("tni,tnj->tnij", x[:-1], np.diff(x, axis=0)), axis=0)
    iu = np.triu_indices(x.shape[-1], 1)
    out = np.full((*x.shape[:2], BLOCK * len(windows)), np.nan)
    for b, w in enumerate(windows):
        x0, inc = x[:-w], x[w:] - x[:-w]
        m = (s[w:] - s[:-w]) - x0[..., :, None] * inc[..., None, :]
        area = 0.5 * (m - np.swapaxes(m, -1, -2))
        block = out[w:, :, b * BLOCK : (b + 1) * BLOCK]
        block[..., :5] = inc
        block[..., 5:] = area[..., iu[0], iu[1]]
    return out
