"""Order-2 log-signature features, execution-lagged targets, and purged/embargoed splits.

The feature tensor is the only expensive artefact that depends on nothing but the raw
data and the window geometry, so it is persisted under ``.cache/dataset/`` and keyed
on :meth:`stillwar.config.Config.feature_digest`. That lets a sensitivity grid reuse
one build instead of rebuilding 45 features on every configuration override.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from stillwar import cache as disk_cache
from stillwar.config import Config
from stillwar.data import Panel, load_panel

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
        The ``n_channels`` increments ``X_W - X_0`` and the ``n_channels * (n_channels - 1) / 2``
        upper-triangular Levy areas.

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


@dataclass(frozen=True)
class Dataset:
    """Standardised model inputs, aligned on decision dates.

    Attributes
    ----------
    x
        ``(V, N, 45)`` log-signature features, z-scored with training-pooled moments.
    seq
        ``(V, N, 60, 1)`` daily log returns for the sequence baseline.
    y
        ``(V, N)`` target ``log(P[t+1+h] / P[t+1])``.
    dates
        Decision dates, ascending.
    tickers
        Asset identifiers, aligned with the second axis.
    idx
        Row indices for ``train``, ``val``, ``test_a`` and ``test_b``.
    n_windows
        Number of log-signature windows, used to slice the increments-only ablation.
    """

    x: np.ndarray
    seq: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    tickers: list[str]
    idx: dict[str, np.ndarray]
    n_windows: int

    def inputs(self, model: str) -> np.ndarray:
        """Model-specific input tensor.

        Parameters
        ----------
        model
            One of ``"logsig"``, ``"logsig_inc"``, ``"mlp"`` or ``"lstm"``.

        Returns
        -------
        ndarray
            ``seq`` for the LSTM, the 15-increment ablation for ``logsig_inc``, and
            the full 45-feature tensor otherwise.
        """
        if model == "lstm":
            return self.seq
        if model == "logsig_inc":
            return self.x[..., [w * BLOCK + i for w in range(self.n_windows) for i in INCREMENT_SLOTS]]
        return self.x

    def historical_forecast(self, alpha: float) -> np.ndarray:
        """Hist-MV signal: 60-day mean log return over a week, shrunk to the cross-sectional mean.

        Parameters
        ----------
        alpha
            Shrinkage intensity; ``0`` keeps the raw signal, ``1`` collapses it to the
            cross-sectional mean and therefore to the GMV limit.

        Returns
        -------
        ndarray
            ``(V, N)`` forecast array.
        """
        mu = self.seq[..., 0].mean(axis=2) * 5.0
        return (1.0 - alpha) * mu + alpha * mu.mean(axis=1, keepdims=True)

    def save(self, path: str | Path) -> None:
        """Persist the dataset to a compressed ``.npz`` for reuse across stages."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            x=self.x,
            seq=self.seq,
            y=self.y,
            dates=np.array([d.strftime("%Y-%m-%d") for d in self.dates]),
            tickers=np.array(self.tickers),
            n_windows=np.array(self.n_windows),
            **{f"idx_{k}": v for k, v in self.idx.items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        """Restore a dataset written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as z:
            keys = [k for k in z.files if k.startswith("idx_")]
            return cls(
                x=z["x"],
                seq=z["seq"],
                y=z["y"],
                dates=pd.DatetimeIndex(pd.to_datetime(z["dates"])),
                tickers=[str(t) for t in z["tickers"]],
                idx={k.removeprefix("idx_"): z[k] for k in keys},
                n_windows=int(z["n_windows"]),
            )


def split_indices(dates: pd.DatetimeIndex, cfg: Config) -> dict[str, np.ndarray]:
    """Row indices for each partition, with the purge and embargo applied.

    The last ``purge_days`` decision dates of the training and validation windows are
    dropped because their five-day labels overlap the next partition, and the first
    ``embargo_days`` dates of validation and test are dropped for the same reason.
    Together they give the twelve-trading-day separation of Section 2.1.

    Parameters
    ----------
    dates
        Decision dates, ascending.
    cfg
        Configuration supplying the split boundaries and the two lengths.

    Returns
    -------
    dict
        ``train``, ``val``, ``test_a`` and ``test_b`` integer row indices.
    """
    s = cfg.splits

    def rows(rng: tuple[str, str]) -> np.ndarray:
        return np.flatnonzero((dates >= rng[0]) & (dates <= rng[1]))

    purge, embargo = s.purge_days, s.embargo_days
    return {
        "train": rows(s.train)[:-purge],
        "val": rows(s.val)[:-purge][embargo:],
        "test_a": rows(s.test_a)[embargo:],
        "test_b": rows(s.test_b)[embargo:],
    }


def build_dataset(panel: Panel, cfg: Config) -> Dataset:
    """Build the standardised feature tensor from an aligned price panel.

    Standardisation uses the mean and standard deviation pooled across training
    stock-days only, so no test-period information enters the features.

    Parameters
    ----------
    panel
        Verified, forward-filled OHLCV panel from :func:`stillwar.data.load_panel`.
    cfg
        Configuration supplying windows, horizon and split boundaries.

    Returns
    -------
    Dataset
        Features, sequences, targets and partition indices.
    """
    f = cfg.features
    windows = tuple(f.windows)
    close = panel.ohlcv[:, :, 3]
    path = np.stack(
        [np.log(np.maximum(panel.ohlcv[..., k], 1e-4)) for k in range(4)]
        + [np.log1p(np.maximum(panel.ohlcv[..., 4], 0.0))],
        axis=-1,
    )
    lead = f.execution_lag + f.horizon_days
    t = np.arange(max(max(windows), SEQ_LEN), len(panel.dates) - lead)

    raw = logsig_features(path, windows)[t].astype(np.float32)
    y = np.log(close[t + lead] / close[t + f.execution_lag]).astype(np.float32)
    daily = np.zeros_like(close)
    daily[1:] = np.log(np.maximum(close[1:] / np.maximum(close[:-1], 1e-4), 1e-6))
    seq = sliding_window_view(daily, SEQ_LEN, axis=0)[t - SEQ_LEN + 1]

    dates = panel.dates[t]
    idx = split_indices(dates, cfg)
    train = raw[idx["train"]].reshape(-1, raw.shape[-1])
    mu, sigma = train.mean(axis=0), np.maximum(train.std(axis=0), 1e-6)
    return Dataset(
        x=((raw - mu) / sigma).astype(np.float32),
        seq=seq[..., None].astype(np.float32),
        y=y,
        dates=dates,
        tickers=panel.tickers,
        idx=idx,
        n_windows=len(windows),
    )


@cache
def _load_cached(path_str: str) -> Dataset:
    """In-process memoisation of a persisted dataset, keyed on its file path."""
    return Dataset.load(path_str)


def load_dataset(cfg: Config) -> Dataset:
    """Memoised dataset for ``cfg``.

    Keyed on :meth:`Config.feature_digest`, not on the whole configuration, so a
    ``trading`` override reuses the existing build instead of recomputing 45 features.

    Parameters
    ----------
    cfg
        Configuration; only the ``data``, ``splits`` and ``features`` sections affect
        the result.
    """
    path = disk_cache.dataset_cache_path(cfg.feature_digest())
    if path.exists() and not disk_cache.force():
        return _load_cached(str(path))
    dataset = build_dataset(load_panel(cfg), cfg)
    dataset.save(path)
    return _load_cached(str(path))
