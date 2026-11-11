"""Standardised feature tensors, execution-lagged targets and purged/embargoed splits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from stillwar import cache as disk_cache
from stillwar.config import Config
from stillwar.data import Panel, load_panel
from stillwar.features.logsig import BLOCK, INCREMENT_SLOTS, SEQ_LEN, logsig_features

__all__ = ["Dataset", "build_dataset", "load_dataset", "split_indices"]


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
            return self.x[
                ..., [w * BLOCK + i for w in range(self.n_windows) for i in INCREMENT_SLOTS]
            ]
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
    def load(cls, path: str | Path) -> Dataset:
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
    splits = cfg.splits

    def rows(rng: tuple[str, str]) -> np.ndarray:
        return np.flatnonzero((dates >= rng[0]) & (dates <= rng[1]))

    purge, embargo = splits.purge_days, splits.embargo_days
    return {
        "train": rows(splits.train)[:-purge],
        "val": rows(splits.val)[:-purge][embargo:],
        "test_a": rows(splits.test_a)[embargo:],
        "test_b": rows(splits.test_b)[embargo:],
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
    features = cfg.features
    windows = tuple(features.windows)
    close = panel.ohlcv[:, :, 3]
    path = np.stack(
        [np.log(np.maximum(panel.ohlcv[..., k], 1e-4)) for k in range(4)]
        + [np.log1p(np.maximum(panel.ohlcv[..., 4], 0.0))],
        axis=-1,
    )
    lead = features.execution_lag + features.horizon_days
    t = np.arange(max(max(windows), SEQ_LEN), len(panel.dates) - lead)

    raw = logsig_features(path, windows)[t].astype(np.float32)
    y = np.log(close[t + lead] / close[t + features.execution_lag]).astype(np.float32)
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
