"""Aligned adjusted-price panel and deterministic universe selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from stillwater.config import Config, Data
from stillwater.data.ingest import verify_manifest
from stillwater.data.universe import CANDIDATES

__all__ = ["OHLC", "Panel", "load_panel", "select_universe"]

OHLC = ("Open", "High", "Low", "Close")


@dataclass(frozen=True)
class Panel:
    """Aligned panel: ``ohlcv`` is (T, N, 5) adjusted O/H/L/C plus raw volume.

    ``close`` is the (T, N) adjusted-close frame.
    """

    tickers: list[str]
    dates: pd.DatetimeIndex
    ohlcv: np.ndarray

    @property
    def close(self) -> pd.DataFrame:
        """Adjusted close prices as a ``(T, N)`` frame indexed by trading date."""
        return pd.DataFrame(self.ohlcv[:, :, 3], index=self.dates, columns=self.tickers)


def _read(raw_dir: Path, ticker: str) -> pd.DataFrame:
    """Read one raw Yahoo Finance CSV."""
    return pd.read_csv(raw_dir / f"{ticker}.csv", index_col=0, parse_dates=True)


def select_universe(cfg: Config) -> list[str]:
    """Top ``n_stocks`` survivors by median ``Close * Volume`` in the ranking year."""
    raw_dir, year = Path(cfg.data.raw_dir), cfg.data.ranking_year
    last_required = pd.Timestamp(cfg.data.end) - pd.Timedelta(days=35)
    liquidity: dict[str, float] = {}
    for ticker in CANDIDATES:
        path = raw_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        frame = _read(raw_dir, ticker)
        if (
            frame.empty
            or frame.index.min() > pd.Timestamp(f"{year}-01-10")
            or frame.index.max() < last_required
        ):
            continue
        year_frame = frame.loc[str(year)]
        if len(year_frame) >= 150:
            liquidity[ticker] = float((year_frame["Close"] * year_frame["Volume"]).median())
    ranked = sorted(liquidity, key=lambda ticker: liquidity[ticker], reverse=True)
    if len(ranked) < cfg.data.n_stocks:
        raise RuntimeError(f"only {len(ranked)} eligible tickers")
    return ranked[: cfg.data.n_stocks]


@cache
def _load_panel(data_json: str) -> Panel:
    """Memoised panel builder keyed on the serialised data section."""
    cfg = Config.model_construct(data=Data.model_validate_json(data_json))
    verify_manifest(cfg.data.raw_dir, cfg.data.manifest)
    tickers = select_universe(cfg)
    frames = {ticker: _read(Path(cfg.data.raw_dir), ticker) for ticker in tickers}
    index = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in frames.values()))))
    out = np.empty((len(index), len(tickers), 5))
    for j, ticker in enumerate(tickers):
        frame = frames[ticker].reindex(index)
        factor = (
            (frame["Adj Close"] / frame["Close"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
        )
        for k, column in enumerate(OHLC[:3]):
            out[:, j, k] = (frame[column] * factor).ffill()
        out[:, j, 3] = frame["Adj Close"].ffill()
        out[:, j, 4] = frame["Volume"].fillna(0.0)
    return Panel(tickers, index, out)


def load_panel(cfg: Config) -> Panel:
    """Verified, forward-filled (never backward) adjusted panel; memoised per data config."""
    return _load_panel(cfg.data.model_dump_json())
