"""Raw OHLCV snapshots: integrity manifest, universe selection and the aligned adjusted-price panel."""

import hashlib
import json
import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from stillwar.config import Config, Data
from stillwar.universe import CANDIDATES

log = logging.getLogger(__name__)
OHLC = ("Open", "High", "Low", "Close")


@dataclass(frozen=True)
class Panel:
    """Aligned panel: ``ohlcv`` is (T, N, 5) adjusted O/H/L/C plus raw volume, ``close`` is (T, N)."""

    tickers: list[str]
    dates: pd.DatetimeIndex
    ohlcv: np.ndarray

    @property
    def close(self) -> pd.DataFrame:
        """Adjusted close prices as a ``(T, N)`` frame indexed by trading date."""
        return pd.DataFrame(self.ohlcv[:, :, 3], index=self.dates, columns=self.tickers)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_manifest(raw_dir: Path, manifest: Path) -> None:
    """Record the SHA-256 of every raw CSV in ``manifest``."""
    files = sorted(raw_dir.glob("*.csv"))
    manifest.write_text(json.dumps({f.name: _sha256(f) for f in files}, indent=2), encoding="utf-8")


def verify_manifest(raw_dir: str | Path, manifest: str | Path) -> int:
    """Raises if any raw CSV is missing or differs from its recorded SHA-256; returns files checked."""
    raw_dir = Path(raw_dir)
    expected = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for key, val in expected.items():
        fname = val["file"] if isinstance(val, dict) else key
        digest = val["sha256"] if isinstance(val, dict) else val
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        if _sha256(path) != digest:
            raise ValueError(f"hash mismatch: {fname}")
    return len(expected)


def download(cfg: Config, tickers: list[str] = CANDIDATES) -> None:
    """Fetches missing raw CSVs from Yahoo Finance (requires the ``download`` extra)."""
    import yfinance as yf

    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        path = raw_dir / f"{t}.csv"
        if path.exists():
            continue
        df = yf.download(t, start=cfg.data.start, end=cfg.data.end, progress=False, auto_adjust=False)
        if df.empty:
            log.warning("no data for %s", t)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(path)
    write_manifest(raw_dir, Path(cfg.data.manifest))


def _read(raw_dir: Path, ticker: str) -> pd.DataFrame:
    return pd.read_csv(raw_dir / f"{ticker}.csv", index_col=0, parse_dates=True)


def select_universe(cfg: Config) -> list[str]:
    """Top ``n_stocks`` survivors by median ``Close * Volume`` in the ranking year."""
    raw_dir, year = Path(cfg.data.raw_dir), cfg.data.ranking_year
    last_required = pd.Timestamp(cfg.data.end) - pd.Timedelta(days=35)
    liquidity = {}
    for t in CANDIDATES:
        path = raw_dir / f"{t}.csv"
        if not path.exists():
            continue
        df = _read(raw_dir, t)
        if df.empty or df.index.min() > pd.Timestamp(f"{year}-01-10") or df.index.max() < last_required:
            continue
        df = df.loc[str(year)]
        if len(df) >= 150:
            liquidity[t] = (df["Close"] * df["Volume"]).median()
    ranked = sorted(liquidity, key=liquidity.get, reverse=True)
    if len(ranked) < cfg.data.n_stocks:
        raise RuntimeError(f"only {len(ranked)} eligible tickers")
    return ranked[: cfg.data.n_stocks]


@cache
def _load_panel(data_json: str) -> Panel:
    cfg = Config.model_construct(data=Data.model_validate_json(data_json))
    verify_manifest(cfg.data.raw_dir, cfg.data.manifest)
    tickers = select_universe(cfg)
    frames = {t: _read(Path(cfg.data.raw_dir), t) for t in tickers}
    index = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))))
    out = np.empty((len(index), len(tickers), 5))
    for j, t in enumerate(tickers):
        df = frames[t].reindex(index)
        factor = (df["Adj Close"] / df["Close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for k, col in enumerate(OHLC[:3]):
            out[:, j, k] = (df[col] * factor).ffill()
        out[:, j, 3] = df["Adj Close"].ffill()
        out[:, j, 4] = df["Volume"].fillna(0.0)
    return Panel(tickers, index, out)


def load_panel(cfg: Config) -> Panel:
    """Verified, forward-filled (never backward) adjusted panel; memoised per data config."""
    return _load_panel(cfg.data.model_dump_json())
