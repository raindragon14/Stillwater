"""Raw OHLCV ingestion and the SHA-256 integrity manifest."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from stillwater.config import Config
from stillwater.data.universe import CANDIDATES

__all__ = ["download", "verify_manifest", "write_manifest"]

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    """Hash a raw CSV after normalising line endings to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_manifest(raw_dir: Path, manifest: Path) -> None:
    """Record the SHA-256 of every raw CSV in ``manifest``."""
    files = sorted(raw_dir.glob("*.csv"))
    manifest.write_text(json.dumps({f.name: _sha256(f) for f in files}, indent=2), encoding="utf-8")


def verify_manifest(raw_dir: str | Path, manifest: str | Path) -> int:
    """Raise if any raw CSV is missing or differs from its recorded SHA-256.

    Returns the number of files checked.
    """
    raw_dir = Path(raw_dir)
    expected = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for key, value in expected.items():
        filename = value["file"] if isinstance(value, dict) else key
        digest = value["sha256"] if isinstance(value, dict) else value
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        if _sha256(path) != digest:
            raise ValueError(f"hash mismatch: {filename}")
    return len(expected)


def download(cfg: Config, tickers: list[str] = CANDIDATES) -> None:
    """Fetch missing raw CSVs from Yahoo Finance (requires the ``download`` extra)."""
    import yfinance as yf

    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        path = raw_dir / f"{ticker}.csv"
        if path.exists():
            continue
        frame = yf.download(
            ticker,
            start=cfg.data.start,
            end=cfg.data.end,
            progress=False,
            auto_adjust=False,
        )
        if frame.empty:
            log.warning("no data for %s", ticker)
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.to_csv(path)
    write_manifest(raw_dir, Path(cfg.data.manifest))
