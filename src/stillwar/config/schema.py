"""Typed, frozen configuration schema for the Stillwar pipeline.

The YAML file in ``configs/`` is the single source of truth for every number in the
pipeline.  Nothing downstream may carry a default that is not also present there:
:class:`Config` is frozen, rejects unknown keys (``extra="forbid"``), and exposes
content digests that are written into every artifact for provenance.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = [
    "Config",
    "Data",
    "Features",
    "Reporting",
    "Splits",
    "Trading",
    "Train",
    "Workers",
]


class _Frozen(BaseModel):
    """Base for every config section: immutable and closed to unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Data(_Frozen):
    """Raw-data location and universe-selection parameters (JOURNAL Section 2.1)."""

    raw_dir: str
    manifest: str
    start: str
    end: str
    n_stocks: int
    ranking_year: int


class Splits(_Frozen):
    """Inclusive date ranges plus the purge and embargo lengths (JOURNAL Section 2.1)."""

    train: tuple[str, str]
    val: tuple[str, str]
    test_a: tuple[str, str]
    test_b: tuple[str, str]
    purge_days: int
    embargo_days: int


class Features(_Frozen):
    """Log-signature windows and the execution-lagged target horizon (Section 2.3)."""

    windows: tuple[int, ...]
    horizon_days: int
    execution_lag: int


class Trading(_Frozen):
    """Every knob of the mean-variance problem and the backtest (Sections 2.2, 2.5, 2.9)."""

    rebalance_freq: int
    c_b: float
    c_s: float
    kappa: float
    gamma: float
    w_max: float
    alpha: float
    cov_lookback: int
    rf_annual: float
    turnover_tol: float
    boundary_tol: float
    log_forecast: bool
    volatility_scaled: bool
    short_circuit: bool
    cap_drift_buffer: float = 0.0


class Train(_Frozen):
    """Optimiser, early-stopping and learning-rate-grid settings (Section 2.4)."""

    hidden_dim: int
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    lambda_rank: float
    lrs: dict[str, tuple[float, ...]]


class Workers(_Frozen):
    """Process counts for the two parallel stages.

    They are separated because their memory footprints differ by an order of magnitude:
    an LSTM fit holds roughly 400 MB of backpropagation state, while a backtest holds
    about 50 MB. Worker count affects wall clock only, never results.
    """

    train: int
    backtest: int


class Reporting(_Frozen):
    """Non-trading constants used only when slicing panels and reporting (Section 2.9)."""

    panel_a_rebalances: int
    effective_ratio_weekly: float = 1.0
    effective_ratio_daily: float = 0.2


class Config(_Frozen):
    """Root configuration; construct with :func:`stillwar.config.load_config`."""

    seeds: tuple[int, ...]
    data: Data
    splits: Splits
    features: Features
    trading: Trading
    train: Train
    workers: Workers
    reporting: Reporting

    def override(self, **dotted: Any) -> Config:
        """Return a copy with overrides written ``section__key=value``.

        Examples
        --------
        >>> cfg.override(trading__kappa=0.0, reporting__panel_a_rebalances=94)
        """
        raw = self.model_dump()
        for key, value in dotted.items():
            *parents, leaf = key.split("__")
            node = raw
            for parent in parents:
                node = node[parent]
            if leaf not in node:
                raise KeyError(key)
            node[leaf] = value
        return Config(**raw)

    @property
    def digest(self) -> str:
        """First 12 hex characters of the SHA-256 of the serialised configuration."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def feature_digest(self) -> str:
        """Digest of only the sections that determine the feature tensor.

        Feature construction depends on the data snapshot, the split boundaries and
        the window geometry; it does not depend on any trading parameter. Caching on
        this digest lets sensitivity grids reuse one feature build.
        """
        payload = {
            "data": self.data.model_dump(),
            "splits": self.splits.model_dump(),
            "features": self.features.model_dump(),
        }
        return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()[:16]
