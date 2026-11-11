"""Immutable, hashable description of a single backtest configuration.

``BacktestParams`` is the only channel through which the portfolio engine learns how
to trade. It exists to remove three defects of the previous design:

1. ``run_backtest`` accepted ``**kwargs`` and fell back to literals duplicated from
   ``configs/base.yaml``, so a caller that forgot a parameter silently got a stale
   default. Table 2 was governed by those literals rather than by the config file.
2. The engine knew the schema of :class:`~stillwar.config.Config`, coupling the
   mathematical core to the configuration layer.
3. Nothing could be hashed, so no backtest result could be cached or attributed.

Every field is part of the cache key via :meth:`BacktestParams.digest`.
"""

import hashlib
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from stillwar.config.schema import Config


@dataclass(frozen=True, slots=True)
class BacktestParams:
    """Frozen parameter set for :func:`stillwar.portfolio.run_backtest`.

    Build one with :meth:`from_config`; vary it with :meth:`replace`. Values are
    never defaulted here — ``configs/base.yaml`` remains the single source of truth.
    """

    c_b: float
    c_s: float
    kappa: float
    gamma: float
    w_max: float
    alpha: float
    rebalance_freq: int
    cov_lookback: int
    rf_annual: float
    turnover_tol: float
    boundary_tol: float
    cap_drift_buffer: float
    volatility_scaled: bool
    short_circuit: bool
    log_forecast: bool

    @classmethod
    def from_config(cls, cfg: Config) -> BacktestParams:
        """Project the ``trading`` section of ``cfg`` onto a backtest parameter set.

        ``panel_a_rebalances`` is deliberately excluded: it is a reporting constant
        that slices an already-computed run, so including it would invalidate the
        backtest cache without changing any result.
        """
        t = cfg.trading
        return cls(
            c_b=t.c_b,
            c_s=t.c_s,
            kappa=t.kappa,
            gamma=t.gamma,
            w_max=t.w_max,
            alpha=t.alpha,
            rebalance_freq=t.rebalance_freq,
            cov_lookback=t.cov_lookback,
            rf_annual=t.rf_annual,
            turnover_tol=t.turnover_tol,
            boundary_tol=t.boundary_tol,
            cap_drift_buffer=t.cap_drift_buffer,
            volatility_scaled=t.volatility_scaled,
            short_circuit=t.short_circuit,
            log_forecast=t.log_forecast,
        )

    def replace(self, **changes: Any) -> BacktestParams:
        """Return a copy with the given fields replaced (validated by ``dataclasses``)."""
        return replace(self, **changes)

    @property
    def eta_b(self) -> float:
        """Buy-side penalty per unit of purchased weight, ``kappa * c_b`` (Eq. 4)."""
        return self.kappa * self.c_b

    @property
    def eta_s(self) -> float:
        """Sell-side penalty per unit of sold weight, ``kappa * c_s`` (Eq. 4)."""
        return self.kappa * self.c_s

    @property
    def no_trade_band(self) -> float:
        """Width of the Theorem 1 no-trade band, ``eta_b + eta_s`` (Eq. 6).

        Equals the physical round-trip cost when ``kappa == 1`` and widens linearly
        in ``kappa``, so a no-trade date at one scale stays a no-trade date at every
        larger scale.
        """
        return self.eta_b + self.eta_s

    def penalty_vectors(self, n_assets: int, vol_scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Per-asset buy and sell penalties.

        Parameters
        ----------
        n_assets
            Number of constituents.
        vol_scale
            Optional cross-sectionally normalised volatility multipliers of shape
            ``(n_assets,)`` used by the volatility-scaled extension (Section 3.7).
            ``None`` reproduces the uniform penalty.

        Returns
        -------
        tuple of ndarray
            ``(eta_b, eta_s)``, each of shape ``(n_assets,)``.
        """
        if vol_scale is None:
            return np.full(n_assets, self.eta_b), np.full(n_assets, self.eta_s)
        scale = np.asarray(vol_scale, dtype=float)
        return self.eta_b * scale, self.eta_s * scale

    def digest(self) -> str:
        """Stable 16-hex-character digest over every field; used as the cache key."""
        payload = repr([(f.name, getattr(self, f.name)) for f in fields(self)]).encode()
        return hashlib.sha256(payload).hexdigest()[:16]
