"""Statistical primitives: Rank IC, HAC alpha, the Deflated Sharpe ratio, persistence.

These are the estimators of JOURNAL Section 2.10. They are deliberately free of any
knowledge about portfolios or files: :mod:`stillwar.backtest.metrics` composes them into the
reportable quantities, and :mod:`stillwar.reporting.tables` formats those.
"""

import numpy as np
from scipy import stats

__all__ = [
    "deflated_sharpe",
    "gate_weights",
    "lag_autocorr",
    "newey_west_alpha",
    "rank_ic",
    "rank_ic_test",
]


def rank_ic(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-date cross-sectional Spearman correlation.

    Parameters
    ----------
    pred, y
        ``(T, N)`` forecasts and realised returns.

    Returns
    -------
    ndarray
        ``(T,)`` rank correlations. Dates on which either cross-section is constant
        carry ``0.0`` rather than ``nan``, so downstream means stay well defined.
    """
    rp, ry = stats.rankdata(pred, axis=1), stats.rankdata(y, axis=1)
    rp, ry = rp - rp.mean(1, keepdims=True), ry - ry.mean(1, keepdims=True)
    den = np.sqrt((rp**2).sum(1) * (ry**2).sum(1))
    ic = np.divide((rp * ry).sum(1), den, out=np.zeros(len(pred)), where=den > 0)
    flat = (pred.std(1) < 1e-8) | (y.std(1) < 1e-8)
    return np.where(flat, 0.0, ic)


def rank_ic_test(ics: np.ndarray, effective_ratio: float) -> dict[str, float]:
    """One-sided t-test of ``E[IC] > 0`` with an overlap-corrected sample size.

    Consecutive five-day targets overlap, so a series sampled on every trading day
    carries only ``n / 5`` independent observations. A series sampled on the weekly
    rebalance dates does not overlap and keeps the full ``n``.

    Parameters
    ----------
    ics
        Per-date rank correlations.
    effective_ratio
        Multiplier applied to ``len(ics)``. Use ``1.0`` for non-overlapping weekly
        rebalance dates (``df = n - 1``) and ``0.2`` for overlapping daily dates
        (``df = n / 5 - 1``).

    Returns
    -------
    dict
        ``mean``, ``std`` (ddof=1), ``n``, ``n_eff``, ``df``, ``t`` and one-sided ``p``.

    Notes
    -----
    The ratio is a required argument on purpose. The previous default of ``0.2`` was
    silently applied to weekly series, which would have understated ``df`` by a factor
    of five relative to the values reported in Section 2.10.
    """
    n = len(ics)
    mean, sd = float(np.mean(ics)), float(np.std(ics, ddof=1)) if n > 1 else 0.0
    n_eff = n * effective_ratio
    df = max(1.0, n_eff - 1.0)
    if sd < 1e-12:
        return {"mean": mean, "std": sd, "n": n, "n_eff": n_eff, "df": df, "t": 0.0, "p": 0.5}
    t = mean / (sd / np.sqrt(n_eff))
    return {
        "mean": mean,
        "std": sd,
        "n": n,
        "n_eff": n_eff,
        "df": df,
        "t": float(t),
        "p": float(stats.t.sf(t, df)),
    }


def lag_autocorr(forecast: np.ndarray, step: int = 5) -> float:
    """Mean over assets of the lag-1 autocorrelation on a subsampled grid.

    Parameters
    ----------
    forecast
        ``(T, N)`` forecast array.
    step
        Subsampling stride; ``5`` samples the weekly rebalance grid.

    Returns
    -------
    float
        Mean autocorrelation across assets with non-degenerate series, or ``0.0``.
    """
    f = forecast[::step]
    a, b = f[:-1], f[1:]
    ok = (a.std(0) > 1e-8) & (b.std(0) > 1e-8)
    corr = [np.corrcoef(a[:, j], b[:, j])[0, 1] for j in np.flatnonzero(ok)]
    return float(np.nanmean(corr)) if corr else 0.0


def gate_weights(weights: np.ndarray) -> dict[str, float]:
    """Average softmax gate activation per time scale (Section 3.1).

    Parameters
    ----------
    weights
        ``(..., 3)`` gate activations, typically ``(T, N, 3)``.

    Returns
    -------
    dict
        ``omega_5``, ``omega_20`` and ``omega_60``.
    """
    mean_w = weights.mean(axis=tuple(range(weights.ndim - 1)))
    return {"omega_5": float(mean_w[0]), "omega_20": float(mean_w[1]), "omega_60": float(mean_w[2])}


def newey_west_alpha(
    strategy: np.ndarray,
    benchmark: np.ndarray,
    lags: int = 5,
    periods: float = 252.0,
) -> dict[str, float]:
    """Annualised alpha of ``strategy`` on ``benchmark`` with Bartlett-kernel HAC errors.

    Parameters
    ----------
    strategy, benchmark
        Daily net return series of equal length.
    lags
        Bartlett bandwidth; ``5`` matches the five-day holding period.
    periods
        Trading days per year used to annualise the intercept.

    Returns
    -------
    dict
        ``alpha`` (annualised) and ``t`` (HAC t-statistic). Both are ``0.0`` when the
        sample is too short to estimate, which callers must report as "not computed"
        rather than as a measured zero.
    """
    n = len(strategy)
    if n != len(benchmark) or n < lags + 5:
        return {"alpha": 0.0, "t": 0.0}

    x = np.column_stack([np.ones(n), benchmark])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ strategy)
    xe = x * (strategy - x @ beta)[:, None]
    s = xe.T @ xe
    for lag in range(1, lags + 1):
        gamma = xe[lag:].T @ xe[:-lag]
        s += (1.0 - lag / (lags + 1.0)) * (gamma + gamma.T)
    se = np.sqrt(max(1e-12, (xtx_inv @ s @ xtx_inv)[0, 0]))
    return {"alpha": float(beta[0] * periods), "t": float(beta[0] / se)}


def deflated_sharpe(sharpe: float, trials: np.ndarray, returns: np.ndarray) -> float:
    """Bailey and Lopez de Prado (2014) probability that the Sharpe beats the best trial.

    Parameters
    ----------
    sharpe
        Annualised Sharpe ratio of the strategy.
    trials
        Annualised Sharpe ratios of every configuration that was evaluated. Only the
        cross-trial variance enters the formula, but the array must be the real set of
        trials: a synthetic distribution makes the statistic meaningless.
    returns
        Daily net returns, used for the skewness and kurtosis adjustment.

    Returns
    -------
    float
        Probability in ``[0, 1]``; ``0.5`` when there are fewer than two trials or
        fewer than ten observations.
    """
    m, t = len(trials), len(returns)
    if m < 2 or t < 10:
        return 0.5

    euler = 0.5772156649
    z = (1 - euler) * stats.norm.ppf(1 - 1 / m) + euler * stats.norm.ppf(1 - 1 / (m * np.e))
    sr_max = np.sqrt(max(1e-8, np.var(trials, ddof=1))) * z
    sr = sharpe / np.sqrt(252.0)
    skew, kurt = stats.skew(returns), stats.kurtosis(returns, fisher=False)
    se = np.sqrt(max(1e-12, (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (t - 1))) * np.sqrt(252.0)
    return float(stats.norm.cdf((sharpe - sr_max) / se))
