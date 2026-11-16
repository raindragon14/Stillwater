"""Portfolio-level aggregation: turn backtest results into reportable numbers.

The invariant of this module is that it returns **only numbers**. No percentages, no
``"±"``, no rounding, no file I/O. Formatting belongs to :mod:`stillwater.reporting.tables` and
:mod:`stillwater.reporting.figures`, and keeping the two apart is what lets
``tests/test_provenance.py`` regenerate a manuscript table and compare it
against the committed one cell by cell.

Every function here has a producer in the pipeline; nothing is computed for its own
sake.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from stillwater.backtest.engine import BacktestResult
from stillwater.backtest.stats import rank_ic, rank_ic_test
from stillwater.data.universe import SECTORS
from stillwater.optimization.kkt import KKTLog

__all__ = [
    "backtest_metrics",
    "declined_trade_test",
    "rank_ic_summary",
    "seed_summary",
    "trigger_concentration",
    "trigger_endpoints",
]


def backtest_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    """Headline performance numbers for one backtest.

    Parameters
    ----------
    result
        Output of :func:`stillwater.backtest.engine.run_backtest`.

    Returns
    -------
    dict
        ``cagr``, ``sharpe``, ``mdd``, ``turnover``, ``psi_kappa`` and
        ``kkt_agreement_rate``, all as raw fractions rather than percentages.
    """
    return {
        "cagr": float(result["net_cagr"]),
        "sharpe": float(result["sharpe"]),
        "mdd": float(result["mdd"]),
        "turnover": float(result["turnover"]),
        "psi_kappa": float(result["psi_kappa"]),
        "kkt_agreement_rate": float(result["kkt_agreement_rate"]),
    }


def seed_summary(values: Sequence[float]) -> dict[str, float]:
    """Mean and sample standard deviation across seeds.

    Parameters
    ----------
    values
        One value per seed.

    Returns
    -------
    dict
        ``mean`` and ``sd`` (``ddof=1``), plus ``n``. ``sd`` is ``0.0`` for a single
        seed, which is what the LSTM and GMV rows of Table 1 report.
    """
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def rank_ic_summary(
    forecasts: np.ndarray,
    targets: np.ndarray,
    rebalance_freq: int,
    effective_ratio: float,
) -> dict[str, float]:
    """Rank IC on the rebalance grid, with its overlap-corrected t-test.

    Parameters
    ----------
    forecasts, targets
        ``(T, N)`` forecast and realised-return arrays aligned on decision dates.
    rebalance_freq
        Stride of the rebalance grid; ``5`` samples every fifth decision date.
    effective_ratio
        ``1.0`` for the non-overlapping weekly series (``df = n_reb - 1``), ``0.2``
        for the overlapping daily series (``df = n / 5 - 1``).

    Returns
    -------
    dict
        ``mean``, ``std``, ``n``, ``n_eff``, ``df``, ``t`` and ``p``.
    """
    ics = rank_ic(forecasts, targets)[::rebalance_freq]
    return rank_ic_test(ics, effective_ratio=effective_ratio)


def declined_trade_test(
    result_penalized: BacktestResult,
    result_unpenalized: BacktestResult,
    rebalance_freq: int,
) -> dict[str, float]:
    """Price the rebalances the KKT band declined, against unpenalized execution.

    For every cycle on which the penalized run sent no order but the unpenalized run
    did, compare the realised net return of the two runs over that holding window.
    A negative mean means the declined trade would have lost money.

    Parameters
    ----------
    result_penalized
        Backtest at the primary penalty scale (``kappa = 5``).
    result_unpenalized
        The same configuration at ``kappa = 0``.
    rebalance_freq
        Trading days per cycle.

    Returns
    -------
    dict
        ``n_declined``, ``mean_bps``, ``total_bps``, ``t``, ``p_two_sided`` and
        ``p_one_sided``. The t-test is on the per-week differences in basis points.

    Notes
    -----
    JOURNAL Section 3.3 reports the two-sided variant of the current run
    (``n = 20``, ``t = -0.84``, ``p = 0.413``); both variants are returned so the
    table can state which it uses.
    """
    k5 = result_penalized["kkt_logs"]
    k0 = result_unpenalized["kkt_logs"]
    declined = [k["cycle"] for k in k5 if not k["solver_traded"] and k0[k["cycle"]]["solver_traded"]]

    diffs = []
    for cycle in declined:
        start = cycle * rebalance_freq
        end = min(len(result_unpenalized["daily_returns"]), (cycle + 1) * rebalance_freq)
        ret_0 = float(np.prod(1.0 + result_unpenalized["daily_returns"][start:end]) - 1.0)
        ret_5 = float(np.prod(1.0 + result_penalized["daily_returns"][start:end]) - 1.0)
        diffs.append((ret_0 - ret_5) * 10_000.0)

    out = {
        "n_declined": float(len(declined)),
        "mean_bps": float(np.mean(diffs)) if diffs else 0.0,
        "total_bps": float(np.sum(diffs)) if diffs else 0.0,
        "t": 0.0,
        "p_two_sided": 1.0,
        "p_one_sided": 0.5,
    }
    if len(diffs) > 1:
        res = stats.ttest_1samp(diffs, 0.0)
        out["t"] = float(res.statistic)
        out["p_two_sided"] = float(res.pvalue)
        out["p_one_sided"] = float(res.pvalue / 2.0) if res.statistic < 0 else float(1.0 - res.pvalue / 2.0)
    return out


def trigger_concentration(kkt_logs: Sequence[KKTLog], tickers: Sequence[str]) -> dict[str, Any]:
    """Which constituents and sectors set the extreme gradients on band-crossing dates.

    Feeds Figure 2 and the Herfindahl statistics of Section 3.5. Each crossing date
    contributes two triggers, one sell endpoint and one buy endpoint.

    Parameters
    ----------
    kkt_logs
        The ``kkt_logs`` field of a :class:`BacktestResult`.
    tickers
        Asset identifiers, aligned with the trigger indices.

    Returns
    -------
    dict
        ``crossing_count``, ``total_triggers``, ``herfindahl_constituents``,
        ``herfindahl_sectors``, ``sector_shares`` and ``top_10_constituents``.
    """
    counts = np.zeros(len(tickers))
    crossing = 0
    for row in kkt_logs:
        if row["kkt_should_trade"] and not row["is_cap_drift"]:
            crossing += 1
            if row["trigger_sell_idx"] >= 0:
                counts[row["trigger_sell_idx"]] += 1
            if row["trigger_buy_idx"] >= 0:
                counts[row["trigger_buy_idx"]] += 1

    total = float(np.sum(counts))
    shares = counts / total if total > 0 else np.full(len(counts), 1.0 / len(counts))

    sector_counts: dict[str, float] = {}
    for j, ticker in enumerate(tickers):
        sector = SECTORS.get(ticker, "Other")
        sector_counts[sector] = sector_counts.get(sector, 0.0) + counts[j]
    sector_shares = {k: v / total for k, v in sector_counts.items()} if total > 0 else {}

    top = np.argsort(counts)[::-1][:10]
    return {
        "crossing_count": crossing,
        "total_triggers": total,
        "herfindahl_constituents": float(np.sum(shares**2)),
        "herfindahl_sectors": float(np.sum(np.square(list(sector_shares.values())))) if sector_shares else 0.0,
        "sector_shares": sector_shares,
        "top_10_constituents": [
            {
                "ticker": tickers[i],
                "count": int(counts[i]),
                "share": float(shares[i]),
                "sector": SECTORS.get(tickers[i], "Other"),
            }
            for i in top
        ],
    }


def trigger_endpoints(kkt_logs: Sequence[KKTLog], tickers: Sequence[str]) -> dict[str, Any]:
    """Split the extreme-gradient endpoints into sell and buy triggers.

    Feeds Figure 2, whose caption promises that the bars "decompose extreme sell
    and buy gradient triggers". :func:`trigger_concentration` counts the two
    endpoints together, which is the right input for a Herfindahl index but the
    wrong one for a stacked bar: a sector can be a net seller of risk and a net
    buyer at the same time, and pooling the endpoints hides that.

    Parameters
    ----------
    kkt_logs
        The ``kkt_logs`` field of a :class:`BacktestResult`.
    tickers
        Asset identifiers, aligned with the trigger indices.

    Returns
    -------
    dict
        ``sell`` and ``buy`` counts per ticker, ``total``, ``crossing_count``,
        ``sell_total``, ``buy_total``, ``sector_sell`` and ``sector_buy``.
    """
    sell = np.zeros(len(tickers))
    buy = np.zeros(len(tickers))
    crossing = 0
    for row in kkt_logs:
        if not (row["kkt_should_trade"] and not row["is_cap_drift"]):
            continue
        crossing += 1
        if row["trigger_sell_idx"] >= 0:
            sell[row["trigger_sell_idx"]] += 1
        if row["trigger_buy_idx"] >= 0:
            buy[row["trigger_buy_idx"]] += 1

    sector_sell: dict[str, float] = {}
    sector_buy: dict[str, float] = {}
    for j, ticker in enumerate(tickers):
        sector = SECTORS.get(ticker, "Other")
        sector_sell[sector] = sector_sell.get(sector, 0.0) + sell[j]
        sector_buy[sector] = sector_buy.get(sector, 0.0) + buy[j]

    return {
        "crossing_count": crossing,
        "sell": sell,
        "buy": buy,
        "total": sell + buy,
        "sell_total": float(np.sum(sell)),
        "buy_total": float(np.sum(buy)),
        "sector_sell": sector_sell,
        "sector_buy": sector_buy,
    }
