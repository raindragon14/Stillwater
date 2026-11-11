"""The Deflated Sharpe trial grid: every configuration the manuscript searched over.

The Deflated Sharpe ratio of Bailey and Lopez de Prado (2014) deflates a strategy's
Sharpe by the best Sharpe expected from ``M`` unskilled trials. That correction is only
meaningful if ``M`` and the cross-trial dispersion come from configurations that were
actually evaluated. This module enumerates them and persists the resulting Sharpe ratios
as a versioned artifact, so ``make all`` can verify the digest instead of recomputing
twenty minutes of backtests.

JOURNAL Section 2.10: ``M = 1,225`` = 1,200 grid + 20 non-inclusive-cost runs + 5
classical strategies. The ten EWMA half-life trials of an earlier draft are dropped:
Section 3.4 contains no half-life analysis, and the search that produced them terminated
at its boundary and emitted a "selection share" greater than one.
"""

from pathlib import Path
from typing import Any

import numpy as np

from stillwar.config import BacktestParams, Config
from stillwar.data import load_panel
from stillwar.features import load_dataset
from stillwar.provenance.metadata import artifact_metadata, cache, write_json

__all__ = ["N_TRIALS", "load_trials", "run_trial_grid"]

GAMMAS: tuple[float, ...] = (1.0, 2.5, 5.0, 10.0)
KAPPAS: tuple[float, ...] = (0.0, 1.0, 2.5, 5.0, 10.0)
COST_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0)
NON_INCLUSIVE = {"c_b": 0.001933, "c_s": 0.002933}

N_TRIALS = 4 * 5 * len(GAMMAS) * len(KAPPAS) * len(COST_MULTIPLIERS) + 20 + 5

_DEPENDS_ON = ("config", "params", "optimization", "backtest", "provenance", "features", "data")


def run_trial_grid(
    cfg: Config,
    forecasts: dict[str, np.ndarray],
    out_path: str | Path = "data/processed/trials.json",
) -> dict[str, Any]:
    """Evaluate every configuration behind the Deflated Sharpe ratio.

    Parameters
    ----------
    cfg
        Configuration.
    forecasts
        ``{model: (n_seeds, n_dates, n_assets)}`` arrays for ``logsig``, ``logsig_inc``,
        ``mlp`` and ``lstm``.
    out_path
        Destination JSON.

    Returns
    -------
    dict
        ``sharpes`` (list of annualised Sharpe ratios, length :data:`N_TRIALS`) and
        ``labels`` (matching human-readable identifiers).

    Notes
    -----
    The grid is 1,200 backtests plus 25, so this is the single most expensive target in
    the repository. It is deliberately excluded from ``make all``; run ``make trials``
    once and commit the artifact.
    """
    ds = load_dataset(cfg)
    panel = load_panel(cfg).close
    dates_b = list(ds.dates[ds.idx["test_b"]])
    base = BacktestParams.from_config(cfg)
    workers = min(len(cfg.seeds), cfg.workers.backtest)
    n_seeds = len(cfg.seeds)

    tasks: list[tuple[Any, ...]] = []
    labels: list[str] = []

    for model, fc in forecasts.items():
        for seed in range(n_seeds):
            for gamma in GAMMAS:
                for kappa in KAPPAS:
                    for mult in COST_MULTIPLIERS:
                        params = base.replace(
                            gamma=gamma,
                            kappa=kappa,
                            c_b=base.c_b * mult,
                            c_s=base.c_s * mult,
                        )
                        tasks.append((panel, dates_b, params, fc[seed]))
                        labels.append(f"{model}|s{seed}|g{gamma}|k{kappa}|m{mult}")

    for model, fc in forecasts.items():
        for seed in range(n_seeds):
            params = base.replace(c_b=NON_INCLUSIVE["c_b"], c_s=NON_INCLUSIVE["c_s"])
            tasks.append((panel, dates_b, params, fc[seed]))
            labels.append(f"{model}|s{seed}|non_inclusive")

    hist = ds.historical_forecast(base.alpha)
    gmv = np.zeros_like(hist)
    classical = [
        ("hist_mv_penalized", base, hist, True),
        ("hist_mv_unpenalized", base.replace(kappa=0.0), hist, True),
        ("gmv_penalized", base, gmv, True),
        ("gmv_unpenalized", base.replace(kappa=0.0), gmv, True),
        ("equal_weight", base, None, True),
    ]
    for name, params, fc, _ in classical:
        tasks.append((panel, dates_b, params, fc))
        labels.append(name)

    results = cache.run_backtests(tasks, workers)
    sharpes = [float(r["sharpe"]) for r in results]
    if len(sharpes) != N_TRIALS:
        raise RuntimeError(f"expected {N_TRIALS} trials, produced {len(sharpes)}")

    payload = {"sharpes": sharpes, "labels": labels, "n_trials": len(sharpes)}
    write_json(out_path, artifact_metadata(cfg, _DEPENDS_ON), payload)
    return payload


def load_trials(path: str | Path = "data/processed/trials.json") -> np.ndarray:
    """Load the persisted trial Sharpe ratios.

    Parameters
    ----------
    path
        Artifact written by :func:`run_trial_grid`.

    Returns
    -------
    ndarray
        Annualised Sharpe ratio of every evaluated configuration.

    Raises
    ------
    FileNotFoundError
        If the artifact is absent, with the command needed to produce it.
    """
    from stillwar.provenance.metadata import read_json

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} is missing; run `make trials` once (approximately 20 minutes) "
            "and commit the result"
        )
    _metadata, results = read_json(path)
    return np.asarray(results["sharpes"], dtype=float)
