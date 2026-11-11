"""Read and write the forecast arrays that link the training stage to the simulation stage.

These files are the contract between ``train`` / ``baselines`` and ``simulate``. They
were previously the weakest link in the pipeline: ``forecasts_baselines.npz`` was read
by ``simulate`` but written by no code in the repository, so a clean checkout could not
run. Every array written here has a producer named in the same module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from stillwater.config import Config

if TYPE_CHECKING:
    from stillwater.features import Dataset

__all__ = [
    "BASELINE_KEYS",
    "load_all",
    "load_forecasts",
    "save_baselines",
    "save_forecasts",
]

MODEL_KEYS: tuple[str, ...] = ("logsig", "logsig_inc", "mlp", "lstm")
BASELINE_KEYS: tuple[str, ...] = ("hist_mv", "gmv")


def _dates(ds: Dataset, partition: str) -> list[str]:
    """Decision dates of a partition as ISO strings."""
    return [d.strftime("%Y-%m-%d") for d in ds.dates[ds.idx[partition]]]


def save_forecasts(
    path: str | Path,
    panel_a: np.ndarray,
    panel_b: np.ndarray,
    ds: Dataset,
    seeds: tuple[int, ...],
) -> None:
    """Persist one model's Panel A and Panel B forecasts.

    Parameters
    ----------
    path
        Destination ``.npz``.
    panel_a, panel_b
        ``(n_seeds, n_dates, n_assets)`` forecast arrays.
    ds
        Dataset supplying tickers and decision dates.
    seeds
        Seed order along the first axis.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        panel_a=panel_a,
        panel_b=panel_b,
        panel_a_mean=panel_a.mean(axis=0),
        panel_b_mean=panel_b.mean(axis=0),
        tickers=np.array(ds.tickers),
        dates_a=np.array(_dates(ds, "test_a")),
        dates_b=np.array(_dates(ds, "test_b")),
        seeds=np.array(seeds),
    )


def load_forecasts(path: str | Path) -> dict[str, Any]:
    """Load a file written by :func:`save_forecasts`.

    Raises
    ------
    FileNotFoundError
        If the file is missing, naming the stage that produces it.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"{target} is missing; run `make train` first")
    with np.load(target) as z:
        return {
            "panel_a": z["panel_a"],
            "panel_b": z["panel_b"],
            "panel_a_mean": z["panel_a_mean"],
            "panel_b_mean": z["panel_b_mean"],
            "tickers": [str(t) for t in z["tickers"]],
            "dates_a": [str(d) for d in z["dates_a"]],
            "dates_b": [str(d) for d in z["dates_b"]],
            "seeds": [int(s) for s in z["seeds"]],
        }


def save_baselines(path: str | Path, baselines: dict[str, np.ndarray], ds: Dataset) -> None:
    """Persist the non-learned benchmark forecasts (Hist-MV and GMV).

    Parameters
    ----------
    path
        Destination ``.npz``.
    baselines
        ``{"hist_mv": (n_dates, n_assets), "gmv": (n_dates, n_assets)}`` for Panel B.
    ds
        Dataset supplying tickers and decision dates.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        hist_mv=baselines["hist_mv"],
        gmv=baselines["gmv"],
        tickers=np.array(ds.tickers),
        dates_b=np.array(_dates(ds, "test_b")),
    )


def load_baselines(path: str | Path) -> dict[str, np.ndarray]:
    """Load a file written by :func:`save_baselines`.

    Raises
    ------
    FileNotFoundError
        If the file is missing, naming the stage that produces it.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"{target} is missing; run `make baselines` first")
    with np.load(target) as z:
        return {key: z[key] for key in BASELINE_KEYS}


def load_all(
    cfg: Config,
    processed: str | Path = "data/processed",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load every forecast array the simulation stage needs.

    Parameters
    ----------
    cfg
        Configuration; unused directly but keeps the signature uniform with the other
        loaders.
    processed
        Directory holding the ``.npz`` artifacts.

    Returns
    -------
    tuple of (dict, dict)
        ``{model_key: (n_seeds, n_dates, n_assets)}`` and
        ``{"hist_mv": ..., "gmv": ...}``.
    """
    root = Path(processed)
    forecasts = {key: load_forecasts(root / f"forecasts_{key}.npz")["panel_b"] for key in MODEL_KEYS}
    return forecasts, load_baselines(root / "forecasts_baselines.npz")
