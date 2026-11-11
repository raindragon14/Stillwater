"""Content-addressed caching, so ``make all`` never recomputes what is already correct.

Two levels:

**In-process** -- ``functools.lru_cache`` on the panel and dataset builders, keyed on
the configuration sections that actually determine them.

**On-disk** -- every expensive stage records a provenance block under ``.cache/``.
A stage reruns only when its configuration digest, its code digest, or one of its
upstream artifact digests changes. Backtest results are cached individually, keyed on
the full content of their inputs, which collapses the ten byte-identical reruns of the
primary LogSig configuration into one.

The cache is an optimisation, never a source of truth: deleting ``.cache/`` changes
runtime, not results.
"""

import hashlib
import pickle
import shutil
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from stillwar.backtest.engine import BacktestResult, run_backtest
from stillwar.config import BacktestParams, Config
from stillwar.provenance.metadata import code_digest

CACHE_DIR = Path(".cache")
BACKTEST_DIR = CACHE_DIR / "backtests"
STAGE_DIR = CACHE_DIR / "stages"
DATASET_DIR = CACHE_DIR / "dataset"

_force = False


def set_force(value: bool) -> None:
    """Globally bypass every cache check (``stillwar <stage> --force``)."""
    global _force
    _force = bool(value)


def force() -> bool:
    """Whether cache checks are currently bypassed."""
    return _force


def _digest(*parts: bytes) -> str:
    """SHA-256 over the concatenation of ``parts``, truncated to 16 hex characters."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()[:16]


def prices_digest(prices: pd.DataFrame) -> str:
    """Digest of a price panel, covering both the index and the values."""
    return _digest(
        np.ascontiguousarray(prices.values, dtype=np.float64).tobytes(),
        repr(list(prices.index)).encode(),
        repr(list(prices.columns)).encode(),
    )


def forecasts_digest(forecasts: np.ndarray | None) -> str:
    """Digest of a forecast array; ``None`` (equal weight) has its own marker."""
    if forecasts is None:
        return "equal-weight"
    return _digest(np.ascontiguousarray(forecasts, dtype=np.float64).tobytes())


def backtest_key(prices: pd.DataFrame, params: BacktestParams, forecasts: np.ndarray | None) -> str:
    """Cache key for one backtest: prices, parameters and forecasts, nothing else."""
    return _digest(
        prices_digest(prices).encode(),
        params.digest().encode(),
        forecasts_digest(forecasts).encode(),
    )


def cached_backtest(
    prices: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    params: BacktestParams,
    forecasts: np.ndarray | None = None,
    force: bool | None = None,
) -> BacktestResult:
    """Memoised :func:`stillwar.backtest.engine.run_backtest`.

    The result depends only on ``prices``, ``params`` and ``forecasts``; ``dates`` is
    a deterministic slice of ``prices.index`` and is therefore not part of the key.
    A corrupted cache entry is detected by an unpickling failure and rebuilt.

    Parameters
    ----------
    prices
        Adjusted close panel indexed by trading date.
    dates
        Decision dates forwarded to :func:`run_backtest`.
    params
        Frozen backtest configuration.
    forecasts
        Forecast array or ``None`` for equal weight.
    force
        Bypass the cache for this call. ``None`` defers to the process-global flag set
        by :func:`set_force`. Worker processes must receive this explicitly, because
        the global does not survive ``spawn``.

    Returns
    -------
    BacktestResult
        Identical to calling :func:`run_backtest` directly.
    """
    if _force if force is None else force:
        return run_backtest(prices, dates, params, forecasts)

    key = backtest_key(prices, params, forecasts)
    path = BACKTEST_DIR / f"{key}.pkl"
    if path.exists():
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except (pickle.UnpicklingError, EOFError, AttributeError):
            path.unlink(missing_ok=True)

    result = run_backtest(prices, dates, params, forecasts)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return result


def _backtest_task(
    args: tuple[pd.DataFrame, Sequence[pd.Timestamp], BacktestParams, np.ndarray | None, bool],
) -> BacktestResult:
    """Picklable worker entry point for :func:`run_backtests`."""
    prices, dates, params, forecasts, force_flag = args
    return cached_backtest(prices, dates, params, forecasts, force=force_flag)


def run_backtests(
    tasks: Sequence[tuple[pd.DataFrame, Sequence[pd.Timestamp], BacktestParams, np.ndarray | None]],
    workers: int,
) -> list[BacktestResult]:
    """Run many backtests, in parallel when ``workers > 1``.

    Replaces the three near-identical process-pool wrappers that previously lived in
    ``experiments`` and ``reporting``. Results come back in submission order.

    Parameters
    ----------
    tasks
        ``(prices, dates, params, forecasts)`` tuples.
    workers
        Maximum concurrent processes; ``1`` runs in-process.

    Returns
    -------
    list of BacktestResult
        One result per task, in order.

    Notes
    -----
    The ``--force`` flag is forwarded explicitly into every task. Relying on the
    module-global would silently disable it, because worker processes are spawned with
    a fresh interpreter.
    """
    payload = [(*task, _force) for task in tasks]
    if workers <= 1 or len(payload) <= 1:
        return [_backtest_task(t) for t in payload]
    with ProcessPoolExecutor(max_workers=min(workers, len(payload))) as pool:
        return list(pool.map(_backtest_task, payload))


def is_fresh(stage: str, cfg: Config, depends_on: Sequence[str], inputs: dict[str, str] | None = None) -> bool:
    """Whether a stage's recorded provenance matches the current code, config and inputs.

    Parameters
    ----------
    stage
        Stage name, e.g. ``"train"`` or ``"simulate"``.
    cfg
        Current configuration.
    depends_on
        Modules whose source determines the stage's output.
    inputs
        Digests of upstream artifacts the stage consumes.

    Returns
    -------
    bool
        ``False`` when the cache is bypassed, when no record exists, or when any
        digest differs.
    """
    if _force:
        return False
    path = STAGE_DIR / f"{stage}.json"
    if not path.exists():
        return False

    import json

    record = json.loads(path.read_text(encoding="utf-8"))
    return (
        record.get("config_digest") == cfg.digest
        and record.get("code_digest") == code_digest(depends_on)
        and record.get("inputs", {}) == (inputs or {})
    )


def mark_done(stage: str, cfg: Config, depends_on: Sequence[str], inputs: dict[str, str] | None = None) -> None:
    """Record that a stage completed successfully for the current code, config and inputs."""
    import json

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "config_digest": cfg.digest,
        "code_digest": code_digest(depends_on),
        "inputs": inputs or {},
    }
    (STAGE_DIR / f"{stage}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def dataset_cache_path(feature_digest: str) -> Path:
    """Location of the persisted feature tensor for a given feature digest."""
    return DATASET_DIR / f"dataset_{feature_digest}.npz"


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file's bytes, or ``"missing"`` when it does not exist.

    Used to make a stage depend on the *content* of its upstream artifacts, not merely
    on the configuration. Without this, retraining would leave ``simulate`` marked
    fresh and the tables would silently describe the previous forecasts.
    """
    target = Path(path)
    if not target.exists():
        return "missing"
    h = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def clear() -> None:
    """Delete the entire cache (``make clean``)."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def stats() -> dict[str, int]:
    """Number and total size of cached backtests, for diagnostics."""
    if not BACKTEST_DIR.exists():
        return {"backtests": 0, "bytes": 0}
    files = list(BACKTEST_DIR.glob("*.pkl"))
    return {"backtests": len(files), "bytes": sum(f.stat().st_size for f in files)}
