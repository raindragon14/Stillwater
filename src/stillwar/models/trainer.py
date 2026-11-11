"""Deterministic CPU trainer with early stopping on the validation score ``S``.

Training is the second most expensive stage after the portfolio solver, so the
scheduler here is deliberately frugal:

* the learning-rate search already fits seed 0; that fit is reused rather than
  repeated, so a two-rate grid costs one extra fit instead of two;
* all four models share a single process pool, so ``torch`` is imported once per
  worker instead of once per model;
* forecasts for every model are exported from the same pass, which is what makes
  ``data/processed/forecasts_baselines.npz`` reproducible.

JOURNAL Section 2.4.
"""

import contextlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stillwar.backtest.stats import rank_ic
from stillwar.config import Config
from stillwar.features import Dataset
from stillwar.models.architectures import build_model
from stillwar.models.loss import composite_loss

MODELS: tuple[str, ...] = ("logsig", "logsig_inc", "mlp", "lstm")


@dataclass(frozen=True)
class FitResult:
    """Outcome of one deterministic training run.

    Attributes
    ----------
    seed
        Pinned seed.
    lr
        Learning rate used.
    best_epoch
        Epoch that minimised the validation score ``S``.
    best_score
        Value of ``S`` at ``best_epoch``.
    best_val_ic
        Mean daily validation rank IC at ``best_epoch``.
    weights
        Frozen state dict of the selected checkpoint.
    """

    seed: int
    lr: float
    best_epoch: int
    best_score: float
    best_val_ic: float
    weights: dict[str, torch.Tensor]


def _setup_deterministic_cpu(threads: int = 1) -> None:
    """Pin torch to ``threads`` and request deterministic kernels.

    A fixed thread count is part of the reproducibility contract: CPU reductions are
    not bit-identical across thread counts, so the count is pinned rather than left to
    the host. ``torch.use_deterministic_algorithms`` is best-effort because some
    kernels have no deterministic variant; failing to enable it must not abort a run.

    Notes
    -----
    This deliberately does **not** write ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``.
    Those variables are inherited by every child process spawned afterwards, so
    setting them here silently changed the floating-point path of later pipeline
    stages -- which is how a sensitivity-grid solve that converges normally came to
    fail only when training had run first in the same process. Thread pinning is done
    through :func:`torch.set_num_threads`, which is process-local.
    """
    torch.set_num_threads(threads)
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True)


def fit_single_seed(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    lr: float,
    cfg: Config,
) -> FitResult:
    """Train one model for one seed and early-stop on the validation score.

    The score is ``S = L_val - 0.5 * IC + 10 * (1 - IC) * 1{IC <= 0}``: the first two
    terms trade calibration against ranking, and the third rejects a checkpoint whose
    cross-sectional rank IC is not positive.

    Parameters
    ----------
    model_name
        One of :data:`MODELS`.
    x_train, y_train
        Training features ``(T, N, F)`` and targets ``(T, N)``.
    x_val, y_val
        Validation features and targets.
    seed
        Pinned seed for weight initialisation and batch order.
    lr
        Learning rate.
    cfg
        Configuration supplying the optimiser and early-stopping settings.

    Returns
    -------
    FitResult
        The checkpoint that minimised ``S``.

    Notes
    -----
    Each epoch shuffles all training dates and walks them in batches of
    ``cfg.train.batch_size`` dates, so every epoch is one full pass and every
    mini-batch is a sample of 32 dates without replacement.
    """
    _setup_deterministic_cpu(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(model_name, hidden=cfg.train.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.train.weight_decay)

    t_xt = torch.tensor(x_train, dtype=torch.float32)
    t_yt = torch.tensor(y_train, dtype=torch.float32)
    t_xv = torch.tensor(x_val, dtype=torch.float32)
    t_yv = torch.tensor(y_val, dtype=torch.float32)

    var_y = float(np.var(y_train))
    batch_size = cfg.train.batch_size
    gen = torch.Generator().manual_seed(seed)

    best_score, best_epoch, best_val_ic, best_weights = float("inf"), -1, -1.0, None
    patience = 0

    for epoch in range(1, cfg.train.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(t_xt), generator=gen)
        for i in range(0, len(t_xt), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = composite_loss(model(t_xt[idx]), t_yt[idx], var_y, cfg.train.lambda_rank)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_v = model(t_xv)
            val_loss = float(composite_loss(pred_v, t_yv, var_y, cfg.train.lambda_rank).item())
            val_ic = float(np.mean(rank_ic(pred_v.numpy(), y_val)))

        score = val_loss - 0.5 * val_ic + (10.0 * (1.0 - val_ic) if val_ic <= 0.0 else 0.0)

        if score < best_score:
            best_score, best_epoch, best_val_ic = score, epoch, val_ic
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.patience:
                break

    return FitResult(
        seed=seed,
        lr=lr,
        best_epoch=best_epoch,
        best_score=best_score,
        best_val_ic=best_val_ic,
        weights=best_weights if best_weights is not None else model.state_dict(),
    )


def _fit_task(args: tuple[Any, ...]) -> FitResult:
    """Picklable worker entry point for the shared training pool."""
    return fit_single_seed(*args)


def _split_inputs(ds: Dataset, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Training and validation tensors for one model."""
    x = ds.inputs(model_name)
    return x[ds.idx["train"]], ds.y[ds.idx["train"]], x[ds.idx["val"]], ds.y[ds.idx["val"]]


def select_best_lr(model_name: str, ds: Dataset, cfg: Config) -> tuple[float, FitResult | None]:
    """Pick the learning rate that minimises the validation score on seed 0.

    Parameters
    ----------
    model_name
        One of :data:`MODELS`.
    ds
        Dataset supplying the train and validation partitions.
    cfg
        Configuration supplying the per-model candidate grid.

    Returns
    -------
    tuple of (float, FitResult or None)
        The selected rate, and the seed-0 fit that produced it when the grid has more
        than one candidate. Returning the fit lets :func:`train_all_models` reuse it
        instead of training seed 0 twice.

    Notes
    -----
    Seed 0 is used both to select the rate and to enter the reported five-seed mean,
    so the seeds are not exchangeable and the reported dispersion is a lower bound.
    This is disclosed rather than silently corrected, because changing it would change
    every published number for a reason unrelated to the paper's argument.
    """
    candidates = cfg.train.lrs.get(model_name, (1e-3,))
    if len(candidates) == 1:
        return candidates[0], None

    xt, yt, xv, yv = _split_inputs(ds, model_name)
    best_lr, best_score, best_fit = candidates[0], float("inf"), None
    for lr in candidates:
        fit = fit_single_seed(model_name, xt, yt, xv, yv, seed=0, lr=lr, cfg=cfg)
        if fit.best_score < best_score:
            best_lr, best_score, best_fit = lr, fit.best_score, fit
    return best_lr, best_fit


def train_all_models(
    cfg: Config,
    ds: Dataset,
    ckpt_dir: str | Path = Path("models/checkpoints"),
) -> dict[str, dict[str, Any]]:
    """Train every model across every seed and return out-of-sample forecasts.

    Runs the learning-rate search serially (it is a handful of fits), then dispatches
    every remaining ``(model, seed)`` pair to a single process pool.

    Parameters
    ----------
    cfg
        Configuration supplying seeds, the LR grid and the worker count.
    ds
        Dataset supplying all partitions.
    ckpt_dir
        Root for checkpoints; each model gets ``<ckpt_dir>/<model>/<model>_seed_<s>.pt``.

    Returns
    -------
    dict
        ``{model: {"fits": {seed: FitResult}, "panel_a": ndarray, "panel_b": ndarray}}``
        where the arrays have shape ``(n_seeds, n_dates, n_assets)``.
    """
    selected = {name: select_best_lr(name, ds, cfg) for name in MODELS}

    tasks: list[tuple[Any, ...]] = []
    pending: list[tuple[str, int]] = []
    for name in MODELS:
        lr, seed0 = selected[name]
        xt, yt, xv, yv = _split_inputs(ds, name)
        for seed in cfg.seeds:
            if seed0 is not None and seed == seed0.seed:
                continue
            tasks.append((name, xt, yt, xv, yv, seed, lr, cfg))
            pending.append((name, seed))

    workers = min(len(cfg.seeds), cfg.workers.train)
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_fit_task, tasks))
    else:
        outcomes = [_fit_task(t) for t in tasks]

    fits: dict[str, dict[int, FitResult]] = {name: {} for name in MODELS}
    for (name, _seed), fit in zip(pending, outcomes, strict=True):
        fits[name][fit.seed] = fit
    for name in MODELS:
        seed0 = selected[name][1]
        if seed0 is not None:
            fits[name][seed0.seed] = seed0

    out: dict[str, dict[str, Any]] = {}
    for name in MODELS:
        lr = selected[name][0]
        save_path = Path(ckpt_dir) / name
        save_path.mkdir(parents=True, exist_ok=True)
        for seed in cfg.seeds:
            fit = fits[name][seed]
            torch.save(
                {
                    "model_state": fit.weights,
                    "best_epoch": fit.best_epoch,
                    "best_score": fit.best_score,
                    "best_val_ic": fit.best_val_ic,
                    "seed": seed,
                    "lr": lr,
                },
                save_path / f"{name}_seed_{seed}.pt",
            )
        out[name] = {
            "fits": fits[name],
            "lr": lr,
            "panel_a": _forecast(fits[name], name, ds, cfg, "test_a"),
            "panel_b": _forecast(fits[name], name, ds, cfg, "test_b"),
        }
    return out


def _forecast(
    fits: dict[int, FitResult],
    model_name: str,
    ds: Dataset,
    cfg: Config,
    partition: str,
) -> np.ndarray:
    """Run every seed's frozen weights over one partition."""
    x = torch.tensor(ds.inputs(model_name)[ds.idx[partition]], dtype=torch.float32)
    out = np.zeros((len(cfg.seeds), len(ds.idx[partition]), len(ds.tickers)), dtype=np.float32)
    for i, seed in enumerate(cfg.seeds):
        model = build_model(model_name, hidden=cfg.train.hidden_dim)
        model.load_state_dict(fits[seed].weights)
        model.eval()
        with torch.no_grad():
            out[i] = model(x).numpy()
    return out


def gate_activations(model_name: str, ds: Dataset, cfg: Config, fit: FitResult) -> np.ndarray:
    """Softmax gate activations for one fitted model over a partition.

    Feeds the ``omega`` statistics of Section 3.1, which previously had no producer.

    Parameters
    ----------
    model_name
        One of :data:`MODELS`; only the gated architectures expose gates.
    ds
        Dataset supplying the partition.
    cfg
        Configuration supplying the hidden width.
    fit
        Fitted weights.

    Returns
    -------
    ndarray
        ``(T, N, 3)`` gate activations, or an empty array for ungated models.
    """
    model = build_model(model_name, hidden=cfg.train.hidden_dim)
    if not hasattr(model, "gates"):
        return np.empty((0, 0, 0))
    model.load_state_dict(fit.weights)
    model.eval()
    x = torch.tensor(ds.inputs(model_name)[ds.idx["test_b"]], dtype=torch.float32)
    with torch.no_grad():
        return model.gates(x).numpy()
