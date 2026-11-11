"""Checkpoints and forecast arrays: the contract between ``train`` and ``simulate``."""

from pathlib import Path

import numpy as np
import pytest
import torch

from stillwar.artifacts import MODEL_KEYS
from stillwar.config import load_config
from stillwar.models import MSGN, LogSigInc, LSTMBaseline, MLPBaseline
from tests.helpers import needs

EXPECTED_PARAMS = {"logsig": 923, "logsig_inc": 353, "mlp": 1505, "lstm": 4513}
BUILDERS = {
    "logsig": lambda: MSGN(hidden_dim=16),
    "logsig_inc": lambda: LogSigInc(hidden_dim=16),
    "mlp": lambda: MLPBaseline(in_features=45, hidden_dim=32),
    "lstm": lambda: LSTMBaseline(input_dim=1, hidden_dim=32),
}


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_parameter_counts(model: str) -> None:
    """Every architecture has exactly the parameter count reported in the manuscript."""
    n = sum(p.numel() for p in BUILDERS[model]().parameters() if p.requires_grad)
    assert n == EXPECTED_PARAMS[model]


@pytest.mark.parametrize("model", MODEL_KEYS)
@needs("models/checkpoints/logsig/logsig_seed_0.pt")
def test_checkpoints_load(model: str, repo_root: Path) -> None:
    """Every seed checkpoint exists, loads, and records the training metadata."""
    cfg = load_config(repo_root / "configs/base.yaml")
    ckpt_dir = repo_root / "models/checkpoints" / model
    for seed in cfg.seeds:
        path = ckpt_dir / f"{model}_seed_{seed}.pt"
        assert path.exists(), f"missing checkpoint {path}"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("model_state", "best_epoch", "best_score", "best_val_ic", "seed", "lr"):
            assert key in payload, f"{path} lacks {key}"
        assert payload["seed"] == seed
        BUILDERS[model]().load_state_dict(payload["model_state"])


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_forecast_arrays(model: str, repo_root: Path) -> None:
    """Forecast tensors have the documented shape, no NaNs, and match the dataset."""
    path = repo_root / "data/processed" / f"forecasts_{model}.npz"
    if not path.exists():
        pytest.skip(f"missing {path}; run `make all` first")

    cfg = load_config(repo_root / "configs/base.yaml")
    with np.load(path) as z:
        assert z["panel_a"].shape[0] == len(cfg.seeds)
        assert z["panel_b"].shape[0] == len(cfg.seeds)
        assert z["panel_a"].shape[2] == cfg.data.n_stocks
        assert z["panel_b"].shape[2] == cfg.data.n_stocks
        assert not np.isnan(z["panel_b"]).any()
        assert len(z["seeds"]) == len(cfg.seeds)


@needs("data/processed/forecasts_baselines.npz")
def test_baseline_forecasts(repo_root: Path) -> None:
    """Hist-MV and GMV arrays exist, are finite, and GMV is identically zero."""
    with np.load(repo_root / "data/processed/forecasts_baselines.npz") as z:
        assert z["hist_mv"].ndim == 2
        assert not np.isnan(z["hist_mv"]).any()
        assert np.all(z["gmv"] == 0.0)


@needs("data/processed/forecasts_baselines.npz")
def test_hist_mv_matches_its_formula(repo_root: Path) -> None:
    """The stored Hist-MV array equals the formula in :meth:`Dataset.historical_forecast`."""
    from stillwar.features import load_dataset

    cfg = load_config(repo_root / "configs/base.yaml")
    ds = load_dataset(cfg)
    expected = ds.historical_forecast(cfg.trading.alpha)[ds.idx["test_b"]]
    with np.load(repo_root / "data/processed/forecasts_baselines.npz") as z:
        np.testing.assert_allclose(z["hist_mv"], expected, rtol=1e-5, atol=1e-6)
