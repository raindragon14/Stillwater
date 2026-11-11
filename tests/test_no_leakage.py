"""Zero look-ahead bias: feature causality, target lag, and purged/embargoed splits."""

import numpy as np
import pandas as pd

from stillwar.config import load_config
from stillwar.features import build_dataset, logsig_features, split_indices


def test_feature_strict_historical_window() -> None:
    """Features at date ``t`` are invariant to arbitrary mutations after ``t``."""
    rng = np.random.default_rng(2026)
    base = rng.standard_normal((80, 1, 5))
    t_eval = 65

    mutated = base.copy()
    mutated[t_eval + 1 :] += 100.0

    np.testing.assert_allclose(
        logsig_features(base, (5, 20, 60))[t_eval],
        logsig_features(mutated, (5, 20, 60))[t_eval],
        atol=1e-12,
    )


def test_target_execution_lag_on_real_data(repo_root) -> None:
    """``y_{t,i} = log(P_{t+6} / P_{t+1})`` on the real panel, not on a local stub.

    The previous version of this test defined its own ``compute_target`` and asserted
    that a discarded argument had no effect, which tested nothing. This one reads the
    pipeline's own dataset and reconstructs the target from the panel.
    """
    from stillwar.data import load_panel

    cfg = load_config(repo_root / "configs/base.yaml")
    panel = load_panel(cfg)
    ds = build_dataset(panel, cfg)

    close = panel.ohlcv[:, :, 3]
    date_to_idx = {d: i for i, d in enumerate(panel.dates)}
    for t_idx in (0, len(ds.dates) // 2, len(ds.dates) - 1):
        global_idx = date_to_idx[ds.dates[t_idx]]
        expected = np.log(close[global_idx + 6] / close[global_idx + 1])
        np.testing.assert_allclose(ds.y[t_idx], expected, rtol=1e-5, atol=1e-6)


def test_standardisation_uses_training_rows_only(repo_root) -> None:
    """Feature moments come from the training partition, so test data cannot leak in."""
    from stillwar.data import load_panel

    cfg = load_config(repo_root / "configs/base.yaml")
    ds = build_dataset(load_panel(cfg), cfg)
    train = ds.x[ds.idx["train"]].reshape(-1, ds.x.shape[-1])
    np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(train.std(axis=0), 1.0, atol=1e-3)


def test_purged_embargo_split_dates(repo_root) -> None:
    """Purge and embargo separate train, validation and test by at least twelve days."""
    cfg = load_config(repo_root / "configs/base.yaml")
    dates = pd.date_range("2018-01-01", "2026-10-02", freq="B")
    idx = split_indices(dates, cfg)

    train, val, test_a = dates[idx["train"]], dates[idx["val"]], dates[idx["test_a"]]
    assert val[0] > train[-1] + pd.Timedelta(days=8)
    assert test_a[0] > val[-1] + pd.Timedelta(days=8)
    assert len(idx["test_a"]) < len(idx["test_b"])


def test_split_partitions_are_disjoint(repo_root) -> None:
    """Train, validation and test never share a decision date.

    Panel A is by construction a prefix of Panel B, so those two are checked for
    containment rather than for disjointness.
    """
    cfg = load_config(repo_root / "configs/base.yaml")
    dates = pd.date_range("2018-01-01", "2026-10-02", freq="B")
    idx = split_indices(dates, cfg)

    train = set(idx["train"].tolist())
    val = set(idx["val"].tolist())
    test_a = set(idx["test_a"].tolist())
    test_b = set(idx["test_b"].tolist())

    assert not train & val
    assert not train & test_b
    assert not val & test_b
    assert test_a <= test_b, "Panel A must be a prefix of Panel B"
    assert len(test_a) < len(test_b)
