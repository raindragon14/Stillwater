"""Cache correctness: keys, invalidation, and the guarantee that caching changes no result."""

import numpy as np
import pandas as pd
import pytest

from stillwar import cache
from stillwar.params import BacktestParams


@pytest.fixture(autouse=True)
def _reset_force():
    """Restore the global cache flag after every test."""
    yield
    cache.set_force(False)


def _panel(n_days: int = 40, n_assets: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, (n_days, n_assets)), axis=0)
    return pd.DataFrame(prices, index=pd.bdate_range("2024-01-01", periods=n_days), columns=list("ABC"))


def test_backtest_key_sensitivity(cfg) -> None:
    """The key changes with prices, parameters and forecasts, and with nothing else."""
    params = BacktestParams.from_config(cfg)
    panel = _panel()
    forecasts = np.zeros((40, 3))

    base = cache.backtest_key(panel, params, forecasts)
    assert base == cache.backtest_key(panel, params, forecasts)
    assert base != cache.backtest_key(panel, params.replace(kappa=1.0), forecasts)
    assert base != cache.backtest_key(panel, params, forecasts + 1.0)
    assert base != cache.backtest_key(panel * 1.01, params, forecasts)
    assert base != cache.backtest_key(panel, params, None)


def test_cached_backtest_is_transparent(tmp_path, monkeypatch, cfg) -> None:
    """A cached run returns exactly what an uncached run returns."""
    monkeypatch.setattr(cache, "BACKTEST_DIR", tmp_path / "backtests")
    params = BacktestParams.from_config(cfg).replace(cov_lookback=10, rebalance_freq=5)
    panel = _panel(60)
    dates = list(panel.index[20:])

    cache.set_force(True)
    direct = cache.cached_backtest(panel, dates, params, None)
    cache.set_force(False)
    cached = cache.cached_backtest(panel, dates, params, None)

    np.testing.assert_allclose(direct["daily_returns"], cached["daily_returns"])
    assert direct["kkt_logs"] == cached["kkt_logs"]


def test_corrupt_cache_entry_is_rebuilt(tmp_path, monkeypatch, cfg) -> None:
    """A truncated pickle is detected and replaced rather than raising."""
    monkeypatch.setattr(cache, "BACKTEST_DIR", tmp_path / "backtests")
    params = BacktestParams.from_config(cfg).replace(cov_lookback=10)
    panel = _panel(60)
    dates = list(panel.index[20:])

    first = cache.cached_backtest(panel, dates, params, None)
    key = cache.backtest_key(panel, params, None)
    (cache.BACKTEST_DIR / f"{key}.pkl").write_bytes(b"not a pickle")

    rebuilt = cache.cached_backtest(panel, dates, params, None)
    np.testing.assert_allclose(first["daily_returns"], rebuilt["daily_returns"])


def test_stage_freshness_tracks_config_and_code(tmp_path, monkeypatch, cfg) -> None:
    """A stage is fresh only while its config and code digests match."""
    monkeypatch.setattr(cache, "STAGE_DIR", tmp_path / "stages")
    deps = ("config", "params")

    assert not cache.is_fresh("demo", cfg, deps)
    cache.mark_done("demo", cfg, deps)
    assert cache.is_fresh("demo", cfg, deps)
    assert not cache.is_fresh("demo", cfg.override(trading__kappa=1.0), deps)
    assert not cache.is_fresh("demo", cfg, deps, inputs={"upstream": "different"})


def test_force_bypasses_freshness(tmp_path, monkeypatch, cfg) -> None:
    """``--force`` makes every stage stale."""
    monkeypatch.setattr(cache, "STAGE_DIR", tmp_path / "stages")
    cache.mark_done("demo", cfg, ("config",))
    assert cache.is_fresh("demo", cfg, ("config",))
    cache.set_force(True)
    assert not cache.is_fresh("demo", cfg, ("config",))
