"""Pipeline wiring: the paths a stage writes must be the paths the next stage reads."""

from pathlib import Path

import pytest

from stillwar.__main__ import ALL_STAGES, STAGES
from stillwar.artifacts import MODEL_KEYS
from stillwar.config import Config, load_config


def test_every_stage_is_reachable() -> None:
    """``make all`` names only stages that exist, and every stage is in ``all``."""
    assert set(ALL_STAGES) <= set(STAGES)
    assert "train" in ALL_STAGES, "training must be part of the default chain"
    assert "baselines" in ALL_STAGES, "the baseline forecasts must have a producer"
    assert "tables" in ALL_STAGES
    assert "trials" not in ALL_STAGES, "the trial grid is deliberately opt-in"


def test_checkpoint_paths_match_between_writer_and_reader(repo_root: Path) -> None:
    """The checkpoint filenames ``train`` writes are the ones the tests read.

    The previous layout wrote ``logsig/logsig_seed_0.pt`` while the tests read
    ``logsig/msgn_seed_0.pt``, so a retrain produced a tree no test ever inspected.
    """
    from stillwar.train import train_all_models

    source = Path(train_all_models.__code__.co_filename).read_text(encoding="utf-8")
    assert 'f"{name}_seed_{seed}.pt"' in source

    tests = (repo_root / "tests" / "test_artifacts.py").read_text(encoding="utf-8")
    assert 'f"{model}_seed_{seed}.pt"' in tests


def test_forecast_artifact_names_are_consistent(repo_root: Path) -> None:
    """``train`` writes ``forecasts_<model>.npz`` for exactly the models ``simulate`` loads."""
    from stillwar.__main__ import stage_train

    source = Path(stage_train.__code__.co_filename).read_text(encoding="utf-8")
    assert 'f"forecasts_{key}.npz"' in source
    assert MODEL_KEYS == ("logsig", "logsig_inc", "mlp", "lstm")


def test_config_has_no_duplicated_defaults(repo_root: Path) -> None:
    """``portfolio.py`` must not carry literals that shadow ``configs/base.yaml``.

    The old ``get_param`` fallback duplicated fourteen defaults, which is how Table 2
    silently came to be governed by ``portfolio.py`` rather than by the configuration.
    """
    source = (repo_root / "src" / "stillwar" / "portfolio.py").read_text(encoding="utf-8")
    assert "get_param" not in source
    assert "**kwargs" not in source


def test_backtest_params_cover_every_trading_field(cfg: Config) -> None:
    """Every ``trading`` field reaches the backtest, with none left behind.

    A field present in the config but absent from ``BacktestParams`` would be silently
    ignored by the engine, which is how Table 2 once came to be governed by literals.
    """
    from dataclasses import fields

    from stillwar.params import BacktestParams

    names = {f.name for f in fields(BacktestParams)}
    trading = set(cfg.trading.model_dump())
    assert trading - names == set(), f"unmapped trading fields: {trading - names}"


def test_feature_digest_ignores_trading_overrides(cfg: Config) -> None:
    """Feature caching keys on data, splits and windows only."""
    assert cfg.feature_digest() == cfg.override(trading__kappa=1.0).feature_digest()
    assert cfg.feature_digest() != cfg.override(features__windows=(5, 20)).feature_digest()


def test_config_digest_changes_with_any_field(cfg: Config) -> None:
    """The config digest is sensitive to every section."""
    base = cfg.digest
    assert base != cfg.override(trading__kappa=1.0).digest
    assert base != cfg.override(train__max_epochs=1).digest
    assert base != cfg.override(reporting__panel_a_rebalances=93).digest


def test_unknown_config_key_is_rejected(repo_root: Path, tmp_path: Path) -> None:
    """A typo in the YAML fails at load time rather than being silently ignored."""
    import yaml

    raw = yaml.safe_load((repo_root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    raw["trading"]["not_a_real_knob"] = 1.0
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(Exception):
        load_config(path)
