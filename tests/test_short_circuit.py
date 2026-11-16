"""Short-circuit equivalence: the production hold shortcut must not change a decision."""

import numpy as np
import pandas as pd

from stillwater.backtest.engine import run_backtest
from stillwater.config import BacktestParams, Config
from stillwater.provenance.metadata import read_json
from tests.helpers import needs

ARTIFACT = "data/processed/exp_short_circuit_equivalence.json"


def _panel(n_days: int = 120, n_assets: int = 40, seed: int = 7) -> pd.DataFrame:
    """Synthetic price panel with enough history for the 20-day covariance window."""
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, (n_days, n_assets)), axis=0)
    return pd.DataFrame(
        prices,
        index=pd.bdate_range("2024-01-01", periods=n_days),
        columns=[f"A{i}" for i in range(n_assets)],
    )


def test_short_circuit_matches_full_solve(cfg: Config) -> None:
    """Disabling the shortcut changes neither the decision nor the realised path.

    The production engine skips the QP when Theorem 1 says the incumbent is optimal,
    so on a hold date ``solver_traded = False`` holds by construction. This test pins
    the shortcut to the full solve on a fixture that exercises both branches.
    """
    base = BacktestParams.from_config(cfg).replace(cov_lookback=20)
    panel = _panel()
    dates = list(panel.index[40:])

    forecasts = np.zeros((len(dates), panel.shape[1]))
    for cycle in range(0, len(dates), 3):
        forecasts[cycle, 0] = 0.06
        forecasts[cycle, 1] = -0.06

    short = run_backtest(panel, dates, base, forecasts)
    full = run_backtest(panel, dates, base.replace(short_circuit=False), forecasts)

    short_decisions = [row["solver_traded"] for row in short["kkt_logs"]]
    full_decisions = [row["solver_traded"] for row in full["kkt_logs"]]
    assert short_decisions == full_decisions
    assert 0 < sum(short_decisions) < len(short_decisions), "fixture must exercise both branches"
    np.testing.assert_allclose(short["daily_returns"], full["daily_returns"], atol=1e-12)
    np.testing.assert_allclose(short["executed_turnovers"], full["executed_turnovers"], atol=1e-12)


@needs(ARTIFACT)
def test_short_circuit_artifact(repo_root, cfg: Config) -> None:
    """The committed artifact proves the Section 2.6 claim on the real panel.

    It also cross-checks the interior counts against Table 3, so the two artifacts
    cannot disagree about how many dates were trades, holds, cap-drift or boundary.
    """
    _meta, results = read_json(repo_root / ARTIFACT)
    primary = results["primary"]

    assert primary["label"] == "LogSig|s0"
    assert primary["n_dates"] == 128
    assert primary["n_disagree"] == 0
    assert primary["agreement_rate"] == 1.0
    assert results["all_agree"] is True
    assert results["n_runs"] == len(results["all_runs"]) == 4 * len(cfg.seeds) + 4
    assert all(row["n_disagree"] == 0 for row in results["all_runs"])

    _meta3, table3 = read_json(repo_root / "data/processed/table3.json")
    assert primary["n_interior_trades"] == table3["trade_total"]
    assert primary["n_interior_holds"] == table3["hold_total"]
    assert primary["n_cap_drift"] == table3["cap_drift_count"]
    assert primary["n_boundary"] == table3["boundary_count"]

    # The primary run of Table 3 has no near-boundary case; the one disclosed in
    # Appendix A.4 is LogSig-Inc seed 4, where the optimal trade falls below the
    # 1e-4 execution tolerance and the solver holds while the test says trade.
    assert primary["kkt_solver_agreement_full_solve"] == 1.0
    near_boundary = [
        row["label"] for row in results["all_runs"] if row["kkt_solver_agreement_full_solve"] < 1.0
    ]
    assert near_boundary == ["LogSig-Inc|s4"]
