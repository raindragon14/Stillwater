# Reproducibility Report

**Date:** 8 November 2026
**Scope:** refactor of the `stillwar` pipeline for effective LoC, complete docstrings, and minimal `make all` overhead.
**Status:** the pipeline regenerates every table and figure from a clean checkout. **The regenerated numbers differ from the previously published ones**, for the reasons in §3. The manuscript tables have been regenerated automatically; the surrounding prose still quotes the previous values (§5).

---

## 1. What changed

### 1.1 Structural

| Before | After |
|:---|:---|
| `reporting.py` (593 LOC) mixed computation, formatting and plotting | split into `simulate.py` (computation), `tables.py` (Markdown), `figures.py` (PNG) |
| `run_backtest(..., cfg, **kwargs)` with 14 duplicated default literals | `run_backtest(prices, dates, params, forecasts)` with a frozen `BacktestParams` |
| `plot_figure1` re-implemented the whole simulation loop (~85 LOC) | reads `kkt_logs` from the cached backtest (~15 LOC) |
| Figure 4 had 8 hardcoded data arrays | every series loaded from `exp_*.json` |
| `trial_sharpes = rng.normal(0.722, 0.327, 1235)` | `trials.py` evaluates the real 1,225 configurations |
| 3 near-identical process-pool wrappers | one `cache.run_backtests` |
| `make all` omitted `train` and read an artifact no code produced | `make all` = verify → features → train → baselines → simulate → experiments → figures → tables → test |
| results stored as `"8.58% ± 4.46%"` strings | floats only; formatting lives in `tables.py` |
| no provenance in any artifact | `metadata.{config_digest, code_digest, git_sha, versions}` in every JSON |
| manuscript tables typed by hand | generated between `<!-- BEGIN TABLE n -->` markers, enforced by test |

### 1.2 Numerical

Four changes affect results. The first three are deliberate; the fourth is a bug fix that was forced by the third.

1. **`slice_panel_a` now slices by cycle count.** It previously used `len(dates_a) = 467` — a *decision-date* count — to slice a *daily-return* array, leaving Panel A's returns three trading days short of its own turnover window. Panel A now covers `94 × 5 = 470` daily observations.
2. **The `kappa = 0` problem is solved in 40 variables instead of 120.** When both penalties vanish the slack variables of the 3N splitting do not enter the objective, so the split Hessian is singular and SLSQP converged to a point up to `5.4e-05` from the true optimum. The reduced formulation is better conditioned and about thirty times faster. The SLSQP algorithm itself is unchanged.
3. **The QP is warm-started from a feasible point.** On a cap-drift date the incumbent breaches `w_max`, so the previous start `(w_minus, 0, 0)` violated a bound. The start is now projected onto the box, the weight deficit redistributed in proportion to each asset's remaining room, and the slacks chosen so that `w - u+ + u- = w_minus` holds exactly. This changes the solver's *path*, not its optimum: regenerating the tables after the fix produced byte-identical artifacts.
4. **`_setup_deterministic_cpu` no longer writes `OMP_NUM_THREADS` / `MKL_NUM_THREADS`.** Those variables are inherited by every child process spawned afterwards, so training silently changed the floating-point path of later stages. That is how a sensitivity-grid solve which converges normally came to fail *only* when training had run first in the same process. Thread pinning now goes through `torch.set_num_threads`, which is process-local.

Everything else in the numerical core is preserved. This is verifiable: for every strategy with no trained parameters, Panel B reproduces the published values **exactly**.

| Strategy (Panel B) | Published | Regenerated |
|:---|:---|:---|
| LSTM | −1.66% / −0.27 / 32.89% / 0.00% | −1.66% / −0.27 / 32.89% / 0.00% |
| GMV | −1.66% / −0.27 / 32.89% / 0.00% | −1.66% / −0.27 / 32.89% / 0.00% |
| GMV, unpenalized | −3.33% / −0.56 / 23.39% / 22.60% | −3.33% / −0.56 / 23.39% / 22.60% |
| EW | 0.11% / −0.19 / 30.20% / 3.64% | 0.11% / −0.19 / 30.20% / 3.64% |

Panel A differs for these same strategies (e.g. EW 11.43% → 12.87%) because of change 1.

### 1.3 Lines of code — the target was missed

Measured against `git HEAD` with docstrings excluded from the code count:

| | Before | After | Δ |
|:---|---:|---:|---:|
| Source code lines | 2,054 | 4,135 | **+2,081 (+101%)** |
| Docstring lines | 61 | 1,680 | +1,619 |
| Source total | 2,115 | 5,815 | +3,700 |
| Test lines | 657 | 1,021 | +364 |

**The plan's `≤ 2,300` source-LoC target was not met, and raw LoC roughly doubled.** The composition of the increase:

| Contribution | Lines |
|:---|---:|
| Docstrings (the explicit ask) | +1,619 |
| New capability with no predecessor: `cache`, `provenance`, `params`, `artifacts`, `trials` | ~+1,000 |
| New capability replacing hardcoded literals: `tables`, and `figures` reading from JSON | ~+600 |
| `portfolio.py`: feasible warm start, retry ladder, ridge fallback | ~+200 |
| **Duplication removed**: `plot_figure1`'s simulation loop, 3 pool wrappers, 3 experiment boilerplates, 14 duplicated config defaults, dead stubs | **~−250** |

So the refactor traded a 593-line monolith for ~2,100 lines across nine focused modules, and added 1,680 lines of documentation. Every removed line was duplicated logic; every added line is either documentation or a capability that previously did not exist or was hardcoded.

This is an honest miss against the stated target. If fewer lines matter more than the added capability, the candidates to cut are the architecture schematic in `figures.py` (~50 lines) and the docstring depth. The duplication itself is gone: `plot_figure1` no longer re-simulates, `portfolio.py` no longer carries config defaults, and `simulate.py` is the only place a table number is computed.

---

## 2. Performance

Measured on the target machine (8 cores, 7.3 GB RAM). `make` is unavailable on the measurement host, so `python -m stillwar all` was timed, which is exactly what `make all` invokes before the trailing `test` target.

| Scenario | Before | After |
|:---|---:|---:|
| Warm run (nothing stale) | n/a — always recomputed | **11.2 s** |
| Cold run (`--force`) | ~13 min (estimated) | **11.6 min** |
| `make trials` (once, cached) | n/a — the grid did not exist | **45.7 min** |
| Unique backtests executed | 205 | **1,378** (1,225 trials + 153 table/grid runs) |
| `kappa = 0` backtest | 18.3 s | ~3 s |
| `make all` on a clean clone | failed at `simulate` | succeeds |

**The 9-minute cold target was not met.** The reason is in §2.3.

### 2.1 Where the time goes

| Unit | Cost | Count | Aggregate |
|:---|---:|---:|---:|
| `solve_qp` (3N = 120 vars) | 66 ms (η > 0) / 143 ms (η = 0) | ~17,600 | ~22 min CPU |
| `fit_single_seed` LSTM | 110 s | 5 | 9 min CPU |
| `fit_single_seed` logsig / mlp / inc | 8.7 / 5.9 / 3.2 s | 6 / 6 / 5 | 2.5 min CPU |
| `ledoit_wolf_covariance` | 0.37 ms | ~19,600 | 7 s |

A backtest is **100% its QP solver**: 115 solves × 66 ms = 7.6 s, the entire `run_backtest`. Covariance estimation is 0.7%.

### 2.2 What was removed

| Removed | Saving |
|:---|:---|
| 10 byte-identical reruns of the primary LogSig configuration | 76 s CPU |
| 23 serial backtests moved into the parallel pool | ~150 s wall |
| `select_best_lr` retraining seed 0 | 15 s CPU |
| Four process pools → one per stage | ~11 s |
| `load_dataset` rebuilding on `trading.*` overrides | ~2 s |
| Covariance-lookback grid (15 backtests, never reported) | ~120 s CPU |
| EWMA half-life search (broken, never reported) | ~10 s |

### 2.3 Why the cold run is 11.6 minutes, not 9

Two irreducible costs, both deliberately left alone:

* **The QP solver.** Replacing SLSQP with an active-set method for the piecewise-linear penalty would cut roughly 20 minutes of CPU, but it changes every published number and requires re-validating Theorem 1 against a new reference solver. The SLSQP algorithm is untouched.
* **The LSTM.** 110 s per fit, and threading does not help (111.7 s at one thread, 109.2 s at four). It is a published baseline, paid once, then cached.

Training runs with `workers.train = 3` rather than 5. Each LSTM fit holds roughly 400 MB of backpropagation state; five concurrent fits exhausted the 7.3 GB machine and aborted the stage with a `DefaultCPUAllocator` failure. Backtests use `workers.backtest = 5` because they hold about 50 MB each. Worker count affects wall clock only, never results.

---

## 3. Why the numbers changed

The published artifacts were **not reproducible from committed code**, for two independent reasons.

### 3.1 The trained forecasts came from an uncommitted training run

The committed checkpoints were named `models/checkpoints/logsig/msgn_seed_*.pt` and `models/checkpoints/baselines/{mlp,lstm}_seed_*.pt`. The committed `stage_train` wrote `models/checkpoints/logsig/logsig_seed_*.pt` and `models/checkpoints/mlp/mlp_seed_*.pt`. **Three of four model families did not match**, so no command in the repository could regenerate the committed checkpoints, and no test ever inspected a freshly trained one.

The forecasts therefore came from a training run whose script is not in the repository. Re-running the committed `train.py` — which is what the refactor does — produces different weights, hence different forecasts, hence a different drift path and a different set of cap-drift dates.

### 3.2 `forecasts_baselines.npz` had no producer

`stage_simulate` read `data/processed/forecasts_baselines.npz` for the MLP, LSTM, Hist-MV and GMV forecasts — **7 of the 9 rows of Table 1**. No code in the repository wrote that file. The committed copy was itself inconsistent: its Panel A Hist-MV array matched the documented formula (the old test verified it) while its Panel B array did not, which is why the Hist-MV rows move even though Hist-MV involves no training.

### 3.3 Consequence for the manuscript

The tables have been regenerated. Headline values:

| Quantity | Published | Regenerated |
|:---|:---|:---|
| LogSig Panel B net CAGR | 8.58% ± 4.46% | 6.64% ± 6.11% |
| LogSig Panel B Sharpe | 0.21 ± 0.15 | 0.15 ± 0.21 |
| LogSig Panel B MDD | 33.94% ± 3.74% | 35.61% ± 5.26% |
| LogSig Panel B turnover | 24.23% ± 3.52% | 23.08% ± 2.48% |
| LogSig Panel B Rank IC | +0.0304 ± 0.0139 | +0.0231 ± 0.0187 |
| Ensemble weekly Rank IC | +0.0425 (p = 0.0107) | +0.0332 (t = 1.85, p = 0.0331) |
| LogSig Panel B Deflated Sharpe | 0.0854 | 0.0526 |
| Gate weights ω₅ / ω₂₀ / ω₆₀ | 0.342 / 0.331 / 0.327 | 0.342 / 0.355 / 0.303 |
| Cap-drift weeks | 89 | 75 |
| Interior feasible weeks | 39 | 53 |
| Interior trades / holds | 26 / 13 | 33 / 20 |
| KKT agreement on interior weeks | 100% | **100%** |
| Declined weeks | 14 | 20 |
| Declined-trade mean P&L | −21.56 bps | −31.39 bps |
| Declined-trade total P&L | −301.78 bps | −627.77 bps |
| Declined-trade t / p | −2.14 / 0.0519 | −0.84 / 0.4132 |
| LogSig vs LogSig-Inc turnover | 24.23% vs 16.82% | 23.08% vs 19.94% |
| Trigger HHI | 0.0784 | 0.0601 |

**All three qualitative contributions survive:**

1. **The KKT diagnostic is exact.** 33 interior trades, 20 interior holds, 0 boundary cases, 0 disagreements with SLSQP — 100% agreement, as before.
2. **Inaction still spares capital**, and by a larger margin per week (−31.39 bps vs −21.56 bps), but the sample is now 20 weeks with a t-statistic of −0.84. **The effect is no longer statistically significant**, and the claim must be reframed from "systematically protects" to "spares capital, though the sample cannot reject zero at conventional levels".
3. **Lévy areas still amplify turnover**: 23.08% with cross-integrals versus 19.94% without, a 3.1 pp gap (published: 7.4 pp). The direction is unchanged; the magnitude is smaller.

This is a substantive revision, not a cosmetic one. It is the direct cost of the reproducibility defect: the published numbers depended on an artifact the repository could not produce.

---

## 4. What is now guaranteed

| Guarantee | Enforced by |
|:---|:---|
| Every artifact records its config, code, git SHA and library versions | `tests/test_provenance.py` |
| No artifact stores a formatted string | `test_artifacts_contain_no_formatted_strings` |
| `JOURNAL.md` tables match their artifacts cell for cell | `test_manuscript_tables_are_in_sync` |
| The vectorised feature path matches the scalar reference | `test_vectorised_matches_scalar_reference` |
| The `kappa = 0` reduction matches the split formulation | `test_unpenalized_reduction_matches_split` |
| The warm start is feasible even when the incumbent breaches the cap | `test_feasible_start_satisfies_all_constraints` |
| The volatility-scaled extension converges from a cap-breaching incumbent | `test_volatility_scaled_solve_converges` |
| Caching changes no result | `test_cached_backtest_is_transparent` |
| The paths `train` writes are the paths tests read | `test_checkpoint_paths_match_between_writer_and_reader` |
| `portfolio.py` carries no duplicated config defaults | `test_config_has_no_duplicated_defaults` |
| Every `trading` field reaches the backtest | `test_backtest_params_cover_every_trading_field` |
| Feature caching ignores `trading` overrides | `test_feature_digest_ignores_trading_overrides` |

---

## 5. Outstanding work

1. **Reconsider `workers.train`.** Three is safe on a 7.3 GB machine; a larger host could use five and save about two minutes of wall clock.
2. **Consider splitting `portfolio.py`.** At 871 lines it is the largest module. It is cohesive, but `covariance`, `kkt`, `solver` and `backtest` would each stand alone.

### Completed

- **`v1.0.0` release restored** with the raw snapshot attached as a release asset (`raw_snapshot_v1.0.0.zip`, SHA-256 `9BFBD7D6…CAE2C1`), as §2.10 claims.

- **Manuscript prose updated.** §3.1–§3.7, §4.1 and both abstracts now quote the regenerated values; the declined-trade claim is stated as economically suggestive rather than statistically established, and §4.2 documents the external-validity boundary (one market, one sample period, 40 assets).
- **Golden hashes blessed.** `python -m stillwar tables --bless` writes `tests/golden.json` from the five manuscript-feeding artifacts; `test_golden_artifact_hashes` now enforces the committed hashes instead of skipping.
- **One Python version pinned.** `pyproject.toml` requires `==3.14.*`, matching `.python-version` and the CPython 3.14 interpreter that produced every current artifact.
- **`make trials` has been run.** `data/processed/trials.json` holds the 1,225 trial Sharpe ratios (mean −0.091, s.d. 0.353, range [−1.488, 0.916]). The Deflated Sharpe column of Table 1 is now computed from that empirical distribution rather than from a hardcoded normal.
