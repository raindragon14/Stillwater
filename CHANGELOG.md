# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `data/processed/exp_short_circuit_equivalence.json`, produced by the `experiments`
  stage: the short-circuit versus full-solve decision agreement behind the claim of
  Section 2.6 and Appendix A.4 (24 runs, 100% agreement, identical return paths).
- `tests/test_short_circuit.py`: a synthetic-panel equivalence test plus an
  artifact-level test that cross-checks the interior counts against Table 3.

### Changed

- Appendix A.4 now states the exact scope of the agreement check instead of claiming
  coverage of the sensitivity configurations, which the artifact does not verify, and
  discloses the one near-boundary case it found (LogSig-Inc, seed 4, 13 August 2026,
  margin 1.5e-6) where the optimal trade falls below the execution tolerance.
- Figure references in `reporting/figures.py`, `backtest/metrics.py`,
  `backtest/engine.py`, `experiments/simulate.py` and `experiments/sensitivity.py`
  follow the manuscript's numbering (Figure 1 = KKT spread, Figure 2 = trigger
  decomposition; architecture and sensitivity are supplementary).
- `docs/word/` is ignored: it is a manual thesis/template workspace, never a tracked
  artifact.
- Manuscript prose no longer exposes repository file names, code parameters or
  code-formatted tickers, so `docs/JOURNAL.md` reads as a manuscript rather than a
  repository artifact.

## [2.0.0] - 2026-11-11

### Added

- Modular package layout: `config`, `data`, `features`, `models`, `optimization`,
  `backtest`, `experiments`, `reporting`, `provenance`, and a single `cli`.
- `--config`, `--set SECTION__KEY=VALUE`, and `--force` CLI options.
- DVC pipeline (`dvc.yaml`), Docker image, devcontainer, pre-commit hooks, and GitHub
  Actions CI.
- Value-level golden artifacts (`tests/golden_values.json`) with a `1e-6` float
  tolerance.
- `docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md`, `docs/RELEASE.md`,
  `docs/REVISION_NOTES.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `CITATION.cff`.
- `py.typed` marker and package data.

### Changed

- Renamed the project from **Stillwar** to **Stillwater**; the Python package, CLI,
  Docker image, and repository URLs follow the new name.
- Split the former `portfolio.py`, `features.py`, `models.py`, `simulate.py`,
  `tables.py`, and `figures.py` into responsibility-based packages.
- Manuscript figures renamed to `fig1_kkt_spread` and `fig2_trigger_decomposition`;
  supplementary figures are generated locally and ignored.
- Golden tests now compare numeric payloads instead of exact hashes, so a different
  BLAS/SciPy build does not fail the suite over a last-bit difference.
- `Development Status` classifier changed to beta, matching a research codebase.

### Removed

- Historical `docs/CODE_REVIEW.md` and `docs/EVALUATION.md` from the current tree
  (preserved in Git history).
- Unused top-level compatibility shims.
- Duplicate `docs/figures/` copies and the stale `docs/JOURNAL.docx`.
- Old figure names (`fig2_kkt_spread`, `fig3_trigger_heatmap`, `fig1_architecture`,
  `fig4_sensitivity_and_extensions`).

### Fixed

- Mypy errors across the package; `make typecheck` now passes.
- Version mismatch between `pyproject.toml` and `stillwater.__version__`.
- Stale module references in docstrings, tests, and provenance dependencies.
