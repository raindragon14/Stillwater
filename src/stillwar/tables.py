"""Render the manuscript tables from the JSON artifacts, and keep ``JOURNAL.md`` in sync.

Formatting lives here and nowhere else. The previous design wrote pre-formatted strings
into the result JSON, which destroyed numeric precision and made programmatic
verification impossible -- the reconciliation test could only compare prefixes. With
numbers kept as numbers, :func:`sync_manuscript` can regenerate a table and compare it
cell by cell, so the manuscript can no longer drift from its artifact.

Each table in ``docs/JOURNAL.md`` sits between ``<!-- BEGIN TABLE n -->`` and
``<!-- END TABLE n -->`` markers. Text outside the markers is never touched.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stillwar.provenance import read_json

__all__ = [
    "bless_golden",
    "render_table1",
    "render_table2",
    "render_table3",
    "render_table_s1",
    "sync_manuscript",
]

JOURNAL = Path("docs/JOURNAL.md")

# The processed artifacts behind every manuscript table (Tables 1-3 and S1).
GOLDEN_ARTIFACTS = (
    "table1.json",
    "table2.json",
    "table3.json",
    "exp_unconstrained_tolerance.json",
    "exp_volatility_penalty.json",
)


def bless_golden(processed: Path, golden: Path) -> dict[str, str]:
    """Write the golden hashes of the manuscript-feeding artifacts to ``golden``.

    The hash matches ``tests.test_provenance.test_golden_artifact_hashes`` exactly: the
    SHA-256 of the sorted-JSON ``results`` block of each artifact, truncated to 16 hex
    characters. Run ``python -m stillwar tables --bless`` once the manuscript numbers
    are final; the committed file turns the golden test from a skip into an enforcement.
    """
    hashes: dict[str, str] = {}
    for name in GOLDEN_ARTIFACTS:
        _meta, results = read_json(processed / name)
        hashes[name] = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()[:16]
    golden.write_text(json.dumps(hashes, indent=4) + "\n", encoding="utf-8")
    return hashes

_TABLE2_LABELS = {
    "kappa_0": r"$\kappa = 0$ (no penalty)",
    "kappa_1": r"$\kappa = 1$",
    "kappa_2.5": r"$\kappa = 2.5$",
    "kappa_5": r"$\kappa = 5$ (primary)",
    "kappa_10": r"$\kappa = 10$",
    "gamma_1": r"$\gamma = 1$",
    "gamma_5": r"$\gamma = 5$",
    "gamma_10": r"$\gamma = 10$",
    "m_0.5": r"$m = 0.5$",
    "m_2": r"$m = 2$",
    "non_inclusive": r"Non-inclusive costs ($c_{b} = 0.001933$, $c_{s} = 0.002933$)",
}

_TABLE1_ORDER = (
    "LogSig",
    "LogSig-Inc",
    "MLP",
    "LSTM",
    "Hist-MV, penalized",
    "Hist-MV, unpenalized",
    "GMV",
    "EW",
    "GMV, unpenalized (supplementary)",
)


def _pct(value: float | None, digits: int = 2) -> str:
    """Format a fraction as a percentage, or an en dash when missing."""
    return "–" if value is None else f"{value * 100:.{digits}f}%"


def _pct_pm(stat: dict[str, float] | None, digits: int = 2) -> str:
    """Format a mean and standard deviation as percentages."""
    if stat is None:
        return "–"
    if stat.get("n", 1) <= 1:
        return _pct(stat["mean"], digits)
    return f"{_pct(stat['mean'], digits)} ± {_pct(stat['sd'], digits)}"


def _num(value: float | None, digits: int = 2) -> str:
    """Format a scalar, or an en dash when missing."""
    return "–" if value is None else f"{value:.{digits}f}"


def _num_pm(stat: dict[str, float] | None, digits: int = 2) -> str:
    """Format a mean and standard deviation as plain numbers."""
    if stat is None:
        return "–"
    if stat.get("n", 1) <= 1:
        return _num(stat["mean"], digits)
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def _signed(stat: dict[str, float] | None, digits: int = 4) -> str:
    """Format a signed mean and standard deviation, e.g. ``+0.0279 ± 0.0130``."""
    if stat is None:
        return "–"
    if stat.get("n", 1) <= 1:
        return f"{stat['mean']:+.{digits}f}"
    return f"{stat['mean']:+.{digits}f} ± {stat['sd']:.{digits}f}"


def _alpha(block: dict[str, float] | None) -> str:
    """Format a Newey-West alpha and its HAC t-statistic."""
    if block is None:
        return "–"
    return f"{block['alpha'] * 100:+.2f}% (t={block['t']:.2f})"


def _rank_ic(block: dict[str, Any] | None) -> str:
    """Format the weekly Rank IC cell of Table 1."""
    if block is None:
        return "–"
    return _signed(block.get("weekly"))


def render_table1(table: dict[str, Any]) -> str:
    """Render Table 1: out-of-sample performance, Panels A and B.

    Parameters
    ----------
    table
        The ``results`` block of ``data/processed/table1.json``.

    Returns
    -------
    str
        A GitHub-flavoured Markdown table, without the caption.
    """
    lines = [
        "| Strategy | Net CAGR | Sharpe | MDD | Turnover | Rank IC | NW alpha vs EW | Deflated Sharpe |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for panel, title in (("panel_a", "**Panel A (test 2024–2025)**"), ("panel_b", "**Panel B (full out-of-sample, incl. the 2026 continuation)**")):
        lines.append(f"| {title} |  |  |  |  |  |  |  |")
        for name in _TABLE1_ORDER:
            if name not in table:
                continue
            block = table[name][panel]
            lines.append(
                "| {name} | {cagr} | {sharpe} | {mdd} | {turnover} | {ic} | {alpha} | {dsr} |".format(
                    name=name,
                    cagr=_pct_pm(block["cagr"]),
                    sharpe=_num_pm(block["sharpe"]),
                    mdd=_pct_pm(block["mdd"]),
                    turnover=_pct_pm(block["turnover"]),
                    ic=_rank_ic(block["rank_ic"]),
                    alpha=_alpha(block["nw_alpha"]),
                    dsr=_num(block["dsr"], 4),
                )
            )
    return "\n".join(lines)


def render_table2(rows: Sequence[dict[str, Any]]) -> str:
    """Render Table 2: sensitivity of LogSig to ``kappa``, ``gamma`` and the cost level.

    Parameters
    ----------
    rows
        The ``results`` block of ``data/processed/table2.json``.

    Returns
    -------
    str
        A GitHub-flavoured Markdown table.
    """
    lines = [
        r"| Setting | Net CAGR | Sharpe | MDD | Turnover | $\psi_{\kappa}$ | $\psi_{\kappa}^{0}$ |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {cagr} | {sharpe} | {mdd} | {turnover} | {psi} | {psi0} |".format(
                label=_TABLE2_LABELS.get(row["label"], row["label"]),
                cagr=_pct(row["cagr"]),
                sharpe=_num(row["sharpe"]),
                mdd=_pct(row["mdd"]),
                turnover=_pct(row["turnover"]),
                psi=_pct(row["psi_kappa"]),
                psi0="–" if row["psi_kappa_0"] is None else _pct(row["psi_kappa_0"]),
            )
        )
    return "\n".join(lines)


def render_table3(counts: dict[str, int]) -> str:
    """Render Table 3: Theorem 1 verdicts against the solver's decision.

    Parameters
    ----------
    counts
        The ``results`` block of ``data/processed/table3.json``.

    Returns
    -------
    str
        A GitHub-flavoured Markdown table.
    """
    return "\n".join(
        [
            "| Theorem 1 verdict | Solver traded | Solver held | Total |",
            "|:---|---:|---:|---:|",
            f"| Trade ($\\Delta g_{{t}} \\gt \\eta_{{b}} + \\eta_{{s}}$) | {counts['trade_traded']} | {counts['trade_held']} | {counts['trade_total']} |",
            f"| Hold ($\\Delta g_{{t}} \\leq \\eta_{{b}} + \\eta_{{s}}$) | {counts['hold_traded']} | {counts['hold_held']} | {counts['hold_total']} |",
            f"| Boundary (margin below $10^{{-6}}$) | {counts['boundary_traded']} | {counts['boundary_held']} | {counts['boundary_count']} |",
            f"| Not applicable (cap-drift: $w^{{-}}$ infeasible for (5)) | {counts['cap_drift_traded']} | {counts['cap_drift_held']} | {counts['cap_drift_count']} |",
            f"| **Total** | {counts['total_solver_traded']} | {counts['total_solver_held']} | {counts['total_dates']} |",
        ]
    )


def render_table_s1(
    tolerance: dict[str, Any],
    volatility: dict[str, Any],
) -> str:
    """Render Table S1: the structural boundary extensions of Section 3.7.

    Parameters
    ----------
    tolerance
        The ``results`` block of ``data/processed/exp_unconstrained_tolerance.json``.
    volatility
        The ``results`` block of ``data/processed/exp_volatility_penalty.json``.

    Returns
    -------
    str
        A GitHub-flavoured Markdown table.
    """
    lines = [
        "| Extension configuration | Net CAGR | Sharpe | MDD | Turnover | Cap-drift weeks | Interior signal trades | Optimal KKT holds |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def row(name: str, cell: dict[str, Any]) -> str:
        panel = cell["panel_b"]
        return (
            f"| {name} | {_pct(panel['cagr'])} | {_num(panel['sharpe'])} | {_pct(panel['mdd'])} | "
            f"{_pct(panel['turnover'])} | {cell['cap_drift_mean']:.1f} | "
            f"{cell['signal_trade_mean']:.1f} | {cell['optimal_hold_mean']:.1f} |"
        )

    baseline = tolerance["tolerance_grid"]["w_max_0.15_eps_0.000"]
    lines.append(
        row(r"Baseline ($\epsilon = 0.0\%$, uniform $\eta$, $w_{\max} = 0.15$)", baseline)
    )
    for eps, label in ((0.005, "0.5\\%"), (0.010, "1.0\\%"), (0.020, "2.0\\%")):
        key = f"w_max_0.15_eps_{eps:.3f}"
        lines.append(row(rf"Asymmetric cap-drift buffer ($\epsilon = {label}$)", tolerance["tolerance_grid"][key]))

    unconstrained = tolerance["unconstrained"]
    lines.append(
        f"| Unconstrained cap ($w_{{\\max}} = 1.0$, mean $\\max_i w_i = "
        f"{unconstrained['max_weight_mean'] * 100:.2f}\\%$) | "
        f"{_pct(unconstrained['panel_b']['cagr']['mean'])} | "
        f"{_num(unconstrained['panel_b']['sharpe']['mean'])} | "
        f"{_pct(unconstrained['panel_b']['mdd']['mean'])} | "
        f"{_pct(unconstrained['panel_b']['turnover']['mean'])} | 0.0 | – | – |"
    )

    scaled = volatility["volatility_scaled"]
    lines.append(
        f"| Volatility-scaled penalty ($\\eta_i \\propto \\sigma_{{t,i}} / \\overline{{\\sigma}}_t$, "
        f"HHI $= {scaled['triggers']['herfindahl_constituents']:.4f}$) | "
        f"{_pct(scaled['panel_b']['cagr'])} | {_num(scaled['panel_b']['sharpe'])} | "
        f"{_pct(scaled['panel_b']['mdd'])} | {_pct(scaled['panel_b']['turnover'])} | – | "
        f"{scaled['triggers']['crossing_count']:.0f} (seed 0) | – |"
    )
    return "\n".join(lines)


def _replace_block(text: str, name: str, body: str) -> str:
    """Replace the content between ``<!-- BEGIN name -->`` and ``<!-- END name -->``."""
    pattern = re.compile(
        rf"(<!-- BEGIN {re.escape(name)} -->\n).*?(\n<!-- END {re.escape(name)} -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise KeyError(f"marker for {name!r} not found in the manuscript")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text)


def sync_manuscript(
    blocks: dict[str, str],
    path: str | Path = JOURNAL,
    write: bool = True,
) -> bool:
    """Rewrite the generated table blocks in the manuscript.

    Parameters
    ----------
    blocks
        ``{marker_name: rendered_markdown}``, e.g. ``{"TABLE 1": render_table1(...)}``.
    path
        Path to the manuscript.
    write
        When ``False``, only report whether the file is already in sync.

    Returns
    -------
    bool
        ``True`` if the file changed (or would change).

    Raises
    ------
    KeyError
        If a marker is missing, which means the manuscript was edited by hand.
    """
    target = Path(path)
    original = target.read_text(encoding="utf-8")
    updated = original
    for name, body in blocks.items():
        updated = _replace_block(updated, name, body)
    changed = updated != original
    if changed and write:
        target.write_text(updated, encoding="utf-8")
    return changed


def load_blocks(processed: str | Path = "data/processed") -> dict[str, str]:
    """Load the manuscript artifacts and render the synced table blocks.

    The manuscript in this repository carries Tables 1--3.  The structural
    extensions (Table S1) remain available through :func:`render_table_s1` for
    supplementary material but are not part of the main manuscript sync.

    Parameters
    ----------
    processed
        Directory holding the JSON artifacts.

    Returns
    -------
    dict
        ``{marker_name: markdown}`` ready for :func:`sync_manuscript`.
    """
    root = Path(processed)
    _meta1, table1 = read_json(root / "table1.json")
    _meta2, table2 = read_json(root / "table2.json")
    _meta3, table3 = read_json(root / "table3.json")
    return {
        "TABLE 1": render_table1(table1),
        "TABLE 2": render_table2(table2),
        "TABLE 3": render_table3(table3),
    }
