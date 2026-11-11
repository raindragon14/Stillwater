"""Render the manuscript figures from the JSON artifacts.

Every series plotted here is loaded from a file. The previous implementation hardcoded
the eight data series of Figure 4 as literals and re-simulated the entire backtest for
Figure 2, so the figures could disagree with the tables they illustrate -- and did.
Nothing in this module may compute a portfolio quantity.

Figures follow the publication house style (scientific-figure-making): the semantic
:class:`PALETTE`, minimalist spines, frameless legends, panel letters, and 300 dpi PNG
plus vector PDF/SVG export via :func:`finalize_figure`.

Sizing follows the journal page, not the screen. The manuscript prints in a single
column of ``TEXT_WIDTH_IN = 6.5`` inches, so every figure is drawn at exactly that
width with an 8 pt base font: what the reader sees on paper is what this module
draws, with no silent downscaling of the type.

Figure numbering follows the manuscript: ``fig1`` is the architecture schematic,
``fig2`` the KKT gradient spread, ``fig3`` the trigger decomposition, ``fig4`` the
sensitivity and extension panels.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

from stillwar.backtest.metrics import trigger_endpoints  # noqa: E402
from stillwar.provenance.metadata import read_json  # noqa: E402

__all__ = [
    "PALETTE",
    "FigureStyle",
    "apply_publication_style",
    "create_subplots",
    "finalize_figure",
    "plot_all",
    "plot_architecture",
    "plot_kkt_spread",
    "plot_sensitivity",
    "plot_triggers",
]

FIG_DIR = Path("reports/figures")
DPI = 300

#: Printed width of one journal column, in inches (A4, 2.2 cm side margins).
TEXT_WIDTH_IN = 6.5

#: Export set: raster for the DOCX, vector for the typesetter.
FORMATS = ("png", "pdf", "svg")

#: Semantic colour map (scientific-figure-making): blues for the proposed method,
#: greens for improvements, reds for baselines/contrasts, neutrals for references.
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

_DARK = "#4D4D4D"
_GRID = "#D9D9D9"


@dataclass(frozen=True)
class FigureStyle:
    """Publication style knobs (scientific-figure-making ``FigureStyle``).

    The defaults are the print-size setting for this manuscript: 8 pt type on a
    6.5-inch canvas, which lands inside the 7-9 pt band journals ask for.
    """

    font_size: float = 8.0
    axes_linewidth: float = 1.0
    use_tex: bool = False
    font_family: tuple[str, ...] = ("DejaVu Sans", "Arial", "Helvetica", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    """Configure matplotlib rcParams: minimalist spines, frameless legend, fonts."""
    style = style or FigureStyle()
    family = _resolve_font_family(style.font_family)
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "DejaVu Sans"],
            "font.size": style.font_size,
            "axes.titlesize": style.font_size,
            "axes.labelsize": style.font_size,
            "xtick.labelsize": style.font_size - 0.5,
            "ytick.labelsize": style.font_size - 0.5,
            "legend.fontsize": style.font_size - 0.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": style.axes_linewidth,
            "axes.edgecolor": _DARK,
            "axes.labelcolor": "black",
            "text.color": "black",
            "xtick.color": _DARK,
            "ytick.color": _DARK,
            "xtick.major.width": style.axes_linewidth,
            "ytick.major.width": style.axes_linewidth,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.1,
            "legend.labelspacing": 0.35,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "patch.linewidth": 0.8,
            "savefig.dpi": DPI,
            "figure.dpi": 150,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _resolve_font_family(candidates: Sequence[str]) -> str:
    """Return the first installed family from ``candidates``.

    Handing matplotlib a fallback list makes it emit one ``findfont`` warning per
    text object -- thousands of lines for a multi-panel figure -- because it resolves
    the list lazily at draw time. Resolving once here keeps the build log readable
    and guarantees the same face in the PNG, PDF and SVG exports.
    """
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return "DejaVu Sans"


def create_subplots(
    nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] | None = None, **kwargs: Any
) -> tuple[plt.Figure, np.ndarray]:
    """Return ``(fig, axes)`` with ``axes`` a flat 1D array."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(axes).ravel()


def finalize_figure(
    fig: plt.Figure,
    out_path: str | Path,
    formats: Sequence[str] | None = None,
    dpi: int = DPI,
    close: bool = True,
    pad: float = 0.02,
    tight: bool = False,
) -> list[Path]:
    """Save the figure to one or more formats (PNG + vector PDF/SVG) and release it.

    The canvas is written at exactly the ``figsize`` passed to ``plt.subplots``, so a
    figure drawn at ``TEXT_WIDTH_IN`` is a ``TEXT_WIDTH_IN``-wide file. Cropping with
    ``bbox_inches="tight"`` would silently shrink the canvas (Figure 1 came out at
    5.08 in), and the word processor would then scale it back up to the column width,
    inflating every font size and dropping the effective resolution below 300 dpi.
    Layout is therefore fixed with ``subplots_adjust`` inside each plot function and
    the saved canvas is never cropped. Pass ``tight=True`` only for a figure whose
    axes geometry cannot be pinned down in advance.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    formats = list(formats) if formats is not None else list(FORMATS)
    stem = out_path.with_suffix("")
    written: list[Path] = []
    for fmt in formats:
        target = stem.with_suffix(f".{fmt}")
        if tight:
            fig.savefig(target, dpi=dpi, bbox_inches="tight", pad_inches=pad, facecolor="white")
        else:
            fig.savefig(target, dpi=dpi, bbox_inches=None, facecolor="white")
        written.append(target)
    if close:
        plt.close(fig)
    return written


def _panel_label(ax: plt.Axes, letter: str, title: str = "") -> None:
    """Write ``(A) Title`` above the axes, left aligned, in the house style."""
    text = f"({letter})" + (f" {title}" if title else "")
    ax.set_title(text, loc="left", fontsize=matplotlib.rcParams["font.size"], pad=4)


def _light_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Dashed light grid behind the data (print-safe, never dominant)."""
    ax.grid(axis=axis, color=_GRID, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


def plot_architecture(out_path: str | Path = FIG_DIR / "fig1_architecture.png") -> None:
    """Draw the end-to-end pipeline schematic (manuscript Figure 1).

    Parameters
    ----------
    out_path
        Destination PNG.

    Notes
    -----
    This is the only figure with no data dependency; it documents the chain described
    in Sections 2.3 to 2.6 and exists so that ``make figures`` produces every figure
    the manuscript references. The seven stages are laid out on one row at print
    width, with the Theorem 1 hold branch drawn underneath stage five.
    """
    stages: list[tuple[str, str, str]] = [
        ("Daily OHLCV\n5 channels", PALETTE["neutral"], "black"),
        ("Order-2\nlog-signature\n45 features", PALETTE["blue_secondary"], "white"),
        ("Multiscale\ngated net\n923 params", PALETTE["blue_main"], "white"),
        ("Forecast\n$\\widehat{y}$, shrink\n$\\alpha=0.5$", PALETTE["blue_secondary"], "white"),
        ("Theorem 1\n$\\Delta g_t\\leq\\eta_b+\\eta_s$", PALETTE["red_2"], "black"),
        ("SLSQP\nEq. (4)-(5)", PALETTE["neutral"], "black"),
        ("Execute at\nclose of $t+1$", PALETTE["neutral"], "black"),
    ]

    width, gap = 1.0, 0.10
    n = len(stages)
    span = n * width + (n - 1) * gap
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 1.95))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()

    box_top, box_bottom = 1.0, 0.44
    for i, (label, colour, text_colour) in enumerate(stages):
        x = i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, box_bottom),
                width,
                box_top - box_bottom,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=colour,
                edgecolor=_DARK,
                linewidth=0.9,
            )
        )
        ax.text(
            x + width / 2,
            (box_top + box_bottom) / 2,
            label,
            ha="center",
            va="center",
            fontsize=6.4,
            linespacing=1.4,
            color=text_colour,
        )
        if i:
            ax.add_patch(
                FancyArrowPatch(
                    (x - gap + 0.01, (box_top + box_bottom) / 2),
                    (x - 0.01, (box_top + box_bottom) / 2),
                    arrowstyle="-|>",
                    mutation_scale=6,
                    color=_DARK,
                    linewidth=0.9,
                    shrinkA=0,
                    shrinkB=0,
                )
            )

    # Hold branch: Theorem 1 short-circuits the solver on interior hold dates.
    hold_x = 4 * (width + gap) + width / 2
    ax.add_patch(
        FancyArrowPatch(
            (hold_x, box_bottom - 0.01),
            (hold_x, 0.30),
            arrowstyle="-|>",
            mutation_scale=6,
            color=PALETTE["red_strong"],
            linewidth=0.9,
            linestyle=(0, (2.5, 1.5)),
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(
        hold_x,
        0.25,
        "Hold: no order sent,\nholdings drift by Eq. (8)",
        ha="center",
        va="top",
        fontsize=6.4,
        linespacing=1.3,
        color=PALETTE["red_strong"],
    )

    ax.set_xlim(0.0, span)
    ax.set_ylim(0.04, 1.10)
    return finalize_figure(fig, out_path)


def plot_kkt_spread(
    kkt_log: Sequence[dict[str, Any]],
    band: float,
    out_path: str | Path = FIG_DIR / "fig2_kkt_spread.png",
) -> None:
    """Plot the gradient spread against the no-trade band (manuscript Figure 2).

    Interior dates are drawn as markers only. A connecting line would interpolate
    across the 75 cap-drift dates, on which no finite spread exists because
    ``w_t^-`` leaves the domain of Theorem 1; those dates are shown as a grey rug
    in a dedicated strip beneath the axes instead.

    Parameters
    ----------
    kkt_log
        The ``results`` block of ``data/processed/kkt_log.json``.
    band
        ``eta_b + eta_s`` at the primary penalty scale.
    out_path
        Destination PNG.

    Notes
    -----
    Reads the same log that produces Table 3, so the figure and the table cannot
    disagree. Cap-drift dates carry ``delta_g = nan`` and are plotted separately.

    The spread is non-negative by construction (it is a max minus a min of the same
    gradient components), so the y-axis starts at zero. The cap-drift rug used to be
    drawn at negative y, which implied spreads below zero and put the marks on top of
    the x-axis; it now has its own narrow axis, sharing the x scale.
    """
    dates = [Date.fromisoformat(row["date"]) for row in kkt_log]
    spreads = np.array([row["delta_g"] for row in kkt_log], dtype=float)
    kinds = np.array(
        [
            "cap_drift"
            if row["is_cap_drift"]
            else ("signal_trade" if row["kkt_should_trade"] else "optimal_hold")
            for row in kkt_log
        ]
    )
    x = np.arange(len(dates))

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 2.8))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.05], hspace=0.12)
    ax = fig.add_subplot(grid[0])
    rug = fig.add_subplot(grid[1], sharex=ax)

    ax.axhspan(0.0, band, color=PALETTE["green_1"], zorder=0)
    ax.axhline(band, color=PALETTE["red_strong"], ls="--", lw=1.0, zorder=2)

    for kind, colour, marker, size, label in (
        ("signal_trade", PALETTE["blue_main"], "o", 16, "Signal trade"),
        ("optimal_hold", PALETTE["green_3"], "s", 16, "Optimal hold"),
    ):
        mask = kinds == kind
        ax.scatter(
            x[mask],
            spreads[mask],
            color=colour,
            marker=marker,
            s=size,
            edgecolor="black",
            linewidth=0.4,
            zorder=4,
            label=f"{label} ({int(mask.sum())} weeks)",
        )

    # Cap-drift dates: no finite spread, drawn as a rug in its own strip.
    drift = kinds == "cap_drift"
    rug.vlines(x[drift], 0.12, 0.88, color="#767676", linewidth=0.7, zorder=3)
    rug.set_ylim(0.0, 1.0)
    rug.set_yticks([])
    for side in ("left", "right", "top", "bottom"):
        rug.spines[side].set_visible(False)
    rug.tick_params(axis="y", length=0)
    rug.set_xlabel("Rebalance date (Panel B, weekly decision close)", labelpad=2)

    ax.set_ylabel(r"Gradient spread $\Delta g_t$")
    ax.tick_params(labelbottom=False)
    tick_idx = list(range(0, len(dates), 16))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([dates[i].strftime("%b %Y") for i in tick_idx])
    ax.set_xlim(-2, len(dates) + 1)
    ax.set_ylim(0.0, float(np.nanmax(spreads)) * 1.12)

    handles = [
        Patch(facecolor=PALETTE["green_1"], edgecolor="none", label=f"No-trade band $\\eta_b+\\eta_s={band:.2f}$"),
        Line2D([], [], color=PALETTE["red_strong"], ls="--", lw=1.0, label="No-trade threshold"),
        Line2D([], [], marker="o", ls="none", color=PALETTE["blue_main"], markeredgecolor="black",
               markeredgewidth=0.4, markersize=3.6, label=f"Signal trade ({int((kinds == 'signal_trade').sum())} weeks)"),
        Line2D([], [], marker="s", ls="none", color=PALETTE["green_3"], markeredgecolor="black",
               markeredgewidth=0.4, markersize=3.6, label=f"Optimal hold ({int((kinds == 'optimal_hold').sum())} weeks)"),
        Line2D([], [], marker="|", ls="none", color="#767676", markersize=5,
               label=f"Cap-drift date ({int(drift.sum())} weeks)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
        borderaxespad=0.2,
        columnspacing=1.4,
    )
    fig.subplots_adjust(bottom=0.30, top=0.975, left=0.088, right=0.985)
    return finalize_figure(fig, out_path)


def plot_triggers(
    kkt_log: Sequence[dict[str, Any]],
    tickers: Sequence[str],
    out_path: str | Path = FIG_DIR / "fig3_trigger_heatmap.png",
) -> None:
    """Plot the sector and constituent trigger decomposition (manuscript Figure 3).

    The caption promises that the bars "decompose extreme sell and buy gradient
    triggers", so each bar is stacked into the sell endpoint (the asset that set
    ``max_j g_j``) and the buy endpoint (the asset that set ``min_i g_i``). The
    pooled count is what a Herfindahl index needs; the split is what shows that a
    sector can be a net seller of risk while another is a net buyer.

    Parameters
    ----------
    kkt_log
        The ``results`` block of ``data/processed/kkt_log.json``.
    tickers
        The 40 constituents, in the order the trigger indices refer to.
    out_path
        Destination PNG.

    Notes
    -----
    Reads the same log that produces Table 3, so the 33 crossing dates, the 66
    endpoints and the 10-of-66 concentration of ``BRPT.JK`` cannot disagree with
    the prose in Section 3.5.
    """
    endpoints = trigger_endpoints(kkt_log, tickers)
    grand = endpoints["sell_total"] + endpoints["buy_total"]

    sectors = sorted(
        endpoints["sector_sell"],
        key=lambda name: endpoints["sector_sell"][name] + endpoints["sector_buy"][name],
    )
    sector_sell = [endpoints["sector_sell"][name] / grand * 100 for name in sectors]
    sector_buy = [endpoints["sector_buy"][name] / grand * 100 for name in sectors]

    order = np.argsort(endpoints["total"])[::-1][:10][::-1]
    names = [tickers[i].replace(".JK", "") for i in order]
    top_sell = [endpoints["sell"][i] / grand * 100 for i in order]
    top_buy = [endpoints["buy"][i] / grand * 100 for i in order]

    fig, axes = create_subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.55))
    ax1, ax2 = axes
    panels = (
        (ax1, sectors, sector_sell, sector_buy, "A", "By IDX-IC sector", 41.0),
        (ax2, names, top_sell, top_buy, "B", "Top-10 constituents", 18.5),
    )
    for ax, labels, sell_values, buy_values, letter, title, xmax in panels:
        y = np.arange(len(labels))
        ax.barh(
            y, sell_values, height=0.70, color=PALETTE["red_2"],
            edgecolor="black", linewidth=0.5, label="Sell trigger (max gradient)",
        )
        ax.barh(
            y, buy_values, height=0.70, left=sell_values, color=PALETTE["blue_main"],
            edgecolor="black", linewidth=0.5, label="Buy trigger (min gradient)",
        )
        for yi, (s, b) in enumerate(zip(sell_values, buy_values, strict=True)):
            total = s + b
            ax.text(
                total + xmax * 0.012, yi, f"{total:.1f}",
                va="center", ha="left", fontsize=6.6,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.4)
        ax.set_xlabel("Share of triggers (%)", labelpad=2)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.7, len(labels) - 0.3)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        _light_grid(ax, axis="x")
        _panel_label(ax, letter, title)

    handles = [
        Patch(facecolor=PALETTE["red_2"], edgecolor="black", linewidth=0.5,
              label="Sell trigger (max gradient)"),
        Patch(facecolor=PALETTE["blue_main"], edgecolor="black", linewidth=0.5,
              label="Buy trigger (min gradient)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        borderaxespad=0.2,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.232, right=0.978, top=0.915, bottom=0.235, wspace=0.62)
    return finalize_figure(fig, out_path)


def plot_sensitivity(
    tolerance: dict[str, Any],
    shrinkage: dict[str, Any],
    volatility: dict[str, Any],
    out_path: str | Path = FIG_DIR / "fig4_sensitivity_and_extensions.png",
) -> None:
    """Plot the three structural extensions (manuscript Figure 4).

    The caption fixes the panel count: (A) cap-drift forced trades against optimal
    KKT holds across the buffer tolerance ``epsilon``, (B) the out-of-sample Sharpe
    ratio across forecast shrinkage ``alpha`` on both panels, and (C) constituent
    trigger concentration under uniform versus volatility-scaled penalties. A fourth
    panel would be a figure the caption does not describe, so the performance index
    that used to sit in (D) is reported in Table S1 instead.

    Parameters
    ----------
    tolerance
        The ``results`` block of ``exp_unconstrained_tolerance.json``.
    shrinkage
        The ``results`` block of ``exp_shrinkage.json``.
    volatility
        The ``results`` block of ``exp_volatility_penalty.json``.
    out_path
        Destination PNG.

    Notes
    -----
    Panel (A) is drawn as grouped bars rather than a line: the tolerance grid is
    ``epsilon in {0, 0.5, 1, 2, 5}``, and a line chart would space those five points
    evenly, drawing an axis that does not exist. Panel (C) reports the Herfindahl
    index of constituent triggers scaled by 100 so that it shares the percent axis
    with the top-1 trigger share.
    """
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 4.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.32)
    ax1 = fig.add_subplot(grid[0, :])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])

    # -- (A) cap-drift forced trades vs. optimal KKT holds --------------------
    cells = sorted(
        (c for c in tolerance["tolerance_grid"].values() if c["w_max"] == 0.15),
        key=lambda c: c["epsilon"],
    )
    eps_labels = ["0\n(base)" if c["epsilon"] == 0 else f"{c['epsilon'] * 100:g}" for c in cells]
    _grouped_bars(
        ax1,
        eps_labels,
        [
            ([c["cap_drift_mean"] for c in cells], PALETTE["red_2"], "Cap-drift forced trades"),
            ([c["optimal_hold_mean"] for c in cells], PALETTE["blue_main"], "Optimal KKT holds"),
        ],
        ylabel="Rebalance weeks (of 128)",
        ylim=(0, 104),
        fmt="{:.1f}",
        fontsize=6.2,
    )
    ax1.set_xlabel(r"Cap-drift buffer $\epsilon$ (%)")
    ax1.legend(
        loc="upper right",
        fontsize=6.2,
        handlelength=1.3,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    _panel_label(ax1, "A", r"Cap-drift forced trades vs. optimal KKT holds")

    # -- (B) forecast shrinkage: out-of-sample Sharpe on both panels ----------
    keys = sorted(shrinkage, key=lambda k: shrinkage[k]["alpha"])
    alphas = [shrinkage[k]["alpha"] for k in keys]
    for panel, colour, marker, label in (
        ("panel_a", PALETTE["blue_main"], "o", "Panel A (2024\u20132025)"),
        ("panel_b", PALETTE["red_strong"], "s", "Panel B (2024\u20132026)"),
    ):
        ax2.plot(
            alphas,
            [shrinkage[k][panel]["sharpe"] for k in keys],
            marker=marker,
            color=colour,
            lw=1.2,
            ms=3.4,
            label=label,
        )
    ax2.axhline(0.0, color=_DARK, lw=0.7, ls=":")
    ax2.set_xlabel(r"Forecast shrinkage $\alpha$")
    ax2.set_ylabel("Out-of-sample Sharpe ratio")
    ax2.set_xticks(alphas)
    ax2.set_ylim(-0.45, 1.08)
    _light_grid(ax2)
    ax2.legend(loc="upper right", fontsize=6.0, handlelength=1.5, handletextpad=0.4)
    _panel_label(ax2, "B", r"Forecast shrinkage $\alpha$")

    # -- (C) volatility-scaled penalty: trigger concentration -----------------
    uniform, scaled = volatility["uniform"], volatility["volatility_scaled"]
    uniform_c = [
        uniform["triggers"]["top_10_constituents"][0]["share"] * 100,
        uniform["triggers"]["herfindahl_constituents"] * 100,
    ]
    scaled_c = [
        scaled["triggers"]["top_10_constituents"][0]["share"] * 100,
        scaled["triggers"]["herfindahl_constituents"] * 100,
    ]
    _grouped_bars(
        ax3,
        ["Top-1 trigger\nshare", "Constituent\nHHI $\\times\\,100$"],
        [
            (uniform_c, PALETTE["neutral"], "Uniform penalty"),
            (scaled_c, PALETTE["blue_main"], "Volatility-scaled penalty"),
        ],
        ylabel="Concentration (%)",
        ylim=(0, 19.5),
        fmt="{:.2f}",
        fontsize=6.2,
    )
    ax3.legend(
        loc="upper right",
        fontsize=6.0,
        handlelength=1.3,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    _panel_label(ax3, "C", "Trigger concentration")

    fig.subplots_adjust(left=0.135, right=0.985, top=0.935, bottom=0.115)
    return finalize_figure(fig, out_path)


def _grouped_bars(
    ax: plt.Axes,
    categories: Sequence[str],
    series: Sequence[tuple[Sequence[float], str, str]],
    ylabel: str,
    ylim: tuple[float, float],
    *,
    fmt: str = "{:.2f}",
    fontsize: float = 6.2,
    bar_width: float | None = None,
) -> None:
    """Grouped bars with value labels.

    ``series`` holds ``(values, colour, label)`` triples, one per group, so a panel
    can show two treatments or three date categories without a second helper.
    """
    n = len(series)
    x = np.arange(len(categories))
    width = bar_width if bar_width is not None else 0.8 / n
    pad = (ylim[1] - ylim[0]) * 0.015
    for i, (values, colour, _label) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width, color=colour, edgecolor="black", linewidth=0.5, label=_label
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + pad,
                fmt.format(bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=6.8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    _light_grid(ax)


def plot_all(processed: str | Path = "data/processed", out_dir: str | Path = FIG_DIR) -> list[Path]:
    """Render every manuscript figure from the JSON artifacts.

    Parameters
    ----------
    processed
        Directory holding the JSON artifacts.
    out_dir
        Directory for the PNG, PDF and SVG files.

    Returns
    -------
    list of Path
        The files written.
    """
    apply_publication_style()
    root, out = Path(processed), Path(out_dir)
    _meta, kkt_log = read_json(root / "kkt_log.json")
    _meta_p, phase4 = read_json(root / "phase4.json")
    _meta_t, tolerance = read_json(root / "exp_unconstrained_tolerance.json")
    _meta_s, shrinkage = read_json(root / "exp_shrinkage.json")
    _meta_v, volatility = read_json(root / "exp_volatility_penalty.json")
    tickers = [str(t) for t in np.load(root / "forecasts_logsig.npz", allow_pickle=True)["tickers"]]

    written = [
        *plot_architecture(out / "fig1_architecture.png"),
        *plot_kkt_spread(kkt_log, phase4["no_trade_band"], out / "fig2_kkt_spread.png"),
        *plot_triggers(kkt_log, tickers, out / "fig3_trigger_heatmap.png"),
        *plot_sensitivity(
            tolerance, shrinkage, volatility, out / "fig4_sensitivity_and_extensions.png"
        ),
    ]
    return sorted(written)
