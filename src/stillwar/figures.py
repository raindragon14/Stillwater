"""Compatibility shim for the old ``stillwar.figures`` import path.

New code should import from :mod:`stillwar.reporting.figures`.
"""

from stillwar.reporting.figures import (
    FigureStyle,
    apply_publication_style,
    plot_all,
    plot_architecture,
    plot_kkt_spread,
    plot_sensitivity,
    plot_triggers,
)

__all__ = [
    "FigureStyle",
    "apply_publication_style",
    "plot_all",
    "plot_architecture",
    "plot_kkt_spread",
    "plot_sensitivity",
    "plot_triggers",
]
