"""Reporting package: manuscript tables, figures and synchronisation."""

from stillwar.reporting.figures import plot_all
from stillwar.reporting.tables import (
    bless_golden_values,
    load_blocks,
    render_table1,
    render_table2,
    render_table3,
    render_table_s1,
    sync_manuscript,
)

__all__ = [
    "bless_golden_values",
    "load_blocks",
    "plot_all",
    "render_table1",
    "render_table2",
    "render_table3",
    "render_table_s1",
    "sync_manuscript",
]
