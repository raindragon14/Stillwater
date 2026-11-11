"""Compatibility shim for the old ``stillwar.tables`` import path.

New code should import from :mod:`stillwar.reporting.tables`.
"""

from stillwar.reporting.tables import (
    GOLDEN_ARTIFACTS,
    JOURNAL,
    bless_golden,
    load_blocks,
    render_table1,
    render_table2,
    render_table3,
    render_table_s1,
    sync_manuscript,
)

__all__ = [
    "GOLDEN_ARTIFACTS",
    "JOURNAL",
    "bless_golden",
    "load_blocks",
    "render_table1",
    "render_table2",
    "render_table3",
    "render_table_s1",
    "sync_manuscript",
]
