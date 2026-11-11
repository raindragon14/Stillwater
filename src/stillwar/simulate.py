"""Compatibility shim for the old ``stillwar.simulate`` import path.

New code should import from :mod:`stillwar.experiments.simulate`.
"""

from stillwar.experiments.simulate import (
    run_all,
    run_phase4,
    run_table1,
    run_table2,
    run_table3,
)

__all__ = ["run_all", "run_phase4", "run_table1", "run_table2", "run_table3"]
