"""Feature package: log-signature primitives and the standardised dataset."""

from stillwater.features.dataset import Dataset, build_dataset, load_dataset, split_indices
from stillwater.features.logsig import (
    BLOCK,
    INCREMENT_SLOTS,
    SEQ_LEN,
    compute_increments_and_levy_areas,
    logsig_features,
)

__all__ = [
    "BLOCK",
    "INCREMENT_SLOTS",
    "SEQ_LEN",
    "Dataset",
    "build_dataset",
    "compute_increments_and_levy_areas",
    "load_dataset",
    "logsig_features",
    "split_indices",
]
