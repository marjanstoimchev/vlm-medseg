"""Dataset abstraction (any instance-seg dataset) + the PanNuke reference impl."""

from .base import BaseDataset, DatasetSpec
from .pannuke import (
    PANNUKE_SPEC,
    PanNukeDataset,
    decode_row,
    decode_sample,
    load_pannuke,
    stratified_indices,
    tissue_distribution,
)
from .registry import available_datasets, get_dataset, register_dataset

__all__ = [
    "BaseDataset", "DatasetSpec",
    "get_dataset", "register_dataset", "available_datasets",
    "PanNukeDataset", "PANNUKE_SPEC",
    "load_pannuke", "decode_sample", "decode_row",
    "stratified_indices", "tissue_distribution",
]
