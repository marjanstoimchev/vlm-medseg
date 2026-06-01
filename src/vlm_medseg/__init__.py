"""vlm-medseg: VLM-driven instance segmentation for medical images.

A grounding VLM (LocateAnything-3B) localizes objects as boxes, SAM2 turns those
boxes into masks, and a decomposed evaluation suite (PQ / AJI / Dice / detection
F1) scores the result — with ground-truth-prompted *oracle* conditions that
isolate the SAM2 ceiling from the VLM's detection bottleneck. PanNuke is the
reference dataset; any instance-seg dataset plugs in via ``DatasetSpec`` /
``BaseDataset`` (see ``vlm_medseg.data``).

The top-level import is torch-free; import heavy stacks from their submodules:

    from vlm_medseg.segment.sam2 import Sam2Masker
    from vlm_medseg.detect.locate_anything import LocateAnythingDetector
"""

from __future__ import annotations

from . import constants
from .data import DatasetSpec, available_datasets, get_dataset, register_dataset
from .prompts import PROMPTS
from .types import (
    Box,
    Detection,
    InstancePrediction,
    PanNukeSample,
    PatchResult,
    Point,
    Sample,
)

__version__ = "0.1.0"

__all__ = [
    "constants",
    "Box", "Point", "Detection", "InstancePrediction", "Sample", "PanNukeSample",
    "PatchResult",
    "DatasetSpec", "get_dataset", "register_dataset", "available_datasets",
    "PROMPTS",
    "__version__",
]
