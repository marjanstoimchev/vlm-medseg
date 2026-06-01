"""Framework-wide data structures (numpy-only, dependency-light)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A box is (x1, y1, x2, y2) in pixel coordinates; a point is (x, y).
Box = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass
class Detection:
    """A single localization emitted by a detector (before SAM2)."""

    label: str
    score: float = 1.0
    class_id: int | None = None     # mapped class (None = class-agnostic)
    box: Box | None = None
    point: Point | None = None


@dataclass
class InstancePrediction:
    """A predicted instance: a binary mask plus its class and provenance."""

    mask: np.ndarray                # (H, W) bool
    class_id: int | None = None
    score: float = 1.0
    box: Box | None = None
    point: Point | None = None


@dataclass
class Sample:
    """One decoded instance-segmentation patch, ready for pipeline + evaluation.

    ``group`` is an optional stratification key (PanNuke tissue type, an organ,
    a scanner, ...); ``-1`` means ungrouped. ``meta`` carries any extra fields.
    """

    image: np.ndarray               # (H, W, 3) uint8
    inst_map: np.ndarray            # (H, W) int32 instance labels, 0 = bg
    inst_classes: dict[int, int]    # instance label -> class id
    sample_id: str
    group: int = -1
    dataset: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def num_instances(self) -> int:
        return len(self.inst_classes)

    @property
    def tissue(self) -> int:
        """Back-compat alias for :attr:`group` (PanNuke tissue id)."""
        return self.group

    def class_map(self) -> np.ndarray:
        """(H, W) int8 semantic map: -1 background, else class id."""
        out = np.full(self.inst_map.shape, -1, dtype=np.int8)
        for inst_id, cls in self.inst_classes.items():
            out[self.inst_map == inst_id] = cls
        return out


# Back-compat alias: the framework is dataset-agnostic, but PanNuke-era code and
# tests refer to PanNukeSample.
PanNukeSample = Sample


@dataclass
class PatchResult:
    """Per-patch pipeline output for one condition (e.g. 'la_box')."""

    sample_id: str
    condition: str
    detections: list[Detection] = field(default_factory=list)
    instances: list[InstancePrediction] = field(default_factory=list)
    detector_seconds: float = 0.0
    masker_seconds: float = 0.0
