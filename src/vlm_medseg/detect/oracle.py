"""Oracle detectors: ground-truth box/point prompts that set the SAM2 ceiling.

Not real detectors — they read the GT maps to emit a perfect prompt per
instance, answering "given perfect localization, how well can SAM2 segment?"
"""

from __future__ import annotations

from ..data.base import DatasetSpec
from ..data.pannuke import PANNUKE_SPEC
from ..geometry import instance_boxes, instance_points
from ..types import Detection, Sample


class OracleBoxDetector:
    name = "oracle_box"

    def __init__(self, class_aware: bool = True, spec: DatasetSpec = PANNUKE_SPEC) -> None:
        self.class_aware = class_aware
        self.spec = spec

    def detect(self, sample: Sample) -> list[Detection]:
        out: list[Detection] = []
        for inst_id, box in instance_boxes(sample.inst_map).items():
            c = sample.inst_classes.get(inst_id)
            out.append(Detection(
                label=self.spec.class_name(c) if c is not None else "object",
                class_id=c if self.class_aware else None, box=box,
            ))
        return out


class OraclePointDetector:
    name = "oracle_point"

    def __init__(self, class_aware: bool = True, spec: DatasetSpec = PANNUKE_SPEC) -> None:
        self.class_aware = class_aware
        self.spec = spec

    def detect(self, sample: Sample) -> list[Detection]:
        out: list[Detection] = []
        for inst_id, pt in instance_points(sample.inst_map).items():
            c = sample.inst_classes.get(inst_id)
            out.append(Detection(
                label=self.spec.class_name(c) if c is not None else "object",
                class_id=c if self.class_aware else None, point=pt,
            ))
        return out
