"""SAHI (Slicing Aided Hyper Inference) for any box detector.

Slice the patch into overlapping tiles, run the wrapped detector on each tile
(so it sees fewer, larger objects), remap boxes to patch coordinates, drop
near-whole-tile "region" boxes (the failure mode of stock VLMs/open-vocab
detectors), and merge cross-tile duplicates with class-agnostic NMS.

    SahiDetector(GroundingDinoDetector(...), tile=128, overlap=0.25)
"""

from __future__ import annotations

import numpy as np

from ..types import Detection, Sample


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if tile >= length:
        return [0]
    xs = list(range(0, length - tile + 1, stride))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    """Greedy class-agnostic NMS (one box per location, highest score wins)."""
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(_iou(d.box, k.box) <= iou_thr for k in keep):
            keep.append(d)
    return keep


class SahiDetector:
    """Tile-and-merge wrapper around a box detector."""

    def __init__(self, base, *, tile: int = 128, overlap: float = 0.25,
                 nms_iou: float = 0.5, max_box_frac: float = 0.9):
        self.base = base
        self.tile = tile
        self.overlap = overlap
        self.nms_iou = nms_iou
        self.max_box_frac = max_box_frac  # drop boxes covering >= this frac of a tile
        self.spec = getattr(base, "spec", None)
        self.name = f"sahi_{getattr(base, 'name', 'det')}"

    def detect(self, sample: Sample) -> list[Detection]:
        img = sample.image
        h, w = img.shape[:2]
        stride = max(1, int(round(self.tile * (1 - self.overlap))))
        out: list[Detection] = []
        for oy in _starts(h, self.tile, stride):
            for ox in _starts(w, self.tile, stride):
                th = min(self.tile, h - oy)
                tw = min(self.tile, w - ox)
                tile = Sample(
                    image=img[oy:oy + th, ox:ox + tw],
                    inst_map=np.zeros((th, tw), np.int32), inst_classes={},
                    sample_id=f"{sample.sample_id}-t{oy}_{ox}",
                    group=sample.group, dataset=sample.dataset,
                )
                for d in self.base.detect(tile):
                    if d.box is None:
                        continue
                    x1, y1, x2, y2 = d.box
                    if (x2 - x1) * (y2 - y1) >= self.max_box_frac * tw * th:
                        continue  # near-whole-tile region box, not a localized object
                    out.append(Detection(label=d.label, score=d.score, class_id=d.class_id,
                                         box=(x1 + ox, y1 + oy, x2 + ox, y2 + oy)))
        return _nms(out, self.nms_iou)
