"""Panoptic Quality — the standard PanNuke instance-segmentation metric.

PQ decomposes into Detection Quality (DQ, an F1) and Segmentation Quality (SQ,
the mean IoU of matched pairs)::

    PQ = SQ * DQ
       = (sum_TP IoU / TP) * (TP / (TP + 0.5 FP + 0.5 FN))

We expose the raw components so a dataset summary can either average per-image
PQ (the official PanNuke convention, since each 256x256 patch is independent)
or pool components across patches first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matching import match_at_iou


@dataclass
class PQComponents:
    """Additive PQ accumulators; PQ/SQ/DQ are derived on demand."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0

    def __add__(self, other: PQComponents) -> PQComponents:
        return PQComponents(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.iou_sum + other.iou_sum,
        )

    @property
    def sq(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def dq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / denom if denom else 0.0

    @property
    def pq(self) -> float:
        return self.sq * self.dq

    def as_dict(self) -> dict:
        return {
            "pq": self.pq, "sq": self.sq, "dq": self.dq,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
        }


def panoptic_quality(
    gt_map: np.ndarray, pred_map: np.ndarray, iou_thresh: float = 0.5
) -> PQComponents:
    """PQ components for one patch (class-agnostic / binary)."""
    m = match_at_iou(gt_map, pred_map, iou_thresh)
    iou_sum = float(sum(iou for _, _, iou in m.pairs))
    return PQComponents(tp=m.tp, fp=m.fp, fn=m.fn, iou_sum=iou_sum)


def _class_submap(inst_map: np.ndarray, classes: dict[int, int], cls: int) -> np.ndarray:
    """Instance map containing only instances of ``cls``, relabeled 1..k."""
    out = np.zeros_like(inst_map)
    new = 0
    for inst_id, c in classes.items():
        if c == cls:
            new += 1
            out[inst_map == inst_id] = new
    return out


def multiclass_pq(
    gt_map: np.ndarray,
    gt_classes: dict[int, int],
    pred_map: np.ndarray,
    pred_classes: dict[int, int],
    num_classes: int,
    iou_thresh: float = 0.5,
) -> dict[int, PQComponents]:
    """Per-class PQ components. Matching is restricted to same-class instances.

    Returns ``{class_id: PQComponents}`` for every class in ``range(num_classes)``.
    The mean of the per-class PQ values is the multi-class PQ (mPQ).
    """
    out: dict[int, PQComponents] = {}
    for c in range(num_classes):
        gt_c = _class_submap(gt_map, gt_classes, c)
        pred_c = _class_submap(pred_map, pred_classes, c)
        out[c] = panoptic_quality(gt_c, pred_c, iou_thresh)
    return out


def mpq_from_components(per_class: dict[int, PQComponents], present_only: bool = True) -> float:
    """Mean PQ across classes. ``present_only`` skips classes with no GT and no pred."""
    vals = []
    for comp in per_class.values():
        if present_only and (comp.tp + comp.fp + comp.fn) == 0:
            continue
        vals.append(comp.pq)
    return float(np.mean(vals)) if vals else 0.0
