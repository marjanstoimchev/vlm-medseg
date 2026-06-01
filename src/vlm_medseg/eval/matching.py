"""Instance matching via a fast label-overlap histogram, shared by PQ/AJI/detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def label_overlap(gt_map: np.ndarray, pred_map: np.ndarray) -> np.ndarray:
    """Pixel-intersection histogram.

    Returns an ``(n_gt + 1, n_pred + 1)`` int64 array where entry ``[i, j]`` is
    the number of pixels labeled ``i`` in ``gt_map`` and ``j`` in ``pred_map``.
    Row/column 0 are background and are retained for area bookkeeping.
    """
    gt = gt_map.ravel()
    pred = pred_map.ravel()
    n_gt = int(gt_map.max())
    n_pred = int(pred_map.max())
    overlap = np.zeros((n_gt + 1, n_pred + 1), dtype=np.int64)
    np.add.at(overlap, (gt, pred), 1)
    return overlap


def iou_matrix(gt_map: np.ndarray, pred_map: np.ndarray) -> np.ndarray:
    """IoU between every GT and predicted instance, shape ``(n_gt, n_pred)``.

    Background (index 0) is excluded from the returned matrix.
    """
    overlap = label_overlap(gt_map, pred_map)
    gt_area = overlap.sum(axis=1, keepdims=True)   # (n_gt+1, 1)
    pred_area = overlap.sum(axis=0, keepdims=True)  # (1, n_pred+1)
    union = gt_area + pred_area - overlap
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, overlap / union, 0.0)
    return iou[1:, 1:]  # drop background row/col


@dataclass
class Matching:
    """Result of matching predicted instances to GT instances.

    ``pairs`` holds 0-based ``(gt_idx, pred_idx, iou)`` triples. ``unmatched_gt``
    are false negatives, ``unmatched_pred`` are false positives.
    """

    pairs: list[tuple[int, int, float]]
    unmatched_gt: list[int]
    unmatched_pred: list[int]
    n_gt: int
    n_pred: int

    @property
    def tp(self) -> int:
        return len(self.pairs)

    @property
    def fp(self) -> int:
        return len(self.unmatched_pred)

    @property
    def fn(self) -> int:
        return len(self.unmatched_gt)


def match_at_iou(
    gt_map: np.ndarray, pred_map: np.ndarray, iou_thresh: float = 0.5
) -> Matching:
    """Match instances at an IoU threshold.

    For ``iou_thresh >= 0.5`` each GT can match at most one prediction and vice
    versa (the panoptic-quality regime), so a greedy highest-IoU assignment is
    optimal. Below 0.5 we still match greedily by descending IoU, which is the
    common convention for detection-style scoring.
    """
    n_gt = int(gt_map.max())
    n_pred = int(pred_map.max())
    if n_gt == 0 or n_pred == 0:
        return Matching([], list(range(n_gt)), list(range(n_pred)), n_gt, n_pred)

    iou = iou_matrix(gt_map, pred_map)
    pairs: list[tuple[int, int, float]] = []
    gt_taken = np.zeros(n_gt, dtype=bool)
    pred_taken = np.zeros(n_pred, dtype=bool)

    # Candidate pairs above threshold, sorted by IoU descending.
    gi, pj = np.where(iou >= iou_thresh)
    order = np.argsort(-iou[gi, pj])
    for k in order:
        i, j = int(gi[k]), int(pj[k])
        if gt_taken[i] or pred_taken[j]:
            continue
        gt_taken[i] = pred_taken[j] = True
        pairs.append((i, j, float(iou[i, j])))

    unmatched_gt = [i for i in range(n_gt) if not gt_taken[i]]
    unmatched_pred = [j for j in range(n_pred) if not pred_taken[j]]
    return Matching(pairs, unmatched_gt, unmatched_pred, n_gt, n_pred)


def relabel_consecutive(mask_label: np.ndarray) -> np.ndarray:
    """Relabel an instance map so ids are 1..N with no gaps (0 stays bg)."""
    ids = np.unique(mask_label)
    ids = ids[ids != 0]
    out = np.zeros_like(mask_label)
    for new, old in enumerate(ids, start=1):
        out[mask_label == old] = new
    return out


def instances_to_label_map(
    masks: list[np.ndarray], shape: tuple[int, int]
) -> np.ndarray:
    """Stack a list of boolean masks into a single instance label map.

    Later masks overwrite earlier ones on overlap (predicted nuclei rarely
    overlap; when they do, this keeps the map a clean partition).
    """
    out = np.zeros(shape, dtype=np.int32)
    for i, m in enumerate(masks, start=1):
        out[m.astype(bool)] = i
    return out
