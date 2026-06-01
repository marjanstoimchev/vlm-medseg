"""Aggregated Jaccard Index (AJI, Kumar et al. / HoVer-Net) and Dice."""

from __future__ import annotations

import numpy as np

from .matching import label_overlap


def aggregated_jaccard_index(gt_map: np.ndarray, pred_map: np.ndarray) -> float:
    n_gt = int(gt_map.max())
    n_pred = int(pred_map.max())
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0

    overlap = label_overlap(gt_map, pred_map)  # (n_gt+1, n_pred+1)
    gt_area = overlap.sum(axis=1)              # incl. bg at index 0
    pred_area = overlap.sum(axis=0)
    inter = overlap[1:, 1:].astype(np.float64)             # (n_gt, n_pred)
    union = gt_area[1:, None] + pred_area[None, 1:] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)

    overall_inter = 0.0
    overall_union = 0.0
    paired_pred: set[int] = set()
    for i in range(n_gt):
        j = int(np.argmax(iou[i]))
        if iou[i, j] <= 0.0:
            overall_union += gt_area[i + 1]   # unmatched GT
            continue
        overall_inter += inter[i, j]
        overall_union += union[i, j]
        paired_pred.add(j)

    for j in range(n_pred):
        if j not in paired_pred:
            overall_union += pred_area[j + 1]  # spurious prediction penalty

    return overall_inter / overall_union if overall_union > 0 else 0.0


def binary_dice(gt_map: np.ndarray, pred_map: np.ndarray) -> float:
    """Pixel-level (ensemble) Dice over the foreground of both maps."""
    g = gt_map > 0
    p = pred_map > 0
    denom = g.sum() + p.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * (g & p).sum() / denom)


def matched_mask_iou(gt_map: np.ndarray, pred_map: np.ndarray, iou_thresh: float = 0.5):
    """Mean IoU and Dice over matched instance pairs only.

    Isolates segmentation quality given a correct detection — the key number for
    judging SAM2 in the oracle conditions.
    """
    from .matching import match_at_iou

    m = match_at_iou(gt_map, pred_map, iou_thresh)
    if not m.pairs:
        return {"mean_iou": 0.0, "mean_dice": 0.0, "n": 0}
    ious = np.array([iou for _, _, iou in m.pairs])
    dices = 2 * ious / (1 + ious)  # Dice from IoU for the matched pair
    return {
        "mean_iou": float(ious.mean()),
        "mean_dice": float(dices.mean()),
        "n": len(m.pairs),
    }
