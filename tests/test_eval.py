"""Unit tests for the evaluation metrics on hand-constructed cases.

Each case has an analytically known answer so a regression in PQ/AJI/matching
is caught without any model or dataset.
"""

import numpy as np

from vlm_medseg.eval.aji import aggregated_jaccard_index, binary_dice
from vlm_medseg.eval.matching import instances_to_label_map, match_at_iou
from vlm_medseg.eval.pq import mpq_from_components, multiclass_pq, panoptic_quality


def _two_squares():
    """20x20 map: instance 1 = top-left 10x10, instance 2 = bottom-right 10x10."""
    m = np.zeros((20, 20), dtype=np.int32)
    m[0:10, 0:10] = 1
    m[10:20, 10:20] = 2
    return m


def test_perfect_match():
    gt = _two_squares()
    pred = gt.copy()
    pq = panoptic_quality(gt, pred)
    assert pq.tp == 2 and pq.fp == 0 and pq.fn == 0
    assert pq.sq == 1.0 and pq.dq == 1.0 and pq.pq == 1.0
    assert aggregated_jaccard_index(gt, pred) == 1.0
    assert binary_dice(gt, pred) == 1.0


def test_one_miss_one_spurious():
    gt = _two_squares()
    pred = np.zeros((20, 20), dtype=np.int32)
    pred[0:10, 0:10] = 1            # matches GT instance 1 exactly
    pred[0:5, 15:20] = 2            # spurious, overlaps nothing
    pq = panoptic_quality(gt, pred)
    assert (pq.tp, pq.fp, pq.fn) == (1, 1, 1)
    assert pq.dq == 0.5             # 1 / (1 + 0.5 + 0.5)
    assert pq.sq == 1.0
    assert pq.pq == 0.5


def test_half_iou():
    # GT 10x10 (100 px); pred is its left half (50 px) -> IoU = 50/100 = 0.5.
    gt = np.zeros((10, 10), dtype=np.int32)
    gt[:, :] = 1
    pred = np.zeros((10, 10), dtype=np.int32)
    pred[:, 0:5] = 1
    pq = panoptic_quality(gt, pred, iou_thresh=0.5)
    assert pq.tp == 1
    assert abs(pq.sq - 0.5) < 1e-9
    assert abs(aggregated_jaccard_index(gt, pred) - 0.5) < 1e-9
    assert abs(binary_dice(gt, pred) - (2 * 50 / 150)) < 1e-9


def test_below_threshold_is_unmatched():
    # IoU = 0.4 < 0.5 -> no match: one FN, one FP.
    gt = np.zeros((10, 10), dtype=np.int32)
    gt[:, 0:5] = 1     # 50 px
    pred = np.zeros((10, 10), dtype=np.int32)
    pred[:, 3:7] = 1  # 40 px, inter=20
    # inter=20, union=50+40-20=70, IoU=0.2857 -> unmatched
    m = match_at_iou(gt, pred, 0.5)
    assert m.tp == 0 and m.fp == 1 and m.fn == 1


def test_multiclass_pq_correct_vs_swapped():
    gt = _two_squares()
    gt_classes = {1: 0, 2: 1}            # instance 1 -> class 0, instance 2 -> class 1
    pred = gt.copy()
    # Correct classes -> per-class PQ = 1 for both present classes -> mPQ = 1.
    pred_correct = {1: 0, 2: 1}
    per_class = multiclass_pq(gt, gt_classes, pred, pred_correct, num_classes=5)
    assert abs(mpq_from_components(per_class) - 1.0) < 1e-9
    # Swapped classes -> same-class matching fails -> mPQ = 0.
    pred_swapped = {1: 1, 2: 0}
    per_class = multiclass_pq(gt, gt_classes, pred, pred_swapped, num_classes=5)
    assert mpq_from_components(per_class) == 0.0


def test_unique_matching_with_two_overlapping_preds():
    # One GT, two predictions overlapping it; only the better one is a TP.
    gt = np.zeros((10, 10), dtype=np.int32)
    gt[:, :] = 1
    masks = [np.zeros((10, 10), bool), np.zeros((10, 10), bool)]
    masks[0][:, 0:9] = True   # IoU 0.9
    masks[1][:, 0:3] = True   # IoU 0.3
    pred = instances_to_label_map(masks, (10, 10))
    m = match_at_iou(gt, pred, 0.5)
    assert m.tp == 1 and m.fp == 1 and m.fn == 0
    assert m.pairs[0][2] >= 0.5
