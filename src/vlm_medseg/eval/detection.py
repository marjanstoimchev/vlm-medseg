"""Detection-level scoring: precision / recall / F1 and counting error.

These read directly off the instance matching, so "a detection" here means an
instance whose mask overlaps a GT instance above the IoU threshold — the same
TP/FP/FN used by PQ's DQ term, exposed in detector-friendly terms.
"""

from __future__ import annotations


def detection_scores(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


def counting_error(n_gt: int, n_pred: int) -> dict:
    """How far off the instance count is — VLMs systematically undercount."""
    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "abs_error": abs(n_pred - n_gt),
        "signed_error": n_pred - n_gt,
        "ratio": (n_pred / n_gt) if n_gt else float("inf") if n_pred else 1.0,
    }
