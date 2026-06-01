"""Per-class confusion over correctly-detected (matched) instances.

Only matched TP pairs contribute: among nuclei the pipeline actually found,
how often is the VLM-assigned class right? This isolates classification skill
from detection recall.
"""

from __future__ import annotations

import numpy as np

from .matching import Matching


def confusion_over_matches(
    matching: Matching,
    gt_class_by_id: dict[int, int],
    pred_class_by_id: dict[int, int],
    num_classes: int,
) -> np.ndarray:
    """``(num_classes, num_classes)`` confusion counts indexed ``[gt, pred]``.

    Matching ``pairs`` carry 0-based indices; instance id = index + 1.
    """
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt_idx, pred_idx, _ in matching.pairs:
        gc = gt_class_by_id.get(gt_idx + 1)
        pc = pred_class_by_id.get(pred_idx + 1)
        if gc is None or pc is None or gc < 0 or pc < 0:
            continue
        conf[gc, pc] += 1
    return conf


def classification_report(conf: np.ndarray) -> dict:
    """Accuracy and per-class precision/recall/F1 from a confusion matrix."""
    total = conf.sum()
    correct = np.trace(conf)
    accuracy = float(correct / total) if total else 0.0

    per_class = {}
    for c in range(conf.shape[0]):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                        "support": int(conf[c, :].sum())}
    return {"accuracy": accuracy, "per_class": per_class, "n_matched": int(total)}
