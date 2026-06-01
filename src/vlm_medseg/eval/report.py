"""Per-patch evaluation and dataset-level aggregation.

``evaluate_patch`` is the single bridge from (sample, predicted instances) to
every metric; ``summarize`` rolls per-patch results into the headline numbers,
stratified by group (e.g. tissue) and class. Both are dataset-agnostic: class /
group display names are passed in as plain lists.
"""

from __future__ import annotations

import numpy as np

from ..types import InstancePrediction, Sample
from .aji import aggregated_jaccard_index, binary_dice, matched_mask_iou
from .classification import classification_report, confusion_over_matches
from .detection import counting_error, detection_scores
from .matching import instances_to_label_map, match_at_iou
from .pq import mpq_from_components, multiclass_pq, panoptic_quality


def evaluate_patch(
    sample: Sample,
    instances: list[InstancePrediction],
    *,
    num_classes: int,
    iou_thresh: float = 0.5,
    class_aware: bool | None = None,
) -> dict:
    """All metrics for one patch.

    ``class_aware`` defaults to True iff every predicted instance carries a class
    id; otherwise only binary/PQ metrics are computed.
    """
    shape = sample.inst_map.shape
    pred_map = instances_to_label_map([ins.mask for ins in instances], shape)
    gt_map = sample.inst_map

    if class_aware is None:
        class_aware = bool(instances) and all(i.class_id is not None for i in instances)

    bpq = panoptic_quality(gt_map, pred_map, iou_thresh)
    mq = matched_mask_iou(gt_map, pred_map, iou_thresh)

    out: dict = {
        "sample_id": sample.sample_id,
        "group": sample.group,
        "binary": {
            **bpq.as_dict(),
            "aji": aggregated_jaccard_index(gt_map, pred_map),
            "dice": binary_dice(gt_map, pred_map),
            "matched_iou": mq["mean_iou"],
            "matched_dice": mq["mean_dice"],
            "n_matched": mq["n"],
        },
        "detection": detection_scores(bpq.tp, bpq.fp, bpq.fn),
        "counting": counting_error(len(sample.inst_classes), len(instances)),
        "class_aware": class_aware,
    }

    if class_aware:
        pred_classes = {i + 1: int(ins.class_id) for i, ins in enumerate(instances)}
        per_class = multiclass_pq(
            gt_map, sample.inst_classes, pred_map, pred_classes, num_classes, iou_thresh
        )
        out["per_class_pq"] = {c: comp.as_dict() for c, comp in per_class.items()}
        out["mpq"] = mpq_from_components(per_class)
        m = match_at_iou(gt_map, pred_map, iou_thresh)
        out["confusion"] = confusion_over_matches(
            m, sample.inst_classes, pred_classes, num_classes
        )
    return out


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize(
    patch_metrics: list[dict],
    *,
    num_classes: int | None = None,
    class_names: list[str] | None = None,
    group_names: list[str] | None = None,
) -> dict:
    """Aggregate per-patch metrics.

    PQ/AJI/Dice are averaged per patch (PanNuke convention); detection P/R/F1
    pools TP/FP/FN; classification pools the confusion. ``class_names`` /
    ``group_names`` label the per-class and per-group breakdowns (indices if None).
    """
    if not patch_metrics:
        return {"n_patches": 0}

    def cname(c):
        return class_names[c] if class_names and c < len(class_names) else str(c)

    def gname(g):
        return group_names[g] if group_names and 0 <= g < len(group_names) else str(g)

    tp = sum(m["binary"]["tp"] for m in patch_metrics)
    fp = sum(m["binary"]["fp"] for m in patch_metrics)
    fn = sum(m["binary"]["fn"] for m in patch_metrics)

    summary: dict = {
        "n_patches": len(patch_metrics),
        "binary_pq": _mean([m["binary"]["pq"] for m in patch_metrics]),
        "sq": _mean([m["binary"]["sq"] for m in patch_metrics]),
        "dq": _mean([m["binary"]["dq"] for m in patch_metrics]),
        "aji": _mean([m["binary"]["aji"] for m in patch_metrics]),
        "dice": _mean([m["binary"]["dice"] for m in patch_metrics]),
        "matched_iou": _mean(
            [m["binary"]["matched_iou"] for m in patch_metrics if m["binary"]["n_matched"]]
        ),
        "detection_pooled": detection_scores(tp, fp, fn),
        "counting": {
            "mean_abs_error": _mean([m["counting"]["abs_error"] for m in patch_metrics]),
            "mean_signed_error": _mean([m["counting"]["signed_error"] for m in patch_metrics]),
            "total_gt": sum(m["counting"]["n_gt"] for m in patch_metrics),
            "total_pred": sum(m["counting"]["n_pred"] for m in patch_metrics),
        },
    }

    groups = {m["group"] for m in patch_metrics}
    if groups != {-1}:
        by_group: dict[int, list[float]] = {}
        for m in patch_metrics:
            by_group.setdefault(m["group"], []).append(m["binary"]["pq"])
        summary["by_group"] = {
            gname(g): {"binary_pq": _mean(v), "n": len(v)}
            for g, v in sorted(by_group.items())
        }

    if all(m.get("class_aware") for m in patch_metrics):
        nc = num_classes or len(patch_metrics[0]["per_class_pq"])
        summary["mpq"] = _mean([m["mpq"] for m in patch_metrics])
        summary["per_class_pq"] = {
            cname(c): _mean([
                m["per_class_pq"][c]["pq"]
                for m in patch_metrics
                if sum(m["per_class_pq"][c][k] for k in ("tp", "fp", "fn")) > 0
            ])
            for c in range(nc)
        }
        conf = np.zeros((nc, nc), dtype=np.int64)
        for m in patch_metrics:
            conf += m["confusion"]
        summary["classification"] = classification_report(conf)
        summary["confusion_matrix"] = conf.tolist()

    return summary
