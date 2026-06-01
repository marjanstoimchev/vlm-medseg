"""Decomposed evaluation: matching, PQ/SQ/DQ, AJI, Dice, detection, classification."""

from .aji import aggregated_jaccard_index, binary_dice, matched_mask_iou
from .classification import classification_report, confusion_over_matches
from .detection import counting_error, detection_scores
from .matching import Matching, instances_to_label_map, iou_matrix, match_at_iou
from .pq import PQComponents, mpq_from_components, multiclass_pq, panoptic_quality
from .report import evaluate_patch, summarize

__all__ = [
    "Matching", "match_at_iou", "iou_matrix", "instances_to_label_map",
    "PQComponents", "panoptic_quality", "multiclass_pq", "mpq_from_components",
    "aggregated_jaccard_index", "binary_dice", "matched_mask_iou",
    "detection_scores", "counting_error",
    "confusion_over_matches", "classification_report",
    "evaluate_patch", "summarize",
]
