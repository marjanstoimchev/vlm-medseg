"""Detector -> SAM2 -> instances pipeline and a per-condition runner.

A *condition* is one (detector, prompt-pathway) pair: ``la_box``, ``oracle_box``,
``gdino_box``, .... ``run_condition`` runs it over patches and optionally scores
it, picking up class/group names from the detector's :class:`DatasetSpec`.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence

from ..data.pannuke import PANNUKE_SPEC
from ..eval.report import evaluate_patch, summarize
from ..types import Detection, InstancePrediction, PatchResult, Sample


class Pipeline:
    """Turn a detector's localizations into SAM2 instance masks."""

    def __init__(self, detector, masker, *, point_select: str = "score") -> None:
        self.detector = detector
        self.masker = masker
        self.point_select = point_select
        self.condition = getattr(detector, "name", "detector")

    def segment(self, sample: Sample, detections: list[Detection]) -> list[InstancePrediction]:
        boxes = [d for d in detections if d.box is not None]
        points = [d for d in detections if d.point is not None and d.box is None]
        instances: list[InstancePrediction] = []

        if boxes:
            masks = self.masker.masks_from_boxes(sample.image, [d.box for d in boxes])
            instances += [
                InstancePrediction(mask=m, class_id=d.class_id, score=d.score, box=d.box)
                for d, m in zip(boxes, masks) if m.any()
            ]
        if points:
            masks, scores = self.masker.masks_from_points(
                sample.image, [d.point for d in points], select=self.point_select
            )
            instances += [
                InstancePrediction(mask=m, class_id=d.class_id, score=s, point=d.point)
                for d, m, s in zip(points, masks, scores) if m.any()
            ]
        return instances

    def run_patch(self, sample: Sample, detections: list[Detection] | None = None) -> PatchResult:
        t0 = time.perf_counter()
        if detections is None:
            detections = self.detector.detect(sample)
        det_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        instances = self.segment(sample, detections)
        return PatchResult(
            sample_id=sample.sample_id, condition=self.condition,
            detections=detections, instances=instances,
            detector_seconds=det_s, masker_seconds=time.perf_counter() - t1,
        )


def run_condition(
    detector,
    masker,
    samples: Sequence[Sample] | Iterable[Sample],
    *,
    spec=None,
    precomputed: dict[str, list[Detection]] | None = None,
    classifier=None,
    evaluate: bool = True,
    num_classes: int | None = None,
    iou_thresh: float = 0.5,
    point_select: str = "score",
    on_result=None,
    progress: bool = True,
) -> dict:
    """Run one condition over patches.

    ``precomputed`` (sample_id -> detections) skips the detector — the path used
    to re-segment/re-evaluate cached LocateAnything boxes offline. ``classifier``
    (a ``NucleusClassifier``) overrides each instance's class from its pixels.
    Returns ``{condition, results, patch_metrics, summary}``.
    """
    if spec is None:
        spec = getattr(detector, "spec", None) or PANNUKE_SPEC
    if num_classes is None:
        num_classes = spec.num_classes

    pipe = Pipeline(detector, masker, point_select=point_select)
    samples = list(samples)
    iterator = samples
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(samples, desc=pipe.condition)
        except Exception:
            iterator = samples

    results: list[PatchResult] = []
    patch_metrics: list[dict] = []
    for sample in iterator:
        dets = precomputed.get(sample.sample_id) if precomputed else None
        res = pipe.run_patch(sample, detections=dets)
        if classifier is not None:
            classifier.classify_instances(sample.image, res.instances)
        results.append(res)
        if evaluate:
            patch_metrics.append(
                evaluate_patch(sample, res.instances, num_classes=num_classes, iou_thresh=iou_thresh)
            )
        if on_result is not None:
            on_result(sample, res)

    out = {"condition": pipe.condition, "results": results}
    if evaluate:
        out["patch_metrics"] = patch_metrics
        out["summary"] = summarize(
            patch_metrics, num_classes=num_classes,
            class_names=spec.class_names, group_names=spec.group_names,
        )
    return out


def run_segmenter_condition(
    segmenter,
    samples: Sequence[Sample] | Iterable[Sample],
    *,
    spec=None,
    classifier=None,
    evaluate: bool = True,
    num_classes: int | None = None,
    iou_thresh: float = 0.5,
    on_result=None,
    progress: bool = True,
) -> dict:
    """Run an end-to-end segmenter (e.g. SAM3 text) that maps a patch straight
    to instances, with no separate detector. ``classifier`` (a
    ``NucleusClassifier``) overrides each instance's class from its pixels —
    the recommended path for SAM3, whose text concepts don't separate subtypes.
    Same return shape as :func:`run_condition`.
    """
    if spec is None:
        spec = getattr(segmenter, "spec", None) or PANNUKE_SPEC
    if num_classes is None:
        num_classes = spec.num_classes
    condition = getattr(segmenter, "name", "segmenter")

    samples = list(samples)
    iterator = samples
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(samples, desc=condition)
        except Exception:
            iterator = samples

    results: list[PatchResult] = []
    patch_metrics: list[dict] = []
    for sample in iterator:
        t0 = time.perf_counter()
        instances = segmenter.segment(sample)
        if classifier is not None:
            classifier.classify_instances(sample.image, instances)
        res = PatchResult(
            sample_id=sample.sample_id, condition=condition,
            instances=instances, masker_seconds=time.perf_counter() - t0,
        )
        results.append(res)
        if evaluate:
            patch_metrics.append(
                evaluate_patch(sample, instances, num_classes=num_classes, iou_thresh=iou_thresh)
            )
        if on_result is not None:
            on_result(sample, res)

    out = {"condition": condition, "results": results}
    if evaluate:
        out["patch_metrics"] = patch_metrics
        out["summary"] = summarize(
            patch_metrics, num_classes=num_classes,
            class_names=spec.class_names, group_names=spec.group_names,
        )
    return out
