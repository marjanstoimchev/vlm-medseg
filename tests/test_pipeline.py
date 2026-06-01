"""Pipeline wiring test with a stub masker (no SAM2).

Confirms detector -> masker -> InstancePrediction -> metrics flows end to end
and that perfect boxes + perfect masks yield PQ = 1.
"""

import numpy as np

from vlm_medseg.detect.oracle import OracleBoxDetector
from vlm_medseg.pipeline.run import run_condition
from vlm_medseg.types import PanNukeSample


class PerfectMasker:
    """Returns the exact GT instance mask whose box matches each prompt box."""

    def __init__(self, sample: PanNukeSample):
        self.sample = sample

    def masks_from_boxes(self, image, boxes):
        from vlm_medseg.geometry import mask_to_box

        masks = []
        for b in boxes:
            best = None
            for i in self.sample.inst_classes:
                m = self.sample.inst_map == i
                gb = mask_to_box(m)
                # match by box equality (oracle boxes are exact)
                if gb is not None and np.allclose(gb, b):
                    best = m
                    break
            masks.append(best if best is not None else np.zeros(image.shape[:2], bool))
        return masks


def _sample():
    m = np.zeros((20, 20), np.int32)
    m[0:8, 0:8] = 1
    m[12:20, 12:20] = 2
    return PanNukeSample(
        image=np.zeros((20, 20, 3), np.uint8),
        inst_map=m,
        inst_classes={1: 0, 2: 1},
        group=3,
        sample_id="t-0",
    )


def test_pipeline_perfect_box_gives_pq_one():
    s = _sample()
    out = run_condition(
        OracleBoxDetector(class_aware=True), PerfectMasker(s), [s],
        progress=False, iou_thresh=0.5,
    )
    summary = out["summary"]
    assert summary["binary_pq"] == 1.0
    assert summary["detection_pooled"]["f1"] == 1.0
    assert summary["mpq"] == 1.0
    # both instances matched, classes correct -> clean diagonal confusion
    assert out["patch_metrics"][0]["confusion"].trace() == 2
