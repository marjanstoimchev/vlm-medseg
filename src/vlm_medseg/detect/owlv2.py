"""OWLv2 open-vocabulary detector (text-prompt) — a cached, local baseline.

Like Grounding-DINO: text queries in, boxes out. OWLv2 returns integer labels
indexing the query list, so class mapping is positional. Pairs well with SAHI
(``--sahi``) for the dense-nuclei regime.
"""

from __future__ import annotations

from ..data.base import DatasetSpec
from ..data.pannuke import PANNUKE_SPEC
from ..segment.sam2 import resolve_device
from ..types import Detection, Sample


class Owlv2Detector:
    name = "owlv2_box"

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16",
        *,
        device: str = "auto",
        class_aware: bool = True,
        spec: DatasetSpec = PANNUKE_SPEC,
        threshold: float = 0.1,
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = resolve_device(device)
        self.class_aware = class_aware
        self.spec = spec
        self.threshold = threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        )
        # Query i corresponds to class i (class-aware) or the generic phrase.
        self.queries = (
            [spec.class_prompts[c] for c in range(spec.num_classes)]
            if class_aware else [spec.generic_prompt]
        )

    def detect(self, sample: Sample) -> list[Detection]:
        import torch
        from PIL import Image

        h, w = sample.inst_map.shape
        inputs = self.processor(
            text=[self.queries], images=Image.fromarray(sample.image), return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]

        dets: list[Detection] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            cid = int(label) if self.class_aware else None
            if cid is not None and not (0 <= cid < self.spec.num_classes):
                cid = None
            dets.append(Detection(
                label=self.spec.class_name(cid) if cid is not None else str(int(label)),
                score=float(score), class_id=cid, box=(x1, y1, x2, y2),
            ))
        return dets
