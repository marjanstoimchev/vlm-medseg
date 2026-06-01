"""Grounding-DINO open-vocabulary detector — a non-VLM Grounded-SAM baseline.

Runs locally (HF transformers, MPS/CPU/CUDA); weights are pulled from the Hub.
"""

from __future__ import annotations

from ..data.base import DatasetSpec
from ..data.pannuke import PANNUKE_SPEC
from ..prompts import grounding_dino_text
from ..segment.sam2 import resolve_device
from ..types import Detection, Sample


class GroundingDinoDetector:
    name = "gdino_box"

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        *,
        device: str = "auto",
        class_aware: bool = True,
        spec: DatasetSpec = PANNUKE_SPEC,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = resolve_device(device)
        self.class_aware = class_aware
        self.spec = spec
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        )

        phrases = list(spec.class_prompts.values()) if class_aware else [spec.generic_prompt]
        self.text = grounding_dino_text(phrases)
        self.label_to_class: dict[str, int] = {}
        if class_aware:
            for cid, phrase in spec.class_prompts.items():
                self.label_to_class[phrase.lower()] = cid
                self.label_to_class[spec.class_name(cid).lower()] = cid

    def _class_for(self, label: str) -> int | None:
        if not self.class_aware:
            return None
        lab = label.strip().lower()
        for key, cid in self.label_to_class.items():
            if key in lab or lab in key:
                return cid
        return None

    def detect(self, sample: Sample) -> list[Detection]:
        import torch
        from PIL import Image

        h, w = sample.inst_map.shape
        inputs = self.processor(
            images=Image.fromarray(sample.image), text=self.text, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]

        # transformers >=4.51 returns integer `labels`; the string phrases moved
        # to `text_labels`. Prefer those so class mapping keeps working.
        labels = results.get("text_labels")
        if labels is None:
            labels = results["labels"]

        dets: list[Detection] = []
        for box, score, label in zip(results["boxes"], results["scores"], labels):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            dets.append(Detection(label=str(label), score=float(score),
                                  class_id=self._class_for(str(label)), box=(x1, y1, x2, y2)))
        return dets
