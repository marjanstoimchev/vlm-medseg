"""SAM3 promptable concept segmentation: text -> instances, end to end.

Unlike the SAM2 box-prompted masker, SAM3 takes a text concept (e.g. "cell
nucleus") and directly returns every matching instance's mask — detection and
segmentation in one model, no separate detector. It is therefore its own
pipeline *condition* (``sam3_text``), run via ``run_segmenter_condition``.

Needs the ``sam2`` extra (transformers>=4.57 carries Sam3Model). The cached
``facebook/sam3`` checkpoint loads into ``Sam3Model`` (a harmless video->image
class-mismatch warning).
"""

from __future__ import annotations

from ..data.base import DatasetSpec
from ..data.pannuke import PANNUKE_SPEC
from ..types import InstancePrediction, Sample
from .sam2 import resolve_device

DEFAULT_MODEL_ID = "facebook/sam3"


class Sam3TextSegmenter:
    """Text-prompted concept segmentation. One forward per class phrase."""

    name = "sam3_text"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "auto",
        dtype=None,
        spec: DatasetSpec = PANNUKE_SPEC,
        class_aware: bool = True,
        threshold: float = 0.5,        # SAM3's official default; lower to probe recall
        mask_threshold: float = 0.5,
    ) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.device = resolve_device(device)
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.spec = spec
        self.class_aware = class_aware
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()

        # SAM3 concept prompts (short noun phrases), per-class or one generic.
        if class_aware:
            self.prompts = [(spec.concept_prompt(c), c) for c in range(spec.num_classes)]
        else:
            self.prompts = [(spec.generic_concept_prompt, None)]

    def segment(self, sample: Sample) -> list[InstancePrediction]:
        import torch
        from PIL import Image

        h, w = sample.inst_map.shape
        pil = Image.fromarray(sample.image)
        instances: list[InstancePrediction] = []
        for phrase, class_id in self.prompts:
            inputs = self.processor(images=pil, text=phrase, return_tensors="pt").to(self.device, self.dtype)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.threshold, mask_threshold=self.mask_threshold,
                target_sizes=[(h, w)],
            )[0]
            for mask, score in zip(res["masks"], res["scores"]):
                m = mask.cpu().numpy().astype(bool)
                if m.any():
                    instances.append(InstancePrediction(mask=m, class_id=class_id, score=float(score)))
        return instances
