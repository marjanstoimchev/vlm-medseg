"""Base Qwen2.5-VL as a grounding detector — the comparison point for LA.

LocateAnything-3B is a *fine-tuned* grounding head on a Qwen2.5-3B backbone; this
runs the stock **Qwen2.5-VL** (3B by default, same scale) on the identical
prompts so you can measure what LA's localization training actually buys. Output
is parsed from Qwen's JSON ``bbox_2d`` grounding format.

Qwen2.5-VL emits boxes in the *resized* input pixel space; we map them back to
the patch via ``image_grid_thw`` (``coord_space="qwen_abs"``). Older Qwen2-VL
checkpoints use 0-1000 normalized coords (``coord_space="norm1000"``).
"""

from __future__ import annotations

import json
import re

from ..data.base import DatasetSpec
from ..data.pannuke import PANNUKE_SPEC
from ..prompts import qwen_grounding_query
from ..segment.sam2 import resolve_device
from ..types import Detection, Sample
from .locate_anything import _label_to_class, build_label_to_class

_BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[([^\]]+)\]')
_LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]*)"')
_NUM_RE = re.compile(r"-?\d+\.?\d*")


def grounding_query(category_phrases: list[str]) -> str:
    return qwen_grounding_query(category_phrases)


def parse_qwen_boxes(
    text: str,
    scale_x: float,
    scale_y: float,
    *,
    width: int | None = None,
    height: int | None = None,
    label_to_class: dict[str, int] | None = None,
    default_label: str = "object",
) -> list[Detection]:
    """Parse Qwen JSON grounding output, scaling boxes by (scale_x, scale_y).

    Tries strict JSON first, then a tolerant regex over ``bbox_2d``/``label``
    pairs for slightly malformed output. If ``width``/``height`` are given, boxes
    are clamped to the image (Qwen sometimes emits coords past the resized size).
    """
    label_to_class = label_to_class or {}
    items: list[tuple[list[float], str]] = []

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            for obj in json.loads(text[start : end + 1]):
                if isinstance(obj, dict) and "bbox_2d" in obj:
                    items.append((obj["bbox_2d"], str(obj.get("label", default_label))))
        except (json.JSONDecodeError, TypeError):
            items = []

    if not items:  # tolerant fallback
        boxes = _BBOX_RE.findall(text)
        labels = _LABEL_RE.findall(text)
        for i, b in enumerate(boxes):
            nums = [float(n) for n in _NUM_RE.findall(b)]
            if len(nums) >= 4:
                items.append((nums[:4], labels[i] if i < len(labels) else default_label))

    dets: list[Detection] = []
    for bbox, label in items:
        nums = [float(v) for v in bbox][:4]
        if len(nums) < 4:
            continue
        x1, y1, x2, y2 = nums
        bx1, by1 = min(x1, x2) * scale_x, min(y1, y2) * scale_y
        bx2, by2 = max(x1, x2) * scale_x, max(y1, y2) * scale_y
        if width is not None:
            bx1, bx2 = max(0.0, min(bx1, width)), max(0.0, min(bx2, width))
        if height is not None:
            by1, by2 = max(0.0, min(by1, height)), max(0.0, min(by2, height))
        if bx2 - bx1 >= 1 and by2 - by1 >= 1:
            dets.append(Detection(label=label, class_id=_label_to_class(label, label_to_class),
                                  box=(bx1, by1, bx2, by2)))
    return dets


class QwenVLDetector:
    """Stock Qwen2.5-VL grounding detector (box pathway)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        *,
        device: str = "auto",
        dtype=None,
        class_aware: bool = True,
        spec: DatasetSpec = PANNUKE_SPEC,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        coord_space: str = "qwen_abs",
        input_short_size: int | None = 896,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = resolve_device(device)
        self.dtype = dtype or (torch.float16 if self.device in ("cuda", "mps") else torch.float32)
        self.class_aware = class_aware
        self.spec = spec
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.coord_space = coord_space
        # Upscale tiny patches so nuclei are visible to the VLM (256 px nuclei are
        # ~10-20 px). 896 = 32x28, aligned to Qwen's 28-px tiling. Coords map back
        # via image_grid_thw, so the returned boxes are still in patch pixels.
        self.input_short_size = input_short_size
        self.name = "qwen_box"

        self.category_phrases = (
            list(spec.class_prompts.values()) if class_aware else [spec.generic_prompt]
        )
        self.label_to_class = build_label_to_class(class_aware, spec)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self._patch = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)

    def raw_generate(self, image):
        """Return (decoder_text, inputs) for one PIL image."""
        import torch
        from PIL import Image

        if self.input_short_size:
            w, h = image.size
            if min(w, h) != self.input_short_size:
                scale = self.input_short_size / min(w, h)
                image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": grounding_query(self.category_phrases)},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.device, self.dtype)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0, temperature=self.temperature or None,
            )
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return text, inputs

    def _scales(self, inputs, orig_w: int, orig_h: int) -> tuple[float, float]:
        if self.coord_space == "norm1000":
            return orig_w / 1000.0, orig_h / 1000.0
        grid = inputs.get("image_grid_thw")
        if grid is None:
            return 1.0, 1.0
        _, gh, gw = (int(v) for v in grid[0].tolist())
        resized_h, resized_w = gh * self._patch, gw * self._patch
        return orig_w / resized_w, orig_h / resized_h

    def detect(self, sample: Sample) -> list[Detection]:
        from PIL import Image

        h, w = sample.inst_map.shape
        text, inputs = self.raw_generate(Image.fromarray(sample.image))
        sx, sy = self._scales(inputs, w, h)
        return parse_qwen_boxes(text, sx, sy, width=w, height=h, label_to_class=self.label_to_class)
