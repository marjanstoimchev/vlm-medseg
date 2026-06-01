"""NVIDIA LocateAnything-3B grounding detector (local HF generate, no gradio).

CUDA-only in practice (the ``magi`` attention auto-falls back to ``sdpa``).
:func:`parse_la_output` is a pure function so the token decoding is unit-tested
without a GPU. Coordinates are emitted normalized to ``[0, 1000]``.
"""

from __future__ import annotations

import re

from ..data.base import DatasetSpec
from ..data.pannuke import PANNUKE_SPEC
from ..prompts import locate_anything_query
from ..types import Detection, Sample

_REF_RE = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)
_BOX_RE = re.compile(r"<(?:box|point)>(.*?)</(?:box|point)>", re.DOTALL)
_NUM_RE = re.compile(r"-?\d+")


def detection_query(category_phrases: list[str]) -> str:
    return locate_anything_query(category_phrases, mode="detection")


def pointing_query(category_phrases: list[str]) -> str:
    return locate_anything_query(category_phrases, mode="pointing")


def build_label_to_class(class_aware: bool, spec: DatasetSpec = PANNUKE_SPEC) -> dict[str, int]:
    """Map a ``<ref>`` label (full phrase or bare class name) to a class id."""
    if not class_aware:
        return {}
    m: dict[str, int] = {}
    for cid, phrase in spec.class_prompts.items():
        m[phrase.lower()] = cid
        m[spec.class_name(cid).lower()] = cid
    return m


def _label_to_class(label: str, label_to_class: dict[str, int]) -> int | None:
    if not label_to_class:
        return None
    lab = label.strip().lower()
    if lab in label_to_class:
        return label_to_class[lab]
    for key, cid in label_to_class.items():
        if key in lab or lab in key:
            return cid
    return None


def parse_la_output(
    text: str,
    width: int,
    height: int,
    *,
    mode: str = "detection",
    label_to_class: dict[str, int] | None = None,
    default_label: str = "nucleus",
) -> list[Detection]:
    """Parse decoder text into detections, pairing each box with its prior ref.

    A block with >=4 numbers yields a box; exactly 2 yields a point.
    """
    label_to_class = label_to_class or {}
    refs = [(m.start(), m.group(1).strip()) for m in _REF_RE.finditer(text)]

    def label_before(pos: int) -> str:
        lab = default_label
        for start, name in refs:
            if start >= pos:
                break
            if name:
                lab = name
        return lab

    dets: list[Detection] = []
    for m in _BOX_RE.finditer(text):
        nums = [int(n) for n in _NUM_RE.findall(m.group(1))]
        label = label_before(m.start())
        cid = _label_to_class(label, label_to_class)
        if len(nums) >= 4:
            x1, y1, x2, y2 = nums[:4]
            bx = (
                min(x1, x2) / 1000 * width, min(y1, y2) / 1000 * height,
                max(x1, x2) / 1000 * width, max(y1, y2) / 1000 * height,
            )
            if bx[2] - bx[0] >= 1 and bx[3] - bx[1] >= 1:
                dets.append(Detection(label=label, class_id=cid, box=bx))
        elif len(nums) == 2:
            x, y = nums
            dets.append(
                Detection(label=label, class_id=cid,
                          point=(x / 1000 * width, y / 1000 * height))
            )
    return dets


class LocateAnythingDetector:
    """LocateAnything-3B detector. ``mode`` is ``detection`` (box) or ``pointing``."""

    def __init__(
        self,
        model_id: str = "nvidia/LocateAnything-3B",
        *,
        device: str = "cuda",
        dtype=None,
        mode: str = "detection",
        class_aware: bool = True,
        spec: DatasetSpec = PANNUKE_SPEC,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.9,
        input_short_size: int | None = 1024,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.mode = mode
        self.class_aware = class_aware
        self.spec = spec
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.input_short_size = input_short_size
        self.name = f"la_{'point' if mode == 'pointing' else 'box'}"
        self.category_phrases = (
            list(spec.class_prompts.values()) if class_aware else [spec.generic_prompt]
        )
        self.label_to_class = build_label_to_class(class_aware, spec)

        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(model_id, dtype=self.dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )

    def _query(self) -> str:
        if self.mode == "pointing":
            return pointing_query(self.category_phrases)
        return detection_query(self.category_phrases)

    def _apply_template(self, messages):
        proc = self.processor
        fn = getattr(proc, "py_apply_chat_template", None) or proc.apply_chat_template
        return fn(messages, tokenize=False, add_generation_prompt=True)

    def raw_generate(self, image) -> str:
        """Run the model and return the raw decoder text for one PIL image."""
        import torch
        from PIL import Image

        if self.input_short_size:
            w, h = image.size
            if min(w, h) != self.input_short_size:
                scale = self.input_short_size / min(w, h)
                image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self._query()},
            ]}
        ]
        text = self._apply_template(messages)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                generation_mode=self.generation_mode,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=1.05,
            )
        return response[0] if isinstance(response, (tuple, list)) else response

    def detect(self, sample: Sample) -> list[Detection]:
        from PIL import Image

        h, w = sample.inst_map.shape
        text = self.raw_generate(Image.fromarray(sample.image))
        return parse_la_output(text, w, h, mode=self.mode, label_to_class=self.label_to_class)
