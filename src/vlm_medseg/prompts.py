"""Per-model prompt templates and prompting guidance, in one place.

Separation of concerns: a :class:`~vlm_medseg.data.base.DatasetSpec` supplies the
*vocabulary* (``class_prompts`` for grounding VLMs, ``concept_prompts`` for SAM3);
this module supplies the per-model *request template* and records the best
prompt style for each method, so the prompt engineering lives in one auditable
spot rather than scattered across detectors.

``PROMPTS`` summarizes, per method: which spec field feeds it, the request
template, the output coordinate convention, the recommended threshold, and notes.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Request templates ─────────────────────────────────────────────────────
def locate_anything_query(phrases: list[str], *, mode: str = "detection") -> str:
    """LocateAnything-3B: autoregressive ``<ref>``/``<box>`` grounding.

    Best with the long descriptive class phrases joined by ", "; coordinates
    come back normalized to [0, 1000].
    """
    cats = ", ".join(phrases)
    if mode == "pointing":
        return f"Point to: {cats}."
    return f"Locate all the instances that matches the following description: {cats}."


def qwen_grounding_query(phrases: list[str]) -> str:
    """Qwen2.5-VL: JSON ``bbox_2d`` grounding (greedy decode).

    Best with descriptive phrases; coordinates are in the resized-input pixel
    space (map back via ``image_grid_thw``).
    """
    cats = ", ".join(phrases)
    return (
        f"Detect every instance of the following in the image: {cats}. "
        'Output a JSON list where each element is '
        '{"bbox_2d": [x1, y1, x2, y2], "label": "<category>"}. '
        "Output only the JSON."
    )


def grounding_dino_text(phrases: list[str]) -> str:
    """Grounding-DINO: lowercase phrases joined by ' . ', terminated with '.'."""
    return " . ".join(p.lower() for p in phrases) + " ."


# SAM3 has no request template — it takes one short noun-phrase concept per
# forward pass, supplied directly from ``DatasetSpec.concept_prompt``.


# ── Best-practice summary, one entry per method ───────────────────────────
@dataclass(frozen=True)
class PromptStyle:
    model: str
    phrase_source: str       # which DatasetSpec field supplies the vocabulary
    request: str             # how the request is phrased
    coord_space: str         # output coordinate convention
    threshold: str           # recommended detection threshold
    notes: str


PROMPTS: dict[str, PromptStyle] = {
    "locate_anything": PromptStyle(
        model="LocateAnything-3B",
        phrase_source="class_prompts (long descriptive) / generic_prompt",
        request="locate_anything_query(...)  ·  'Locate all the instances ...'",
        coord_space="0-1000 normalized",
        threshold="n/a (decoded set)",
        notes="Fine-tuned grounding head on Qwen2.5-3B; ', '-joined category list.",
    ),
    "qwen_vl": PromptStyle(
        model="Qwen2.5-VL-3B-Instruct",
        phrase_source="class_prompts (long descriptive) / generic_prompt",
        request="qwen_grounding_query(...)  ·  JSON bbox_2d list",
        coord_space="resized-abs via image_grid_thw (or norm1000 for Qwen2-VL)",
        threshold="n/a (decoded set)",
        notes="Stock backbone; the comparison point for LocateAnything.",
    ),
    "grounding_dino": PromptStyle(
        model="Grounding-DINO",
        phrase_source="class_prompts (lowercased) / generic_prompt",
        request="grounding_dino_text(...)  ·  \"a . b . c .\"",
        coord_space="absolute pixels",
        threshold="box 0.25 / text 0.20",
        notes="Non-VLM open-vocab detector baseline.",
    ),
    "sam3": PromptStyle(
        model="SAM3",
        phrase_source="concept_prompts (short noun phrases) / generic_concept",
        request="one concept per forward, passed directly (no template)",
        coord_space="native instance masks (end to end)",
        threshold="0.5 (official; lower to probe recall on out-of-domain data)",
        notes="Promptable concept segmentation; short noun phrases beat long ones.",
    ),
}
