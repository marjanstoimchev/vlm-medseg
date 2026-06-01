"""Segmentation back-ends.

- :class:`Sam2Masker` — box/point-prompted masker (fed by a detector).
- :class:`Sam3TextSegmenter` — text-prompted concept segmentation, end to end.
"""

from .sam2 import DEFAULT_MODEL_ID, Sam2Masker, resolve_device

__all__ = ["Sam2Masker", "resolve_device", "DEFAULT_MODEL_ID", "Sam3TextSegmenter"]


def __getattr__(name: str):  # lazy: avoid importing Sam3 stack unless used
    if name == "Sam3TextSegmenter":
        from .sam3 import Sam3TextSegmenter

        return Sam3TextSegmenter
    raise AttributeError(name)
