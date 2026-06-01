"""Detectors: oracle (GT prompts), LocateAnything-3B, Grounding-DINO.

Only :mod:`oracle` is import-light. The model detectors import torch /
transformers lazily inside their constructors, so importing this package stays
cheap until a model is actually instantiated.
"""

from .base import Detector
from .oracle import OracleBoxDetector, OraclePointDetector

__all__ = ["Detector", "OracleBoxDetector", "OraclePointDetector"]


def __getattr__(name: str):  # lazy heavy detectors
    if name == "LocateAnythingDetector":
        from .locate_anything import LocateAnythingDetector

        return LocateAnythingDetector
    if name == "GroundingDinoDetector":
        from .grounding_dino import GroundingDinoDetector

        return GroundingDinoDetector
    if name == "Owlv2Detector":
        from .owlv2 import Owlv2Detector

        return Owlv2Detector
    if name == "QwenVLDetector":
        from .qwen_vl import QwenVLDetector

        return QwenVLDetector
    raise AttributeError(name)
