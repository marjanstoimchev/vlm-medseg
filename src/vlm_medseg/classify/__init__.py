"""Post-hoc nucleus classification: frozen encoder + trained linear probe.

Decouples segmentation from recognition — segment class-agnostically, then
classify each mask from the pixels. See ``scripts/curate_classifier_data.py`` and
``scripts/train_classifier.py``.
"""

from .crops import instance_crop
from .dataset import CropDataset, assign_splits, cap_per_class, curate, extract_features
from .encoders import Dinov3Encoder, Uni2hEncoder, available_encoders, build_encoder
from .heads import DEFAULT_HEAD, HEAD_NAMES, build_head
from .linear_probe import NucleusClassifier, save_probe
from .train import train_linear_probe, train_probe

__all__ = [
    "instance_crop",
    "build_encoder", "available_encoders", "Uni2hEncoder", "Dinov3Encoder",
    "build_head", "DEFAULT_HEAD", "HEAD_NAMES",
    "CropDataset", "curate", "extract_features", "assign_splits", "cap_per_class",
    "train_probe", "train_linear_probe",
    "NucleusClassifier", "save_probe",
]
