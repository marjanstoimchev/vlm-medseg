"""Inference: assign a class to each predicted nucleus mask with a trained probe.

Drop-in for the pipeline's ``classifier=`` hook: run the segmenter
class-agnostically, then ``classify_instances`` fills each instance's ``class_id``
from the pixels (frozen encoder + the trained linear head).
"""

from __future__ import annotations

from pathlib import Path

from .crops import instance_crop
from .encoders import build_encoder


class NucleusClassifier:
    def __init__(self, encoder, clf, class_names: list[str], *, margin: float = 0.4, min_size: int = 48):
        self.encoder = encoder
        self.clf = clf
        self.class_names = class_names
        self.margin = margin
        self.min_size = min_size

    def classify_instances(self, image, instances):
        """Set ``class_id`` on each instance in place (and return the list)."""
        if not instances:
            return instances
        crops = [instance_crop(image, ins.mask, margin=self.margin, min_size=self.min_size)
                 for ins in instances]
        preds = self.clf.predict(self.encoder.embed(crops))
        for ins, p in zip(instances, preds):
            ins.class_id = int(p)
        return instances

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> NucleusClassifier:
        import joblib

        d = joblib.load(path)
        encoder = build_encoder(d["encoder"], device=device)
        return cls(encoder, d["clf"], d["class_names"], margin=d["margin"], min_size=d["min_size"])


def save_probe(path, clf, class_names, *, encoder: str, margin: float, min_size: int, metrics=None):
    """Persist a trained probe (head + crop params + encoder name) for inference."""
    import joblib

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "clf": clf, "class_names": class_names, "encoder": encoder,
        "margin": margin, "min_size": min_size, "metrics": metrics,
    }, path)
