"""Detection caching.

LocateAnything is the expensive stage (seconds/patch on a GPU); SAM2 and the
metrics are cheap. Caching the *detections* (tiny JSONL) lets the costly VLM
pass run once on Kaggle, then SAM2 + evaluation re-run anywhere — including
locally on MPS — without touching the GPU again.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..types import Detection


def _det_to_dict(d: Detection) -> dict:
    return {
        "label": d.label,
        "score": d.score,
        "class_id": d.class_id,
        "box": list(d.box) if d.box is not None else None,
        "point": list(d.point) if d.point is not None else None,
    }


def _dict_to_det(r: dict) -> Detection:
    return Detection(
        label=r["label"],
        score=r.get("score", 1.0),
        class_id=r.get("class_id"),
        box=tuple(r["box"]) if r.get("box") else None,
        point=tuple(r["point"]) if r.get("point") else None,
    )


def save_detections(path: str | Path, records: list[dict]) -> None:
    """Write one JSON object per patch.

    Each record: ``{sample_id, condition, detector_seconds, detections, raw?}``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            out = dict(rec)
            out["detections"] = [_det_to_dict(d) for d in rec["detections"]]
            fh.write(json.dumps(out) + "\n")


def load_detections(path: str | Path) -> dict[str, list[Detection]]:
    """Map ``sample_id -> [Detection]`` from a cache file."""
    out: dict[str, list[Detection]] = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["sample_id"]] = [_dict_to_det(d) for d in rec["detections"]]
    return out
