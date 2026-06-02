"""RationAI/PanNuke as a :class:`BaseDataset` (the reference dataset).

PanNuke gives each patch a list of per-nucleus binary masks (``instances``) with
a parallel class list (``categories``); we fold these into one ``(H, W)`` label
map (0 = background) plus an ``{instance_id: class_id}`` dict.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..constants import (
    CLASS_COLORS,
    CLASS_PROMPTS,
    CONCEPT_PROMPTS,
    DATASET_ID,
    GENERIC_CONCEPT,
    GENERIC_PROMPT,
    NUCLEI_CLASSES,
    TISSUE_TYPES,
)
from ..types import Sample
from .base import BaseDataset, DatasetSpec

PANNUKE_SPEC = DatasetSpec(
    name="pannuke",
    class_names=list(NUCLEI_CLASSES),
    class_prompts=dict(CLASS_PROMPTS),
    generic_prompt=GENERIC_PROMPT,
    group_label="tissue",
    group_names=list(TISSUE_TYPES),
    class_colors=dict(CLASS_COLORS),
    concept_prompts=dict(CONCEPT_PROMPTS),
    generic_concept=GENERIC_CONCEPT,
)


def _as_class_id(value) -> int:
    if isinstance(value, str):
        return NUCLEI_CLASSES.index(value)
    return int(value)


def decode_row(row: dict, sample_id: str, *, dataset: str = "pannuke", fold: str = "") -> Sample:
    """Turn one raw PanNuke row into a :class:`Sample`."""
    image = np.asarray(row["image"].convert("RGB"), dtype=np.uint8)
    h, w = image.shape[:2]

    inst_map = np.zeros((h, w), dtype=np.int32)
    inst_classes: dict[int, int] = {}
    for i, (mask_img, cat) in enumerate(zip(row["instances"], row["categories"]), start=1):
        m = np.asarray(mask_img)
        if m.ndim == 3:
            m = m[..., 0]
        fg = m > 0
        if not fg.any():
            continue
        inst_map[fg] = i
        inst_classes[i] = _as_class_id(cat)

    kept = sorted(inst_classes)
    if kept != list(range(1, len(kept) + 1)):
        remap = {old: new for new, old in enumerate(kept, start=1)}
        new_map = np.zeros_like(inst_map)
        new_classes: dict[int, int] = {}
        for old, new in remap.items():
            new_map[inst_map == old] = new
            new_classes[new] = inst_classes[old]
        inst_map, inst_classes = new_map, new_classes

    return Sample(
        image=image,
        inst_map=inst_map,
        inst_classes=inst_classes,
        sample_id=sample_id,
        group=int(row["tissue"]),
        dataset=dataset,
        meta={"fold": fold} if fold else {},
    )


# Back-compat functional alias.
def decode_sample(row: dict, sample_id: str, fold: str = "") -> Sample:
    return decode_row(row, sample_id, fold=fold)


def load_pannuke(fold: str = "fold1", *, streaming: bool = False, cache_dir: str | None = None):
    """Return the raw HF dataset object for one fold (fold1|fold2|fold3)."""
    from datasets import load_dataset

    return load_dataset(DATASET_ID, split=fold, streaming=streaming, cache_dir=cache_dir)


class PanNukeDataset(BaseDataset):
    spec = PANNUKE_SPEC

    def __init__(self, fold: str = "fold1", *, cache_dir: str | None = None):
        self.fold = fold
        self.ds = load_pannuke(fold, cache_dir=cache_dir)
        self._groups: list[int] | None = None

    def __len__(self) -> int:
        return len(self.ds)

    def decode(self, idx: int) -> Sample:
        return decode_row(self.ds[int(idx)], f"{self.fold}-{idx:05d}", fold=self.fold)

    def group_ids(self) -> list[int]:
        if self._groups is None:
            self._groups = [int(t) for t in self.ds["tissue"]]
        return self._groups


def tissue_distribution(dataset) -> Counter:
    return Counter(int(t) for t in dataset["tissue"])


def stratified_indices(dataset, n: int, *, seed: int = 0) -> list[int]:
    """Stratified sampling over a raw HF dataset (back-compat helper)."""
    import random

    tissues = [int(t) for t in dataset["tissue"]]
    total = len(tissues)
    n = min(n, total)
    by_tissue: dict[int, list[int]] = {}
    for idx, t in enumerate(tissues):
        by_tissue.setdefault(t, []).append(idx)
    rng = random.Random(seed)
    raw = {t: len(ix) / total * n for t, ix in by_tissue.items()}
    quota = {t: int(v) for t, v in raw.items()}
    remainder = n - sum(quota.values())
    for t in sorted(by_tissue, key=lambda t: raw[t] - quota[t], reverse=True):
        if remainder <= 0:
            break
        quota[t] += 1
        remainder -= 1
    chosen: list[int] = []
    for t, idxs in by_tissue.items():
        chosen.extend(rng.sample(idxs, min(quota.get(t, 0), len(idxs))))
    return sorted(chosen)
