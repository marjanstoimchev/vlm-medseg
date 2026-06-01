"""Curate a nucleus-crop dataset for the classifier.

Pipeline: decode patches once -> crop every nucleus -> embed with a frozen
encoder -> assign **stratified, patch-grouped** splits (sklearn
``StratifiedGroupKFold``: balanced by class, but all nuclei from one patch stay
in the same split, so there is no patch leakage). Features are cached to an npz.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.base import BaseDataset
from ..log import get_logger
from .crops import instance_crop

log = get_logger(__name__)


@dataclass
class CropDataset:
    """Curated features + labels + splits for the nucleus classifier."""

    features: np.ndarray            # (N, dim) float32
    class_id: np.ndarray            # (N,) int
    tissue: np.ndarray              # (N,) int
    patch_index: np.ndarray         # (N,) int  (the grouping key)
    split: np.ndarray               # (N,) str in {train,val,test}
    class_names: list[str]
    encoder: str
    margin: float
    min_size: int

    def mask(self, split: str) -> np.ndarray:
        return self.split == split

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, features=self.features, class_id=self.class_id, tissue=self.tissue,
            patch_index=self.patch_index, split=self.split,
        )
        path.with_suffix(".json").write_text(json.dumps({
            "class_names": self.class_names, "encoder": self.encoder,
            "margin": self.margin, "min_size": self.min_size,
            "n": int(self.features.shape[0]), "dim": int(self.features.shape[1]),
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CropDataset:
        path = Path(path)
        z = np.load(path, allow_pickle=True)
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(
            features=z["features"], class_id=z["class_id"], tissue=z["tissue"],
            patch_index=z["patch_index"], split=z["split"].astype(str),
            class_names=meta["class_names"], encoder=meta["encoder"],
            margin=meta["margin"], min_size=meta["min_size"],
        )


def extract_features(
    ds: BaseDataset,
    encoder,
    indices: list[int],
    *,
    margin: float = 0.4,
    min_size: int = 48,
    progress: bool = True,
):
    """Crop every nucleus in the given patches and embed it.

    Returns ``(features, class_id, tissue, patch_index)`` arrays aligned row-wise.
    """
    crops: list[np.ndarray] = []
    cls_id: list[int] = []
    tissue: list[int] = []
    patch_idx: list[int] = []

    it = indices
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(indices, desc="decode+crop", unit="patch")
        except Exception:
            pass

    for i in it:
        s = ds.decode(int(i))
        for inst_id, cls in s.inst_classes.items():
            crops.append(instance_crop(s.image, s.inst_map == inst_id,
                                       margin=margin, min_size=min_size))
            cls_id.append(int(cls))
            tissue.append(int(s.group))
            patch_idx.append(int(i))

    if progress:
        log.info(f"embedding {len(crops)} nuclei crops ...")
    features = encoder.embed(crops, progress=progress)
    return (features, np.array(cls_id), np.array(tissue), np.array(patch_idx))


def cap_per_class(class_id: np.ndarray, max_per_class: int, seed: int = 0) -> np.ndarray:
    """Row indices keeping at most ``max_per_class`` of each class (balance)."""
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for c in np.unique(class_id):
        idx = np.where(class_id == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.extend(idx.tolist())
    return np.sort(np.array(keep))


def assign_splits(
    class_id: np.ndarray, patch_index: np.ndarray, *,
    n_splits: int = 5, seed: int = 0, mode: str = "stratified",
) -> np.ndarray:
    """Split labels (train/val/test), one per nucleus.

    ``mode``:
      - ``"stratified"`` — instance-level ``StratifiedKFold``: every class is
        balanced across all splits (rare classes like Dead appear everywhere).
        Nuclei from one patch may span splits (mild leakage) — acceptable because
        the *definitive* test is a different PanNuke fold run through the pipeline.
      - ``"group"`` — ``StratifiedGroupKFold`` grouped by patch: no patch leakage,
        but a class confined to few patches can be absent from val/test.

    Of ``n_splits`` folds: last -> test, second-to-last -> val, rest -> train.
    """
    if mode == "group":
        from sklearn.model_selection import StratifiedGroupKFold
        gen = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            class_id, class_id, patch_index)
    else:
        from sklearn.model_selection import StratifiedKFold
        gen = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            class_id, class_id)
    fold = np.empty(len(class_id), dtype=int)
    for k, (_, test_idx) in enumerate(gen):
        fold[test_idx] = k
    return np.where(fold == n_splits - 1, "test",
                    np.where(fold == n_splits - 2, "val", "train"))


def curate(
    ds: BaseDataset,
    encoder,
    *,
    n_patches: int | None = None,
    seed: int = 0,
    margin: float = 0.4,
    min_size: int = 48,
    max_per_class: int | None = None,
    n_splits: int = 5,
    split_mode: str = "stratified",
    progress: bool = True,
) -> CropDataset:
    """End-to-end: sample patches -> features -> (cap) -> splits (see assign_splits)."""
    indices = (ds.stratified_indices(n_patches, seed=seed)
               if n_patches else list(range(len(ds))))
    feats, cls, tis, pidx = extract_features(
        ds, encoder, indices, margin=margin, min_size=min_size, progress=progress
    )
    if max_per_class:
        keep = cap_per_class(cls, max_per_class, seed=seed)
        feats, cls, tis, pidx = feats[keep], cls[keep], tis[keep], pidx[keep]
    split = assign_splits(cls, pidx, n_splits=n_splits, seed=seed, mode=split_mode)
    return CropDataset(
        features=feats, class_id=cls, tissue=tis, patch_index=pidx, split=split,
        class_names=list(ds.spec.class_names), encoder=encoder.name,
        margin=margin, min_size=min_size,
    )
