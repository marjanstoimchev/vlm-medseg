"""Dataset abstraction: subclass :class:`BaseDataset` to plug in any instance-
segmentation dataset. The pipeline, metrics, detectors and plots are all driven
by the :class:`DatasetSpec` (class names, VLM prompts, stratification groups,
colors), so nothing downstream is PanNuke-specific.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..types import Sample


@dataclass
class DatasetSpec:
    """Everything the dataset-agnostic layers need to know about a dataset."""

    name: str
    class_names: list[str]
    class_prompts: dict[int, str]              # class id -> grounding-VLM phrase
    generic_prompt: str = "object"             # class-agnostic phrase
    group_label: str = "group"                 # e.g. "tissue"
    group_names: list[str] = field(default_factory=list)
    class_colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # SAM3 prefers short noun-phrase concepts; falls back to class_prompts if unset.
    concept_prompts: dict[int, str] = field(default_factory=dict)
    generic_concept: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_name(self, cid: int) -> str:
        return self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)

    def group_name(self, gid: int) -> str:
        return self.group_names[gid] if 0 <= gid < len(self.group_names) else str(gid)

    def concept_prompt(self, cid: int) -> str:
        """Short SAM3 concept for a class (falls back to the VLM phrase / name)."""
        return self.concept_prompts.get(cid) or self.class_prompts.get(cid) or self.class_name(cid)

    @property
    def generic_concept_prompt(self) -> str:
        return self.generic_concept or self.generic_prompt


class BaseDataset(ABC):
    """Indexed access to decoded :class:`Sample` objects plus stratified sampling.

    Subclasses implement :meth:`__len__`, :meth:`decode`, and (optionally)
    :meth:`group_ids` for stratification; everything else is generic.
    """

    spec: DatasetSpec

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def decode(self, idx: int) -> Sample: ...

    def group_ids(self) -> list[int]:
        """Group id per index (default: ungrouped)."""
        return [-1] * len(self)

    def stratified_indices(self, n: int, *, seed: int = 0) -> list[int]:
        """``n`` indices stratified across groups (random if ungrouped).

        Per-group quotas use largest-remainder allocation so they sum to ``n``;
        indices are returned sorted for cache-friendly iteration.
        """
        total = len(self)
        n = min(n, total)
        groups = self.group_ids()
        rng = random.Random(seed)

        if not groups or len(set(groups)) <= 1:
            return sorted(rng.sample(range(total), n))

        by_group: dict[int, list[int]] = {}
        for idx, g in enumerate(groups):
            by_group.setdefault(g, []).append(idx)

        raw = {g: len(ix) / total * n for g, ix in by_group.items()}
        quota = {g: int(v) for g, v in raw.items()}
        remainder = n - sum(quota.values())
        for g in sorted(by_group, key=lambda g: raw[g] - quota[g], reverse=True):
            if remainder <= 0:
                break
            quota[g] += 1
            remainder -= 1

        chosen: list[int] = []
        for g, idxs in by_group.items():
            chosen.extend(rng.sample(idxs, min(quota.get(g, 0), len(idxs))))
        return sorted(chosen)

    def iter_samples(self, indices: list[int] | None = None) -> Iterator[Sample]:
        for i in indices if indices is not None else range(len(self)):
            yield self.decode(int(i))

    def sample(self, n: int, *, seed: int = 0) -> list[Sample]:
        """Decode a stratified sample of ``n`` patches."""
        return [self.decode(i) for i in self.stratified_indices(n, seed=seed)]
