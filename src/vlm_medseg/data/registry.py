"""Name -> dataset factory registry.

    from vlm_medseg.data import get_dataset
    ds = get_dataset("pannuke", fold="fold1")

Add a dataset by subclassing :class:`~vlm_medseg.data.base.BaseDataset` and
registering its factory:

    register_dataset("mydata", MyDataset)
"""

from __future__ import annotations

from collections.abc import Callable

from .base import BaseDataset

_REGISTRY: dict[str, Callable[..., BaseDataset]] = {}


def register_dataset(name: str, factory: Callable[..., BaseDataset]) -> None:
    _REGISTRY[name] = factory


def get_dataset(name: str, **kwargs) -> BaseDataset:
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_datasets() -> list[str]:
    return sorted(_REGISTRY)


def _register_builtins() -> None:
    from .pannuke import PanNukeDataset

    register_dataset("pannuke", PanNukeDataset)


_register_builtins()
