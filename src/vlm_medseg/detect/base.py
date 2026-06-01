"""Detector interface.

A detector consumes a :class:`Sample` and emits :class:`Detection` objects
carrying a box and/or a point. Real detectors read only ``sample.image``; oracle
detectors read the GT maps to establish the SAM2 ceiling.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Detection, Sample


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, sample: Sample) -> list[Detection]:
        """Return localizations for one patch."""
        ...
