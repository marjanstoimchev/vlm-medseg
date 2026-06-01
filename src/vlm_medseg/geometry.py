"""Mask geometry helpers (numpy/scipy only)."""

from __future__ import annotations

import numpy as np

from .types import Box, Point


def mask_to_box(mask: np.ndarray) -> Box | None:
    """Tight (x1, y1, x2, y2) box of a boolean mask, or None if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def mask_to_interior_point(mask: np.ndarray) -> Point | None:
    """A point guaranteed to lie inside the mask (distance-transform peak).

    Preferred over the raw centroid, which can fall outside a concave nucleus.
    """
    if not mask.any():
        return None
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(mask)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return (float(x), float(y))


def instance_boxes(inst_map: np.ndarray) -> dict[int, Box]:
    """Bounding box per instance id (1..N) in a label map."""
    out: dict[int, Box] = {}
    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        box = mask_to_box(inst_map == inst_id)
        if box is not None:
            out[int(inst_id)] = box
    return out


def instance_points(inst_map: np.ndarray) -> dict[int, Point]:
    """Interior point per instance id (1..N) in a label map."""
    out: dict[int, Point] = {}
    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        pt = mask_to_interior_point(inst_map == inst_id)
        if pt is not None:
            out[int(inst_id)] = pt
    return out
