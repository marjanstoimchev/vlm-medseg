"""Extract a context crop around a nucleus for the mask classifier.

A tight nucleus is only ~10-40 px; subtype is easier to read with surrounding
tissue context, so we crop a square window around the instance bbox (expanded by
``margin``, floored at ``min_size``) and let the encoder resize it.
"""

from __future__ import annotations

import numpy as np


def _crop_window(mask: np.ndarray, margin: float, min_size: int, shape) -> tuple[int, int, int] | None:
    """Square window ``(y0, x0, side)`` centered on the mask, clipped to the image."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    side = max(x2 - x1, y2 - y1)
    side = max(side + 2 * int(round(side * margin)), min_size)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = shape[:2]
    side = min(side, h, w)
    x0 = min(max(0, cx - side // 2), w - side)
    y0 = min(max(0, cy - side // 2), h - side)
    return y0, x0, side


def instance_crop(image: np.ndarray, mask: np.ndarray, *, margin: float = 0.4, min_size: int = 48) -> np.ndarray:
    """Square RGB context crop centered on the mask (clipped to the image)."""
    win = _crop_window(mask, margin, min_size, image.shape)
    if win is None:
        return image
    y0, x0, s = win
    return image[y0:y0 + s, x0:x0 + s]


def instance_crop_and_mask(
    image: np.ndarray, mask: np.ndarray, *, margin: float = 0.4, min_size: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    """Same window as :func:`instance_crop`, returning ``(rgb_crop, mask_crop)``.

    The aligned mask crop lets callers draw the nucleus boundary on the context
    crop (the encoder still sees the raw RGB crop, without any overlay).
    """
    win = _crop_window(mask, margin, min_size, image.shape)
    if win is None:
        return image, np.zeros(image.shape[:2], dtype=bool)
    y0, x0, s = win
    return image[y0:y0 + s, x0:x0 + s], mask[y0:y0 + s, x0:x0 + s]
