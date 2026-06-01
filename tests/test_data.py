"""Decode + stratified-sampler tests using synthetic rows (no network)."""

import numpy as np
from PIL import Image

from vlm_medseg.data.pannuke import decode_sample, stratified_indices
from vlm_medseg.geometry import mask_to_box, mask_to_interior_point


def _mask(shape, region):
    m = np.zeros(shape, np.uint8)
    y0, y1, x0, x1 = region
    m[y0:y1, x0:x1] = 255
    return Image.fromarray(m).convert("1")


def test_decode_sample_builds_inst_map_and_classes():
    img = Image.fromarray(np.zeros((16, 16, 3), np.uint8))
    instances = [_mask((16, 16), (0, 4, 0, 4)), _mask((16, 16), (8, 12, 8, 12))]
    row = {"image": img, "instances": instances, "categories": [0, 2], "tissue": 3}
    s = decode_sample(row, "x-0", "fold1")
    assert s.num_instances == 2
    assert s.inst_classes == {1: 0, 2: 2}
    assert int(s.inst_map.max()) == 2
    assert s.tissue == 3
    cm = s.class_map()
    assert cm[1, 1] == 0 and cm[9, 9] == 2 and cm[15, 15] == -1


def test_decode_drops_empty_masks_and_repacks_ids():
    img = Image.fromarray(np.zeros((16, 16, 3), np.uint8))
    empty = Image.fromarray(np.zeros((16, 16), np.uint8)).convert("1")
    instances = [_mask((16, 16), (0, 4, 0, 4)), empty, _mask((16, 16), (8, 12, 8, 12))]
    row = {"image": img, "instances": instances, "categories": [0, 1, 2], "tissue": 5}
    s = decode_sample(row, "x-1")
    # the empty middle mask is dropped; ids stay contiguous 1..2
    assert sorted(s.inst_classes) == [1, 2]
    assert set(s.inst_classes.values()) == {0, 2}


def test_geometry_box_and_interior_point():
    m = np.zeros((20, 20), bool)
    m[5:15, 4:10] = True
    assert mask_to_box(m) == (4.0, 5.0, 10.0, 15.0)
    x, y = mask_to_interior_point(m)
    assert m[int(y), int(x)]  # point lies inside


class _StubDS:
    def __init__(self, tissues):
        self._t = tissues

    def __len__(self):
        return len(self._t)

    def __getitem__(self, key):
        assert key == "tissue"
        return self._t


def test_stratified_indices_quota_and_coverage():
    # 100 patches: tissue 0 x70, tissue 1 x20, tissue 2 x10.
    tissues = [0] * 70 + [1] * 20 + [2] * 10
    ds = _StubDS(tissues)
    idx = stratified_indices(ds, 20, seed=0)
    assert len(idx) == 20
    assert idx == sorted(idx)
    chosen = [tissues[i] for i in idx]
    # proportional: ~14 / ~4 / ~2, and every tissue represented
    assert set(chosen) == {0, 1, 2}
    assert chosen.count(0) == 14
