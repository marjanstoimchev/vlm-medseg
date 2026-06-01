"""Unit tests for LocateAnything output parsing (no GPU / model needed).

The generation runs only on Kaggle, but the brittle token-decoding is a pure
function and is fully tested here.
"""

from vlm_medseg.detect.locate_anything import (
    build_label_to_class,
    detection_query,
    parse_la_output,
    pointing_query,
)


def test_detection_parse_scales_and_maps_class():
    l2c = build_label_to_class(class_aware=True)
    text = ("<ref>neoplastic tumor cell nucleus</ref><box><100><200><300><400></box>"
            "<ref>inflammatory immune cell nucleus</ref><box><0><0><1000><1000></box>")
    dets = parse_la_output(text, 256, 256, mode="detection", label_to_class=l2c)
    assert len(dets) == 2
    # 0-1000 normalized -> pixels on a 256 patch.
    x1, y1, x2, y2 = dets[0].box
    assert abs(x1 - 25.6) < 1e-6 and abs(y2 - 102.4) < 1e-6
    assert dets[0].class_id == 0          # Neoplastic
    assert dets[1].class_id == 1          # Inflammatory
    assert dets[1].box == (0.0, 0.0, 256.0, 256.0)


def test_space_separated_numbers():
    dets = parse_la_output("<box>100 200 300 400</box>", 1000, 1000, mode="detection")
    assert len(dets) == 1
    assert dets[0].box == (100.0, 200.0, 300.0, 400.0)


def test_pointing_parse():
    dets = parse_la_output("<ref>cell nucleus</ref><point><500><500></point>",
                           256, 256, mode="pointing")
    assert len(dets) == 1
    assert dets[0].point == (128.0, 128.0)
    assert dets[0].box is None


def test_inverted_box_is_normalized():
    # x2<x1, y2<y1 should be reordered, not dropped.
    dets = parse_la_output("<box><300><400><100><200></box>", 1000, 1000, mode="detection")
    assert dets[0].box == (100.0, 200.0, 300.0, 400.0)


def test_degenerate_box_dropped():
    dets = parse_la_output("<box><100><100><100><100></box>", 1000, 1000, mode="detection")
    assert dets == []


def test_generic_has_no_class():
    dets = parse_la_output("<box><10><10><90><90></box>", 100, 100,
                           mode="detection", label_to_class={})
    assert dets[0].class_id is None


def test_query_strings():
    assert "Locate all the instances" in detection_query(["cell nucleus"])
    assert pointing_query(["cell nucleus"]).startswith("Point to:")
