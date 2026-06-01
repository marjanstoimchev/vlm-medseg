"""Unit tests for Qwen2.5-VL grounding output parsing (no GPU / model needed)."""

from vlm_medseg.detect.locate_anything import build_label_to_class
from vlm_medseg.detect.qwen_vl import grounding_query, parse_qwen_boxes


def test_parse_json_scales_and_maps_class():
    l2c = build_label_to_class(class_aware=True)
    text = ('[{"bbox_2d": [10, 20, 30, 40], "label": "neoplastic tumor cell nucleus"}, '
            '{"bbox_2d": [0, 0, 100, 100], "label": "inflammatory immune cell nucleus"}]')
    # resized 1000x1000 -> original 256x256: scale = 0.256
    dets = parse_qwen_boxes(text, 0.256, 0.256, label_to_class=l2c)
    assert len(dets) == 2
    x1, y1, x2, y2 = dets[0].box
    assert abs(x1 - 2.56) < 1e-6 and abs(y2 - 10.24) < 1e-6
    assert dets[0].class_id == 0 and dets[1].class_id == 1


def test_parse_tolerant_fallback_on_extra_text():
    text = ('Sure! Here are the boxes:\n'
            '[{"bbox_2d": [5, 5, 15, 15], "label": "cell"}] hope this helps')
    dets = parse_qwen_boxes(text, 1.0, 1.0)
    assert len(dets) == 1
    assert dets[0].box == (5.0, 5.0, 15.0, 15.0)
    assert dets[0].class_id is None  # empty label map


def test_degenerate_box_dropped():
    dets = parse_qwen_boxes('[{"bbox_2d": [5, 5, 5, 5], "label": "x"}]', 1.0, 1.0)
    assert dets == []


def test_grounding_query_lists_categories():
    q = grounding_query(["cell nucleus"])
    assert "cell nucleus" in q and "bbox_2d" in q
