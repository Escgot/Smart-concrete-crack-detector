"""
CrackScan Phase 2 — Test Suite
================================
Covers: preprocessing, softmax, severity heuristics, bounding box
decoding, overlay rendering, MockDetector (with boxes), and all
database operations including the new Phase 2 columns.

Run: pytest tests/ -v   (no model weights required)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_img():
    """Plain grey 256×256 BGR image."""
    return np.full((256, 256, 3), 180, dtype=np.uint8)


@pytest.fixture
def dark_img():
    """Dark image — MockDetector treats as likely cracked."""
    return np.full((256, 256, 3), 30, dtype=np.uint8)


@pytest.fixture
def bright_img():
    """Bright image — MockDetector treats as likely uncracked."""
    return np.full((256, 256, 3), 230, dtype=np.uint8)


@pytest.fixture
def synthetic_crack_img():
    """Image with a dark diagonal line simulating a crack."""
    img = np.full((256, 256, 3), 200, dtype=np.uint8)
    cv2.line(img, (30, 30), (220, 220), (20, 20, 20), thickness=2)
    return img


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Redirect DB to a temp file for each test."""
    db_path = tmp_path / "test_inspections.db"
    import api.database as database
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    return database


# ---------------------------------------------------------------------------
# Phase 1 preprocessing tests (unchanged)
# ---------------------------------------------------------------------------

class TestPreprocessingCls:
    def test_output_shape(self, blank_img):
        from api.detector import preprocess_cls
        tensor = preprocess_cls(blank_img)
        assert tensor.shape == (1, 3, 224, 224)

    def test_output_dtype(self, blank_img):
        from api.detector import preprocess_cls
        assert preprocess_cls(blank_img).dtype == np.float32

    def test_values_normalised(self, blank_img):
        from api.detector import preprocess_cls
        tensor = preprocess_cls(blank_img)
        assert tensor.min() > -4.0
        assert tensor.max() < 4.0

    def test_accepts_various_sizes(self):
        from api.detector import preprocess_cls
        for shape in [(100, 100, 3), (1024, 768, 3), (64, 256, 3)]:
            tensor = preprocess_cls(np.zeros(shape, dtype=np.uint8))
            assert tensor.shape == (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Phase 2 preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessingDet:
    def test_output_shape(self, blank_img):
        from api.detector import preprocess_det
        tensor, scale, px, py = preprocess_det(blank_img)
        assert tensor.shape == (1, 3, 640, 640)

    def test_output_dtype(self, blank_img):
        from api.detector import preprocess_det
        tensor, *_ = preprocess_det(blank_img)
        assert tensor.dtype == np.float32

    def test_values_in_unit_range(self, blank_img):
        from api.detector import preprocess_det
        tensor, *_ = preprocess_det(blank_img)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_scale_and_pads_returned(self, blank_img):
        from api.detector import preprocess_det
        _, scale, px, py = preprocess_det(blank_img)
        assert isinstance(scale, float) and scale > 0
        assert isinstance(px, int) and isinstance(py, int)

    def test_accepts_non_square_input(self):
        from api.detector import preprocess_det
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, scale, px, py = preprocess_det(img)
        assert tensor.shape == (1, 3, 640, 640)


# ---------------------------------------------------------------------------
# Softmax tests
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        from api.detector import softmax
        probs = softmax(np.array([1.5, -0.3, 2.1, 0.0]))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_max_corresponds_to_argmax(self):
        from api.detector import softmax
        probs = softmax(np.array([0.1, 5.0, -1.0]))
        assert np.argmax(probs) == 1

    def test_numerical_stability(self):
        from api.detector import softmax
        probs = softmax(np.array([1000.0, 999.0, 998.0]))
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))


# ---------------------------------------------------------------------------
# Phase 2: BoundingBox dataclass
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def _make_box(self, x1=10, y1=20, x2=60, y2=80, conf=0.85, sev="moderate"):
        from api.detector import BoundingBox
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, severity=sev)

    def test_width_height(self):
        b = self._make_box()
        assert b.box_width == 50
        assert b.box_height == 60

    def test_to_dict_keys(self):
        d = self._make_box().to_dict()
        for key in ("x1","y1","x2","y2","width","height","confidence","severity"):
            assert key in d

    def test_to_dict_values(self):
        d = self._make_box(x1=5, y1=10, x2=55, y2=110, conf=0.9, sev="severe").to_dict()
        assert d["width"] == 50
        assert d["height"] == 100
        assert d["confidence"] == pytest.approx(0.9)
        assert d["severity"] == "severe"


# ---------------------------------------------------------------------------
# Phase 2: severity heuristics for boxes
# ---------------------------------------------------------------------------

class TestBoxSeverity:
    def test_hairline_small_area(self):
        from api.detector import _box_severity
        assert _box_severity(0.001) == "hairline"

    def test_moderate_medium_area(self):
        from api.detector import _box_severity
        assert _box_severity(0.02) == "moderate"

    def test_severe_large_area(self):
        from api.detector import _box_severity
        assert _box_severity(0.10) == "severe"

    def test_boundary_hairline_moderate(self):
        from api.detector import _box_severity
        assert _box_severity(0.005) == "moderate"  # boundary → moderate


class TestAggregateSeverity:
    def test_empty_boxes_returns_none(self):
        from api.detector import _aggregate_severity
        assert _aggregate_severity([]) == "none"

    def test_returns_worst_severity(self):
        from api.detector import _aggregate_severity, BoundingBox
        boxes = [
            BoundingBox(0,0,10,10,0.9,"hairline"),
            BoundingBox(0,0,50,50,0.8,"severe"),
            BoundingBox(0,0,20,20,0.7,"moderate"),
        ]
        assert _aggregate_severity(boxes) == "severe"

    def test_single_box(self):
        from api.detector import _aggregate_severity, BoundingBox
        boxes = [BoundingBox(0,0,10,10,0.8,"moderate")]
        assert _aggregate_severity(boxes) == "moderate"


# ---------------------------------------------------------------------------
# Phase 3: Width measurement & severity
# ---------------------------------------------------------------------------

class TestMeasureCrackWidth:
    def test_empty_crop_returns_zero(self, blank_img):
        from api.detector import measure_crack_width_px, BoundingBox
        # Box outside image or zero area
        b = BoundingBox(x1=0, y1=0, x2=0, y2=0, confidence=0.9, severity="none")
        assert measure_crack_width_px(blank_img, b) == 0.0

    def test_no_crack_pixels_returns_zero(self, blank_img):
        from api.detector import measure_crack_width_px, BoundingBox
        b = BoundingBox(x1=10, y1=10, x2=50, y2=50, confidence=0.9, severity="none")
        assert measure_crack_width_px(blank_img, b) == 0.0

    def test_synthetic_crack_width(self):
        from api.detector import measure_crack_width_px, BoundingBox
        import cv2
        # White background (concrete)
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Draw a 6px thick black line
        cv2.line(img, (20, 50), (80, 50), (0, 0, 0), thickness=6)

        b = BoundingBox(x1=10, y1=10, x2=90, y2=90, confidence=0.9, severity="none")
        w_px = measure_crack_width_px(img, b)

        # Distance transform medial axis value for 6px line is around 3-4, *2 = 6-8
        assert 5.0 <= w_px <= 8.5

class TestSeverityByWidth:
    def test_hairline(self):
        from api.detector import classify_severity_by_width
        assert classify_severity_by_width(0.1) == "hairline"
        assert classify_severity_by_width(0.19) == "hairline"

    def test_moderate(self):
        from api.detector import classify_severity_by_width
        assert classify_severity_by_width(0.2) == "moderate"
        assert classify_severity_by_width(0.49) == "moderate"

    def test_severe(self):
        from api.detector import classify_severity_by_width
        assert classify_severity_by_width(0.5) == "severe"
        assert classify_severity_by_width(1.5) == "severe"

class TestBoundingBoxWidth:
    def test_to_dict_includes_width(self):
        from api.detector import BoundingBox
        b = BoundingBox(x1=10, y1=10, x2=50, y2=50, confidence=0.9, severity="moderate", width_px=10.0, width_mm=0.25)
        d = b.to_dict()
        assert d["crack_width_px"] == 10.0
        assert d["crack_width_mm"] == 0.25


# ---------------------------------------------------------------------------
# Phase 2: decode_detections
# ---------------------------------------------------------------------------

class TestDecodeDetections:
    def _make_raw(self, cx, cy, bw, bh, obj_conf, cls_conf, num_anchors=8400):
        """
        Construct a fake ONNX detection output:
        shape (1, 6, num_anchors) with one real detection at index 0.
        """
        raw = np.zeros((1, 6, num_anchors), dtype=np.float32)
        raw[0, 0, 0] = cx
        raw[0, 1, 0] = cy
        raw[0, 2, 0] = bw
        raw[0, 3, 0] = bh
        raw[0, 4, 0] = obj_conf
        raw[0, 5, 0] = cls_conf
        return raw

    def test_no_detections_below_threshold(self):
        from api.detector import decode_detections
        raw = self._make_raw(320, 320, 100, 100, 0.1, 0.1)  # score=0.01 < 0.25
        boxes = decode_detections(raw, scale=1.0, pad_x=0, pad_y=0,
                                  orig_h=640, orig_w=640)
        assert boxes == []

    def test_detects_one_box_above_threshold(self):
        from api.detector import decode_detections
        raw = self._make_raw(320, 320, 200, 200, 0.9, 0.9)   # score=0.81
        boxes = decode_detections(raw, scale=1.0, pad_x=0, pad_y=0,
                                  orig_h=640, orig_w=640)
        assert len(boxes) == 1

    def test_box_coords_within_image(self):
        from api.detector import decode_detections
        raw = self._make_raw(320, 320, 200, 200, 0.9, 0.9)
        boxes = decode_detections(raw, scale=1.0, pad_x=0, pad_y=0,
                                  orig_h=640, orig_w=640)
        b = boxes[0]
        assert 0 <= b.x1 < b.x2 <= 640
        assert 0 <= b.y1 < b.y2 <= 640

    def test_confidence_set_correctly(self):
        from api.detector import decode_detections
        raw = self._make_raw(320, 320, 100, 100, 0.9, 0.95)  # score=0.855
        boxes = decode_detections(raw, scale=1.0, pad_x=0, pad_y=0,
                                  orig_h=640, orig_w=640)
        if boxes:
            assert 0.0 <= boxes[0].confidence <= 1.0


# ---------------------------------------------------------------------------
# Phase 2: draw_boxes_overlay
# ---------------------------------------------------------------------------

class TestDrawBoxesOverlay:
    def _make_boxes(self):
        from api.detector import BoundingBox
        return [
            BoundingBox(10, 10, 100, 60, 0.9, "severe"),
            BoundingBox(130, 80, 220, 160, 0.7, "hairline"),
        ]

    def test_returns_same_shape(self, blank_img):
        from api.detector import draw_boxes_overlay
        result = draw_boxes_overlay(blank_img, self._make_boxes())
        assert result.shape == blank_img.shape

    def test_does_not_modify_original(self, blank_img):
        from api.detector import draw_boxes_overlay
        original = blank_img.copy()
        draw_boxes_overlay(blank_img, self._make_boxes())
        assert np.array_equal(blank_img, original)

    def test_boxes_drawn_pixel_diff(self, blank_img):
        from api.detector import draw_boxes_overlay
        result = draw_boxes_overlay(blank_img, self._make_boxes())
        assert not np.array_equal(result, blank_img)

    def test_empty_boxes_returns_copy(self, blank_img):
        from api.detector import draw_boxes_overlay
        result = draw_boxes_overlay(blank_img, [])
        assert np.array_equal(result, blank_img)


# ---------------------------------------------------------------------------
# Phase 1 severity heuristic (kept)
# ---------------------------------------------------------------------------

class TestSeverityHeuristicCls:
    def test_not_cracked_returns_none(self, blank_img):
        from api.detector import estimate_severity_cls
        assert estimate_severity_cls(blank_img, is_cracked=False) == "none"

    def test_cracked_returns_valid_severity(self, synthetic_crack_img):
        from api.detector import estimate_severity_cls
        result = estimate_severity_cls(synthetic_crack_img, is_cracked=True)
        assert result in ("hairline", "moderate", "severe")

    def test_draw_overlay_same_shape(self, synthetic_crack_img):
        from api.detector import draw_severity_overlay
        assert draw_severity_overlay(synthetic_crack_img).shape == synthetic_crack_img.shape


# ---------------------------------------------------------------------------
# MockDetector (Phase 2 — includes bounding boxes)
# ---------------------------------------------------------------------------

class TestMockDetector:
    def test_predict_returns_result(self, blank_img):
        from api.detector import MockDetector, DetectionResult
        assert isinstance(MockDetector().predict(blank_img), DetectionResult)

    def test_dark_image_likely_cracked(self, dark_img):
        from api.detector import MockDetector
        assert MockDetector().predict(dark_img).is_cracked

    def test_bright_image_likely_uncracked(self, bright_img):
        from api.detector import MockDetector
        assert not MockDetector().predict(bright_img).is_cracked

    def test_confidence_in_range(self, blank_img):
        from api.detector import MockDetector
        r = MockDetector().predict(blank_img)
        assert 0.0 <= r.confidence <= 1.0

    def test_severity_valid(self, dark_img):
        from api.detector import MockDetector
        r = MockDetector().predict(dark_img)
        assert r.severity in ("none", "hairline", "moderate", "severe")

    def test_class_probs_sum_to_one(self, blank_img):
        from api.detector import MockDetector
        r = MockDetector().predict(blank_img)
        assert abs(sum(r.class_probabilities.values()) - 1.0) < 1e-4

    def test_action_is_string(self, blank_img):
        from api.detector import MockDetector
        r = MockDetector().predict(blank_img)
        assert isinstance(r.action, str) and len(r.action) > 0

    def test_inference_time_positive(self, blank_img):
        from api.detector import MockDetector
        assert MockDetector().predict(blank_img).inference_time_ms > 0

    def test_cracked_has_bounding_boxes(self, dark_img):
        """Phase 2: cracked images should include at least one box."""
        from api.detector import MockDetector
        r = MockDetector().predict(dark_img)
        if r.is_cracked:
            assert len(r.bounding_boxes) > 0

    def test_uncracked_has_no_boxes(self, bright_img):
        """Phase 2: uncracked images should have no boxes."""
        from api.detector import MockDetector
        r = MockDetector().predict(bright_img)
        if not r.is_cracked:
            assert len(r.bounding_boxes) == 0

    def test_boxes_have_valid_coords(self, dark_img):
        from api.detector import MockDetector
        r = MockDetector().predict(dark_img)
        for b in r.bounding_boxes:
            assert b.x1 < b.x2
            assert b.y1 < b.y2
            assert 0.0 <= b.confidence <= 1.0

    def test_detection_mode_is_mock(self, blank_img):
        from api.detector import MockDetector
        assert MockDetector().predict(blank_img).detection_mode == "mock"

    def test_mock_generates_width_values(self, dark_img):
        from api.detector import MockDetector
        r = MockDetector().predict(dark_img)
        for b in r.bounding_boxes:
            assert b.width_px is not None and b.width_px > 0
            assert b.width_mm is not None and b.width_mm > 0

    def test_boxes_as_dicts(self, dark_img):
        from api.detector import MockDetector
        r = MockDetector().predict(dark_img)
        for d in r.boxes_as_dicts():
            for key in ("x1","y1","x2","y2","confidence","severity","crack_width_px","crack_width_mm"):
                assert key in d


# ---------------------------------------------------------------------------
# Database tests (Phase 2 columns)
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_init_creates_table(self, tmp_db):
        conn = tmp_db.get_connection()
        tables = [t[0] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "inspections" in tables

    def test_phase3_columns_exist(self, tmp_db):
        conn = tmp_db.get_connection()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()}
        assert "bounding_boxes" in cols
        assert "box_count"      in cols
        assert "detection_mode" in cols
        assert "scale_mm_per_px" in cols

    def test_save_returns_id(self, tmp_db):
        row_id = tmp_db.save_inspection(
            is_cracked=True, confidence=0.95, severity="moderate",
            action="Schedule inspection", class_probs={"cracked": 0.95, "uncracked": 0.05},
            infer_ms=12.5, model_ver="mock",
        )
        assert isinstance(row_id, int) and row_id > 0

    def test_save_with_bounding_boxes(self, tmp_db):
        boxes = [{"x1":10,"y1":20,"x2":60,"y2":80,"confidence":0.9,"severity":"moderate","crack_width_px":10.5,"crack_width_mm":0.4}]
        row_id = tmp_db.save_inspection(
            is_cracked=True, confidence=0.9, severity="moderate",
            action="Schedule inspection", class_probs={"cracked":0.9,"uncracked":0.1},
            infer_ms=18.0, bounding_boxes=boxes, detection_mode="detection",
            scale_mm_per_px=0.8,
        )
        rows = tmp_db.get_inspections(limit=1)
        assert rows[0]["box_count"] == 1
        assert rows[0]["detection_mode"] == "detection"
        assert rows[0]["scale_mm_per_px"] == 0.8
        assert isinstance(rows[0]["bounding_boxes"], list)
        assert rows[0]["bounding_boxes"][0]["severity"] == "moderate"
        assert rows[0]["bounding_boxes"][0]["crack_width_mm"] == 0.4

    def test_default_detection_mode_is_classification(self, tmp_db):
        tmp_db.save_inspection(
            is_cracked=False, confidence=0.88, severity="none",
            action="No action", class_probs={}, infer_ms=8.0,
        )
        rows = tmp_db.get_inspections()
        assert rows[0]["detection_mode"] == "classification"

    def test_filter_by_mode(self, tmp_db):
        for mode in ["detection", "classification", "mock"]:
            tmp_db.save_inspection(
                is_cracked=True, confidence=0.8, severity="moderate",
                action="a", class_probs={}, infer_ms=10.0,
                detection_mode=mode,
            )
        det_rows = tmp_db.get_inspections(mode="detection")
        assert len(det_rows) == 1
        assert det_rows[0]["detection_mode"] == "detection"

    def test_save_and_retrieve_gps(self, tmp_db):
        tmp_db.save_inspection(
            is_cracked=True, confidence=0.9, severity="severe",
            action="Immediate action", class_probs={}, infer_ms=10.0,
            latitude=36.8, longitude=10.2, location_name="Tunis Bridge",
        )
        row = tmp_db.get_inspections(limit=1)[0]
        assert row["latitude"]      == pytest.approx(36.8)
        assert row["location_name"] == "Tunis Bridge"

    def test_stats_empty(self, tmp_db):
        s = tmp_db.get_stats()
        assert s["total"] == 0

    def test_stats_after_insertions(self, tmp_db):
        for cracked, sev in [(True, "moderate"), (True, "severe"), (False, "none")]:
            tmp_db.save_inspection(
                is_cracked=cracked, confidence=0.9, severity=sev,
                action="action", class_probs={}, infer_ms=10.0,
            )
        s = tmp_db.get_stats()
        assert s["total"]   == 3
        assert s["cracked"] == 2
        assert s["crack_rate_pct"] == pytest.approx(66.7, abs=0.1)

    def test_stats_avg_boxes(self, tmp_db):
        tmp_db.save_inspection(
            is_cracked=True, confidence=0.9, severity="moderate",
            action="a", class_probs={}, infer_ms=10.0,
            bounding_boxes=[{"x1":0,"y1":0,"x2":10,"y2":10,"confidence":0.8,"severity":"hairline"},
                             {"x1":20,"y1":20,"x2":50,"y2":50,"confidence":0.7,"severity":"moderate"}],
            detection_mode="detection",
        )
        s = tmp_db.get_stats()
        assert s["avg_boxes_per_crack"] == pytest.approx(2.0)

    def test_filter_by_severity(self, tmp_db):
        for sev in ["hairline", "moderate", "severe", "none"]:
            tmp_db.save_inspection(
                is_cracked=sev!="none", confidence=0.9, severity=sev,
                action="a", class_probs={}, infer_ms=5.0,
            )
        severes = tmp_db.get_inspections(severity="severe")
        assert len(severes) == 1
        assert severes[0]["severity"] == "severe"

    def test_delete_inspection(self, tmp_db):
        row_id = tmp_db.save_inspection(
            is_cracked=True, confidence=0.8, severity="hairline",
            action="Monitor", class_probs={}, infer_ms=9.0,
        )
        assert tmp_db.delete_inspection(row_id) is True
        assert tmp_db.get_inspections() == []

    def test_delete_nonexistent(self, tmp_db):
        assert tmp_db.delete_inspection(9999) is False
