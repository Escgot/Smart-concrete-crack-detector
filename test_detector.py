"""
Tests for crack detector engine and database.
Run: pytest tests/ -v

These tests use MockDetector and a temp database — no model weights needed.
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
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_output_shape(self, blank_img):
        from api.detector import preprocess
        tensor = preprocess(blank_img)
        assert tensor.shape == (1, 3, 224, 224)

    def test_output_dtype(self, blank_img):
        from api.detector import preprocess
        tensor = preprocess(blank_img)
        assert tensor.dtype == np.float32

    def test_values_normalised(self, blank_img):
        """After ImageNet normalisation, values should be roughly in [-3, 3]"""
        from api.detector import preprocess
        tensor = preprocess(blank_img)
        assert tensor.min() > -4.0
        assert tensor.max() < 4.0

    def test_accepts_various_sizes(self):
        from api.detector import preprocess
        for shape in [(100, 100, 3), (1024, 768, 3), (64, 256, 3)]:
            img = np.zeros(shape, dtype=np.uint8)
            tensor = preprocess(img)
            assert tensor.shape == (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Softmax tests
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        from api.detector import softmax
        logits = np.array([1.5, -0.3, 2.1, 0.0])
        probs = softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_max_corresponds_to_argmax(self):
        from api.detector import softmax
        logits = np.array([0.1, 5.0, -1.0])
        probs = softmax(logits)
        assert np.argmax(probs) == 1

    def test_numerical_stability(self):
        """Large logits should not produce NaN/Inf"""
        from api.detector import softmax
        logits = np.array([1000.0, 999.0, 998.0])
        probs = softmax(logits)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))


# ---------------------------------------------------------------------------
# Severity heuristic tests
# ---------------------------------------------------------------------------

class TestSeverityHeuristic:
    def test_not_cracked_returns_none(self, blank_img):
        from api.detector import estimate_severity
        assert estimate_severity(blank_img, is_cracked=False) == "none"

    def test_cracked_returns_severity(self, synthetic_crack_img):
        from api.detector import estimate_severity
        result = estimate_severity(synthetic_crack_img, is_cracked=True)
        assert result in ("hairline", "moderate", "severe")

    def test_clean_surface_hairline(self, bright_img):
        """Very uniform bright surface should give hairline if cracked=True"""
        from api.detector import estimate_severity
        # bright uniform image has very few edges
        result = estimate_severity(bright_img, is_cracked=True)
        assert result in ("hairline", "moderate")  # should be low severity


class TestSeverityOverlay:
    def test_returns_same_shape(self, synthetic_crack_img):
        from api.detector import draw_severity_overlay
        overlay = draw_severity_overlay(synthetic_crack_img)
        assert overlay.shape == synthetic_crack_img.shape

    def test_returns_ndarray(self, blank_img):
        from api.detector import draw_severity_overlay
        result = draw_severity_overlay(blank_img)
        assert isinstance(result, np.ndarray)

    def test_does_not_modify_original(self, synthetic_crack_img):
        from api.detector import draw_severity_overlay
        original = synthetic_crack_img.copy()
        draw_severity_overlay(synthetic_crack_img)
        assert np.array_equal(synthetic_crack_img, original)


# ---------------------------------------------------------------------------
# MockDetector tests
# ---------------------------------------------------------------------------

class TestMockDetector:
    def test_predict_returns_result(self, blank_img):
        from api.detector import MockDetector, DetectionResult
        det = MockDetector()
        result = det.predict(blank_img)
        assert isinstance(result, DetectionResult)

    def test_dark_image_likely_cracked(self, dark_img):
        from api.detector import MockDetector
        det = MockDetector()
        result = det.predict(dark_img)
        # Dark images should be classified as cracked by mock logic
        assert result.is_cracked

    def test_bright_image_likely_uncracked(self, bright_img):
        from api.detector import MockDetector
        det = MockDetector()
        result = det.predict(bright_img)
        assert not result.is_cracked

    def test_confidence_in_range(self, blank_img):
        from api.detector import MockDetector
        det = MockDetector()
        result = det.predict(blank_img)
        assert 0.0 <= result.confidence <= 1.0

    def test_severity_valid(self, dark_img):
        from api.detector import MockDetector
        det = MockDetector()
        result = det.predict(dark_img)
        assert result.severity in ("none", "hairline", "moderate", "severe")

    def test_class_probs_sum_to_one(self, blank_img):
        from api.detector import MockDetector
        det = MockDetector()
        result = det.predict(blank_img)
        total = sum(result.class_probabilities.values())
        assert abs(total - 1.0) < 1e-4

    def test_action_is_string(self, blank_img):
        from api.detector import MockDetector
        det = MockDetector()
        result = det.predict(blank_img)
        assert isinstance(result.action, str) and len(result.action) > 0

    def test_inference_time_positive(self, blank_img):
        from api.detector import MockDetector
        det = MockDetector()
        result = det.predict(blank_img)
        assert result.inference_time_ms > 0


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_init_creates_table(self, tmp_db):
        import sqlite3
        conn = tmp_db.get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "inspections" in table_names

    def test_save_returns_id(self, tmp_db):
        row_id = tmp_db.save_inspection(
            is_cracked=True, confidence=0.95, severity="moderate",
            action="Schedule inspection", class_probs={"cracked": 0.95, "uncracked": 0.05},
            infer_ms=12.5, model_ver="mock",
        )
        assert isinstance(row_id, int) and row_id > 0

    def test_save_and_retrieve(self, tmp_db):
        tmp_db.save_inspection(
            is_cracked=True, confidence=0.9, severity="severe",
            action="Immediate action", class_probs={"cracked": 0.9, "uncracked": 0.1},
            infer_ms=10.0, latitude=36.8, longitude=10.2,
            location_name="Tunis Bridge", user_note="East face",
            model_ver="test",
        )
        rows = tmp_db.get_inspections(limit=10)
        assert len(rows) == 1
        row = rows[0]
        assert row["is_cracked"] is True
        assert row["severity"] == "severe"
        assert row["latitude"] == pytest.approx(36.8)
        assert row["location_name"] == "Tunis Bridge"

    def test_class_probs_deserialised(self, tmp_db):
        tmp_db.save_inspection(
            is_cracked=False, confidence=0.88, severity="none",
            action="No action", class_probs={"cracked": 0.12, "uncracked": 0.88},
            infer_ms=8.0,
        )
        rows = tmp_db.get_inspections()
        assert isinstance(rows[0]["class_probs"], dict)

    def test_stats_empty(self, tmp_db):
        s = tmp_db.get_stats()
        assert s["total"] == 0
        assert s["cracked"] == 0

    def test_stats_after_insertions(self, tmp_db):
        for cracked, sev in [(True, "moderate"), (True, "severe"), (False, "none")]:
            tmp_db.save_inspection(
                is_cracked=cracked, confidence=0.9, severity=sev,
                action="action", class_probs={}, infer_ms=10.0,
            )
        s = tmp_db.get_stats()
        assert s["total"] == 3
        assert s["cracked"] == 2
        assert s["uncracked"] == 1
        assert s["crack_rate_pct"] == pytest.approx(66.7, abs=0.1)

    def test_filter_by_severity(self, tmp_db):
        for sev in ["hairline", "moderate", "severe", "none"]:
            tmp_db.save_inspection(
                is_cracked=sev != "none", confidence=0.9, severity=sev,
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
        deleted = tmp_db.delete_inspection(row_id)
        assert deleted is True
        rows = tmp_db.get_inspections()
        assert len(rows) == 0

    def test_delete_nonexistent(self, tmp_db):
        deleted = tmp_db.delete_inspection(9999)
        assert deleted is False
