"""
CrackScan API — FastAPI  (Phase 3: Segmentation + Measurement)
=============================================================
Endpoints:
  POST /detect           — image + calibration params → crack detection
  GET  /inspections      — paginated inspection log
  GET  /stats            — dashboard statistics
  GET  /growth           — crack width trend for a GPS location
  DELETE /inspections/{id}

Model priority (auto-detected at startup):
  crack_detector_seg.onnx  →  Phase 3: pixel masks + EN 206 mm measurements
  crack_detector_det.onnx  →  Phase 2: bounding boxes + area severity
  crack_detector.onnx      →  Phase 1: binary classification
  MockDetector             →  deterministic fallback, no weights required

Run:
    uvicorn api.main:app --reload --port 8001
Docs:
    http://localhost:8001/docs
"""

from __future__ import annotations

import struct
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.detector import (
    load_detector,
    draw_segmentation_overlay,
    draw_boxes_overlay,
    draw_severity_overlay,
    DetectionResult,
)
from api import database as db

# ── Startup ───────────────────────────────────────────────────

UPLOADS_DIR = Path(__file__).parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="CrackScan API — Phase 3 Segmentation + Measurement",
    description=(
        "AI-powered concrete crack detection with pixel-level segmentation "
        "and EN 206–informed physical measurements (mm).\n\n"
        "**Calibration hierarchy**: ArUco card > LiDAR > EXIF focal length "
        "> SDNET empirical default.\n\n"
        "**Model priority**: Phase 3 seg → Phase 2 det → Phase 1 cls → Mock."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

detector = None


@app.on_event("startup")
def startup():
    global detector
    db.init_db()
    detector = load_detector()


# ── Pydantic schemas ─────────────────────────────────────────

class MeasurementOut(BaseModel):
    max_width_mm:         float
    mean_width_mm:        float = 0.0
    length_mm:            float = 0.0
    area_mm2:             float = 0.0
    width_uncertainty_mm: float = 0.0
    en206_class:          str   = "none"
    method:               str   = "none"


class BoundingBoxOut(BaseModel):
    x1:          int
    y1:          int
    x2:          int
    y2:          int
    width:       int
    height:      int
    confidence:  float
    severity:    str
    measurement: Optional[MeasurementOut] = None


class MeasurementAggregateOut(BaseModel):
    max_width_mm:         float = 0.0
    total_length_mm:      float = 0.0
    total_area_mm2:       float = 0.0
    crack_count:          int   = 0
    en206_class:          str   = "none"
    width_uncertainty_mm: float = 0.0


class DetectResponse(BaseModel):
    inspection_id:           int
    is_cracked:              bool
    confidence:              float
    severity:                str
    severity_color:          str
    action:                  str
    class_probabilities:     dict
    inference_time_ms:       float
    overlay_url:             str
    model_version:           str
    detection_mode:          str
    bounding_boxes:          list[BoundingBoxOut]
    box_count:               int
    # Phase 3
    measurement_aggregate:   Optional[MeasurementAggregateOut] = None
    calibration_method:      str   = "none"
    px_per_mm:               float = 0.0
    calibration_uncertainty: float = 0.0


class StatsResponse(BaseModel):
    total:                  int
    cracked:                int
    uncracked:              int
    crack_rate_pct:         float
    by_severity:            dict
    by_mode:                dict               = {}
    max_recorded_width_mm:  float              = 0.0
    avg_crack_width_mm:     float              = 0.0


# ── Helpers ───────────────────────────────────────────────────

_ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
_MAX_MB   = 10


def _read_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            422, "Could not decode image. Ensure it is a valid JPEG/PNG/WebP."
        )
    return img


def _save_image(img_bgr: np.ndarray, suffix: str = "") -> tuple[str, str]:
    filename = f"{uuid.uuid4().hex}{suffix}.jpg"
    path = UPLOADS_DIR / filename
    cv2.imwrite(str(path), img_bgr)
    return filename, f"/uploads/{filename}"


def _extract_exif(raw: bytes) -> Optional[bytes]:
    """
    Pull the raw EXIF APP1 segment from a JPEG byte string.
    Returns None if the file is not JPEG or contains no EXIF.
    """
    if len(raw) < 12 or raw[:2] != b"\xff\xd8":
        return None
    i = 2
    while i < len(raw) - 4:
        marker = raw[i:i+2]
        if marker == b"\xff\xe1":               # APP1 — may contain EXIF
            seg_len = struct.unpack(">H", raw[i+2:i+4])[0]
            seg_data = raw[i+4: i+4 + seg_len - 2]
            if seg_data[:6] == b"Exif\x00\x00":
                return seg_data
        # Skip over other segments
        if raw[i] != 0xff:
            break
        seg_len = struct.unpack(">H", raw[i+2:i+4])[0]
        i += 2 + seg_len
    return None


# ── Routes ────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    mode = type(detector).__name__
    return {
        "status":   "ok",
        "phase":    3,
        "detector": mode,
        "message":  "CrackScan API v3 — segmentation + measurement active. See /docs",
    }


@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
async def detect(
    file:            UploadFile        = File(..., description="Concrete surface image (JPEG/PNG/WebP)"),
    latitude:        Optional[float]   = Form(None, description="GPS latitude"),
    longitude:       Optional[float]   = Form(None, description="GPS longitude"),
    location_name:   Optional[str]     = Form(None, description="Human-readable location label"),
    user_note:       Optional[str]     = Form(None, description="Inspector note"),
    # Calibration inputs
    lidar_distance_m:  Optional[float] = Form(None, description="LiDAR distance to surface in metres (iOS 12 Pro+)"),
    manual_px_per_mm:  Optional[float] = Form(None, description="Override calibration: known pixels-per-mm"),
):
    """
    Submit a concrete surface image and receive:

    - **is_cracked** — binary classification
    - **bounding_boxes** — detected crack regions with per-crack measurements
    - **measurement_aggregate** — worst-case EN 206 width + total length/area
    - **calibration_method** — how px→mm was determined
    - **severity** — EN 206–grounded for Phase 3; area-heuristic for Phase 2
    - **overlay_url** — annotated image (masks/boxes drawn)

    **Calibration tip**: place a printed 50 mm ArUco DICT_4X4_50 card in
    the frame for automatic traceable calibration (±5 %).
    """
    if file.content_type not in _ALLOWED:
        raise HTTPException(
            415,
            f"Unsupported media type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    raw = await file.read()
    if len(raw) > _MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {_MAX_MB} MB limit.")

    exif_bytes = _extract_exif(raw)
    img        = _read_image(raw)

    result: DetectionResult = detector.predict(
        img,
        exif_bytes=exif_bytes,
        lidar_distance_m=lidar_distance_m,
        manual_px_per_mm=manual_px_per_mm,
    )

    # Build annotated overlay
    mode = result.detection_mode
    if mode == "segmentation" and result.bounding_boxes:
        overlay_img = draw_segmentation_overlay(img, result.bounding_boxes)
    elif mode == "detection" and result.bounding_boxes:
        overlay_img = draw_boxes_overlay(img, result.bounding_boxes)
    elif result.is_cracked:
        overlay_img = draw_severity_overlay(img)
    else:
        overlay_img = img

    _, overlay_url = _save_image(overlay_img, "_overlay")

    # Persist
    inspection_id = db.save_inspection(
        is_cracked=result.is_cracked,
        confidence=result.confidence,
        severity=result.severity,
        action=result.action,
        class_probs=result.class_probabilities,
        infer_ms=result.inference_time_ms,
        image_path=overlay_url,
        latitude=latitude,
        longitude=longitude,
        location_name=location_name,
        user_note=user_note,
        model_ver=result.model_version,
        bounding_boxes=result.boxes_as_dicts(),
        detection_mode=result.detection_mode,
        measurement_aggregate=result.measurement_aggregate,
        calibration_method=result.calibration_method,
        calibration_uncertainty=result.calibration_uncertainty,
        px_per_mm=result.px_per_mm,
    )

    # Boxes for response (strip numpy mask arrays)
    boxes_out = []
    for b in result.bounding_boxes:
        m_out = None
        if b.measurement:
            m_out = MeasurementOut(**b.measurement)
        boxes_out.append(BoundingBoxOut(
            x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
            width=b.width, height=b.height,
            confidence=b.confidence, severity=b.severity,
            measurement=m_out,
        ))

    agg_out = None
    if result.measurement_aggregate:
        agg_out = MeasurementAggregateOut(**{
            k: result.measurement_aggregate.get(k, 0)
            for k in MeasurementAggregateOut.model_fields
        })

    return DetectResponse(
        inspection_id=inspection_id,
        is_cracked=result.is_cracked,
        confidence=result.confidence,
        severity=result.severity,
        severity_color=result.severity_color,
        action=result.action,
        class_probabilities=result.class_probabilities,
        inference_time_ms=result.inference_time_ms,
        overlay_url=overlay_url,
        model_version=result.model_version,
        detection_mode=result.detection_mode,
        bounding_boxes=boxes_out,
        box_count=len(result.bounding_boxes),
        measurement_aggregate=agg_out,
        calibration_method=result.calibration_method,
        px_per_mm=result.px_per_mm,
        calibration_uncertainty=result.calibration_uncertainty,
    )


@app.get("/inspections", tags=["Log"])
def list_inspections(
    limit:    int            = Query(50, ge=1, le=200),
    offset:   int            = Query(0, ge=0),
    severity: Optional[str]  = Query(None, description="none/hairline/moderate/severe"),
    mode:     Optional[str]  = Query(None, description="segmentation/detection/classification/mock"),
):
    """Paginated inspection history, newest first."""
    rows = db.get_inspections(limit=limit, offset=offset, severity=severity, mode=mode)
    return {"count": len(rows), "inspections": rows}


@app.get("/stats", response_model=StatsResponse, tags=["Dashboard"])
def stats():
    """Aggregate statistics for the dashboard, including Phase 3 width metrics."""
    return db.get_stats()


@app.get("/growth", tags=["Analysis"])
def crack_growth(
    latitude:  float = Query(..., description="GPS latitude of the inspection point"),
    longitude: float = Query(..., description="GPS longitude of the inspection point"),
    radius_m:  float = Query(50.0, description="Search radius in metres", ge=1, le=500),
):
    """
    Return the chronological crack width measurement series for a GPS location
    and a simple trend assessment (growing / stable / closing).

    Only inspections with a Phase 3 measurement and a traceable calibration
    method (aruco / lidar / manual) are used for trend calculation.

    Example use: compare two inspections of the same bridge pier taken
    3 months apart to flag active crack growth before the next maintenance cycle.
    """
    data = db.get_crack_growth(latitude, longitude, radius_m)
    return data


@app.delete("/inspections/{inspection_id}", tags=["Log"])
def delete_inspection(inspection_id: int):
    deleted = db.delete_inspection(inspection_id)
    if not deleted:
        raise HTTPException(404, "Inspection not found.")
    return {"deleted": inspection_id}
