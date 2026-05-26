"""
Concrete Crack Detector API — FastAPI  (Phase 2: Bounding Box Detection)
=========================================================================
Endpoints:
  POST /detect          — image upload → crack detection + bounding boxes
  GET  /inspections     — inspection log (paginated)
  GET  /stats           — dashboard statistics
  DELETE /inspections/{id}

Run:
    uvicorn api.main:app --reload --port 8001
Docs:
    http://localhost:8001/docs
"""

import io
import os
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.detector import (
    load_detector,
    draw_boxes_overlay,
    draw_severity_overlay,
    DetectionResult,
)
from api import database as db

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

UPLOADS_DIR = Path(__file__).parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="CrackScan API — Phase 3 Width-Based Severity",
    description=(
        "AI-powered concrete surface crack detection with bounding box localisation\n"
        "and physically-grounded crack width measurement (Phase 3).\n\n"
        "Upload an image → get crack classification, bounding boxes, measured\n"
        "crack widths in mm, severity per EN 206, and recommended action.\n"
        "All inspections are logged with optional GPS.\n\n"
        "**Scale**: Set `scale_mm_per_px` (Ground Sampling Distance) to match\n"
        "your camera setup. Default 0.78 mm/px is calibrated for SDNET2018.\n\n"
        "**Model priority**: Phase 2 detection (crack_detector_det.onnx) → "
        "Phase 1 classification (crack_detector.onnx) → MockDetector."
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
    mode = getattr(detector, '__class__', type(detector)).__name__
    print(f"[startup] API ready — detector: {mode}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BoundingBoxOut(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    confidence: float
    severity: str
    crack_width_px: Optional[float] = None   # Phase 3: measured crack width in pixels
    crack_width_mm: Optional[float] = None   # Phase 3: measured crack width in mm


class DetectResponse(BaseModel):
    inspection_id: int
    is_cracked: bool
    confidence: float
    severity: str
    severity_color: str
    action: str
    class_probabilities: dict
    inference_time_ms: float
    overlay_url: str
    model_version: str
    detection_mode: str                     # "detection" | "classification" | "mock"
    bounding_boxes: list[BoundingBoxOut]    # empty list for cls/mock modes
    box_count: int
    scale_mm_per_px: float                  # Phase 3: GSD used for this inspection


class StatsResponse(BaseModel):
    total: int
    cracked: int
    uncracked: int
    crack_rate_pct: float
    by_severity: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE_MB   = 10


def read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image. Ensure it is a valid JPEG/PNG/WebP.",
        )
    return img


def save_image(img_bgr: np.ndarray, suffix: str = "") -> tuple[str, str]:
    filename = f"{uuid.uuid4().hex}{suffix}.jpg"
    path = UPLOADS_DIR / filename
    cv2.imwrite(str(path), img_bgr)
    return filename, f"/uploads/{filename}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    mode = getattr(detector, "__class__", type(detector)).__name__
    return {
        "status": "ok",
        "phase": 2,
        "detector": mode,
        "message": "CrackScan API v2 — bounding box detection active. See /docs",
    }


@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
async def detect(
    file:          UploadFile        = File(..., description="Concrete surface image (JPEG/PNG/WebP)"),
    latitude:      Optional[float]   = Form(None, description="GPS latitude"),
    longitude:     Optional[float]   = Form(None, description="GPS longitude"),
    location_name: Optional[str]     = Form(None, description="Human-readable location"),
    user_note:     Optional[str]     = Form(None, description="Inspector note"),
    conf_threshold: Optional[float]  = Form(None, description="Override detection confidence threshold (0.1–0.9)"),
    scale_mm_per_px: Optional[float] = Form(None, description="Ground Sampling Distance in mm/px. Default 0.78 for SDNET2018."),
):
    """
    Upload a concrete surface image and receive:

    - **is_cracked**: binary classification
    - **bounding_boxes**: list of detected crack regions (Phase 2 model only)
    - **severity**: worst-case severity across all boxes
    - **overlay_url**: annotated image with coloured bounding boxes drawn
    - **detection_mode**: `detection` | `classification` | `mock`
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    raw = await file.read()
    if len(raw) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image too large. Max {MAX_SIZE_MB} MB.")

    img = read_image(raw)

    # Phase 3: resolve GSD scale
    from api.detector import DEFAULT_SCALE_MM_PER_PX
    effective_scale = scale_mm_per_px if scale_mm_per_px and scale_mm_per_px > 0 else DEFAULT_SCALE_MM_PER_PX

    # Run detection
    result: DetectionResult = detector.predict(img, scale_mm_per_px=effective_scale)

    # Build annotated overlay
    if result.detection_mode == "detection" and result.bounding_boxes:
        overlay_img = draw_boxes_overlay(img, result.bounding_boxes)
    elif result.is_cracked:
        overlay_img = draw_severity_overlay(img)   # Phase 1 contour fallback
    else:
        overlay_img = img

    _, orig_url    = save_image(img, "_orig")
    _, overlay_url = save_image(overlay_img, "_overlay")

    # Persist to DB
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
        scale_mm_per_px=effective_scale,
    )

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
        bounding_boxes=[BoundingBoxOut(**b) for b in result.boxes_as_dicts()],
        box_count=len(result.bounding_boxes),
        scale_mm_per_px=effective_scale,
    )


@app.get("/inspections", tags=["Log"])
def list_inspections(
    limit:    int            = Query(50, ge=1, le=200),
    offset:   int            = Query(0, ge=0),
    severity: Optional[str] = Query(None, description="Filter: none/hairline/moderate/severe"),
    mode:     Optional[str] = Query(None, description="Filter by detection_mode: detection/classification/mock"),
):
    """Return paginated inspection history, newest first."""
    rows = db.get_inspections(limit=limit, offset=offset, severity=severity, mode=mode)
    return {"count": len(rows), "inspections": rows}


@app.get("/stats", response_model=StatsResponse, tags=["Dashboard"])
def stats():
    """Aggregate statistics for the dashboard."""
    return db.get_stats()


@app.delete("/inspections/{inspection_id}", tags=["Log"])
def delete_inspection(inspection_id: int):
    deleted = db.delete_inspection(inspection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    return {"deleted": inspection_id}
