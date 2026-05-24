"""
Concrete Crack Detector API — FastAPI
Endpoints: /detect (image upload), /inspections (log), /stats (dashboard)

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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.detector import load_detector, draw_severity_overlay, DetectionResult
from api import database as db

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

UPLOADS_DIR = Path(__file__).parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Concrete Crack Detector API",
    description=(
        "AI-powered concrete surface crack detection.\n\n"
        "Upload an image → get crack classification, severity, and recommended action.\n"
        "All inspections are logged with optional GPS coordinates."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Load detector at startup (once)
detector = None

@app.on_event("startup")
def startup():
    global detector
    db.init_db()
    detector = load_detector()
    print("[startup] API ready.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class InspectionOut(BaseModel):
    id: int
    created_at: str
    is_cracked: bool
    confidence: float
    severity: str
    severity_color: str
    action: str
    class_probs: dict
    infer_ms: float
    latitude: Optional[float]
    longitude: Optional[float]
    location_name: Optional[str]
    user_note: Optional[str]
    image_url: Optional[str]
    model_ver: str


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
MAX_SIZE_MB = 10


def read_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes to BGR numpy array."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image. Ensure it is a valid JPEG/PNG.")
    return img


def save_image(img_bgr: np.ndarray, suffix: str = "") -> tuple[str, str]:
    """Save image to uploads dir. Returns (filename, URL path)."""
    filename = f"{uuid.uuid4().hex}{suffix}.jpg"
    path = UPLOADS_DIR / filename
    cv2.imwrite(str(path), img_bgr)
    return filename, f"/uploads/{filename}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "model": getattr(detector, "__class__", type(detector)).__name__,
        "message": "Concrete Crack Detector API — see /docs",
    }


@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
async def detect(
    file: UploadFile = File(..., description="Concrete surface image (JPEG/PNG/WebP)"),
    latitude:      Optional[float] = Form(None, description="GPS latitude"),
    longitude:     Optional[float] = Form(None, description="GPS longitude"),
    location_name: Optional[str]   = Form(None, description="Human-readable location"),
    user_note:     Optional[str]   = Form(None, description="Inspector note"),
):
    """
    Upload a concrete surface image and get:
    - Binary classification: cracked / uncracked
    - Severity: none / hairline / moderate / severe
    - Recommended action
    - Annotated overlay image URL
    - Inspection saved to the log
    """
    # Validate
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP."
        )

    raw = await file.read()
    if len(raw) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image too large. Max {MAX_SIZE_MB} MB.")

    img = read_image(raw)

    # Run detection
    result: DetectionResult = detector.predict(img)

    # Save original + overlay
    _, orig_url = save_image(img, "_orig")
    overlay_img  = draw_severity_overlay(img) if result.is_cracked else img
    _, overlay_url = save_image(overlay_img, "_overlay")

    # Log to database
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
    )


@app.get("/inspections", tags=["Log"])
def list_inspections(
    limit:    int = Query(50, ge=1, le=200),
    offset:   int = Query(0, ge=0),
    severity: Optional[str] = Query(None, description="Filter: none/hairline/moderate/severe"),
):
    """Return paginated inspection history, newest first."""
    rows = db.get_inspections(limit=limit, offset=offset, severity=severity)
    return {"count": len(rows), "inspections": rows}


@app.get("/stats", response_model=StatsResponse, tags=["Dashboard"])
def stats():
    """Aggregate statistics for the dashboard."""
    return db.get_stats()


@app.delete("/inspections/{inspection_id}", tags=["Log"])
def delete_inspection(inspection_id: int):
    """Delete an inspection record by ID."""
    deleted = db.delete_inspection(inspection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    return {"deleted": inspection_id}
