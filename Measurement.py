"""
CrackScan Phase 3 — Crack Measurement Engine
=============================================
Converts binary segmentation masks into physical measurements (mm).

Pipeline per crack:
  binary mask → scipy distance transform → skimage skeletonize
  → width/length/area → EN 206 classification

Gracefully degrades to a contour-based approximation when
scipy/skimage are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize as _sk_skeletonize
    _FULL_PIPELINE = True
except ImportError:  # pragma: no cover
    _FULL_PIPELINE = False


# ── EN 206 width thresholds ──────────────────────────────────

def _en206_class(max_width_mm: float) -> str:
    """Physics-grounded crack classification per EN 206 / structural practice."""
    if max_width_mm <= 0.0:
        return "none"
    elif max_width_mm < 0.20:
        return "hairline"
    elif max_width_mm <= 0.50:
        return "moderate"
    else:
        return "severe"


# ── Data class ────────────────────────────────────────────────

@dataclass
class CrackMeasurement:
    """Physical measurements for a single segmented crack region."""
    max_width_mm: float = 0.0
    mean_width_mm: float = 0.0
    length_mm: float = 0.0
    area_mm2: float = 0.0
    width_uncertainty_mm: float = 0.0   # 1-sigma, propagated from calibration
    pixel_count: int = 0
    skeleton_pixels: int = 0
    en206_class: str = "none"
    method: str = "none"               # "skeleton" | "contour" | "none"

    def to_dict(self) -> dict:
        return {
            "max_width_mm":       round(self.max_width_mm, 3),
            "mean_width_mm":      round(self.mean_width_mm, 3),
            "length_mm":          round(self.length_mm, 1),
            "area_mm2":           round(self.area_mm2, 2),
            "width_uncertainty_mm": round(self.width_uncertainty_mm, 3),
            "en206_class":        self.en206_class,
            "method":             self.method,
        }


# ── Core measurement ─────────────────────────────────────────

def measure_mask(
    mask: np.ndarray,
    px_per_mm: float,
    calibration_uncertainty: float = 0.0,
) -> CrackMeasurement:
    """
    Measure a single binary crack mask.

    Args:
        mask:                    H×W uint8 mask (crack=255, background=0).
        px_per_mm:               Pixels per millimetre from calibration.
        calibration_uncertainty: Fractional uncertainty in px_per_mm (0–1).
                                 Propagated into width_uncertainty_mm.

    Returns:
        CrackMeasurement with physical dimensions.
    """
    if px_per_mm <= 0:
        px_per_mm = 21.3  # SDNET default

    binary = (mask > 127).astype(np.uint8)
    pixel_count = int(binary.sum())

    if pixel_count < 10:
        return CrackMeasurement()

    if _FULL_PIPELINE:
        return _measure_skeleton(binary, px_per_mm, calibration_uncertainty, pixel_count)
    else:
        return _measure_contour(binary, px_per_mm, calibration_uncertainty, pixel_count)


def _measure_skeleton(
    binary: np.ndarray,
    px_per_mm: float,
    uncertainty_frac: float,
    pixel_count: int,
) -> CrackMeasurement:
    """
    Primary path: Euclidean Distance Transform + Skeletonization.

    Width at each skeleton pixel = 2 × dist[pixel] (dist gives radius to
    nearest background, so diameter = crack width at that cross-section).
    """
    dist = distance_transform_edt(binary)
    skeleton = _sk_skeletonize(binary.astype(bool))
    skel_pixels = int(skeleton.sum())

    if skel_pixels == 0:
        return CrackMeasurement(pixel_count=pixel_count)

    widths_px = dist[skeleton] * 2.0          # radii → diameters
    max_w_px   = float(widths_px.max())
    mean_w_px  = float(widths_px.mean())

    max_w_mm   = max_w_px   / px_per_mm
    mean_w_mm  = mean_w_px  / px_per_mm
    length_mm  = skel_pixels / px_per_mm
    area_mm2   = pixel_count / (px_per_mm ** 2)

    # Uncertainty propagation: δw/w = δ(px_per_mm)/px_per_mm
    uncertainty_mm = max_w_mm * uncertainty_frac

    return CrackMeasurement(
        max_width_mm=max_w_mm,
        mean_width_mm=mean_w_mm,
        length_mm=length_mm,
        area_mm2=area_mm2,
        width_uncertainty_mm=uncertainty_mm,
        pixel_count=pixel_count,
        skeleton_pixels=skel_pixels,
        en206_class=_en206_class(max_w_mm),
        method="skeleton",
    )


def _measure_contour(
    binary: np.ndarray,
    px_per_mm: float,
    uncertainty_frac: float,
    pixel_count: int,
) -> CrackMeasurement:
    """
    Fallback path when scipy/skimage unavailable.
    Approximates width via minimum bounding rectangle of the largest contour.
    Accuracy is lower (~±25 %) but requires no extra dependencies.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return CrackMeasurement(pixel_count=pixel_count)

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    dims = sorted(rect[1])                      # (short_side, long_side)
    width_px, length_px = dims[0], dims[1]

    max_w_mm  = width_px  / px_per_mm
    length_mm = length_px / px_per_mm
    area_mm2  = pixel_count / (px_per_mm ** 2)
    uncertainty_mm = max_w_mm * max(uncertainty_frac, 0.25)   # ≥25 % for contour method

    return CrackMeasurement(
        max_width_mm=max_w_mm,
        mean_width_mm=max_w_mm * 0.70,          # empirical: mean ≈ 70 % of max
        length_mm=length_mm,
        area_mm2=area_mm2,
        width_uncertainty_mm=uncertainty_mm,
        pixel_count=pixel_count,
        en206_class=_en206_class(max_w_mm),
        method="contour",
    )


# ── Aggregation ───────────────────────────────────────────────

def aggregate_measurements(measurements: list[CrackMeasurement]) -> dict:
    """
    Reduce per-crack measurements to image-level summary.

    Returns the worst-case (maximum) width and total affected geometry,
    which are the inputs to EN 206 / structural triage decision logic.
    """
    if not measurements:
        return {
            "max_width_mm":       0.0,
            "total_length_mm":    0.0,
            "total_area_mm2":     0.0,
            "crack_count":        0,
            "en206_class":        "none",
            "width_uncertainty_mm": 0.0,
        }

    worst = max(measurements, key=lambda m: m.max_width_mm)
    total_length = sum(m.length_mm for m in measurements)
    total_area   = sum(m.area_mm2  for m in measurements)

    return {
        "max_width_mm":         round(worst.max_width_mm, 3),
        "total_length_mm":      round(total_length, 1),
        "total_area_mm2":       round(total_area, 2),
        "crack_count":          len(measurements),
        "en206_class":          worst.en206_class,
        "width_uncertainty_mm": round(worst.width_uncertainty_mm, 3),
    }


# ── Overlay drawing ───────────────────────────────────────────

_MASK_COLOR  = (0, 220, 100)   # BGR green tint for mask
_SKEL_COLOR  = (0, 80, 255)    # BGR red for skeleton

_SEV_BGR = {
    "none":     (100, 200, 100),
    "hairline": (200, 200,   0),
    "moderate": (  0, 160, 255),
    "severe":   (  0,   0, 220),
}


def draw_segmentation_overlay(
    img_bgr: np.ndarray,
    boxes_with_masks: list,            # list[BoundingBox] from detector
) -> np.ndarray:
    """
    Render segmentation masks, bounding rectangles, and measurement
    annotations on a copy of the original image.
    """
    overlay = img_bgr.copy()
    color_layer = np.zeros_like(overlay)

    for box in boxes_with_masks:
        if box.mask is None:
            continue
        color = _SEV_BGR.get(box.severity, _MASK_COLOR)
        color_layer[box.mask > 127] = color

    cv2.addWeighted(color_layer, 0.40, overlay, 0.60, 0, overlay)

    for box in boxes_with_masks:
        color = _SEV_BGR.get(box.severity, _MASK_COLOR)
        cv2.rectangle(overlay, (box.x1, box.y1), (box.x2, box.y2), color, 2)

        m = box.measurement or {}
        if m.get("max_width_mm", 0) > 0:
            label = (
                f"{m['max_width_mm']:.2f}mm "
                f"±{m.get('width_uncertainty_mm', 0):.2f} "
                f"[{m.get('en206_class', '?').upper()}]"
            )
        else:
            label = f"{box.severity.upper()} ({box.confidence:.0%})"

        label_y = max(box.y1 - 6, 14)
        cv2.putText(overlay, label, (box.x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),     3, cv2.LINE_AA)
        cv2.putText(overlay, label, (box.x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay
