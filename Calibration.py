"""
CrackScan Phase 3 — Calibration Engine
=======================================
Converts pixel distances → millimetres via four methods (priority order):

  1. Manual override  — caller supplies exact px_per_mm (tests / CLI)
  2. LiDAR depth      — iOS 12 Pro+ distance injected by client app
  3. ArUco marker     — DICT_4X4_50, 50 mm card; ~5 % uncertainty
  4. EXIF focal length — rough 35mm-equivalent estimate; ~15 % uncertainty
  5. SDNET default    — empirical constant for the training dataset; ~20 %

The CalibrationResult is attached to every DetectionResult so the
measurement confidence interval can be reported to the inspector.

ArUco reference card
--------------------
Print a DICT_4X4_50 ArUco marker at exactly 50 mm × 50 mm and tape it
to the concrete surface before photographing. The detector locates the
four corners and computes px/mm from the mean side length. This is the
recommended production method for handheld phone inspection.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ── Constants ────────────────────────────────────────────────

ARUCO_CARD_SIZE_MM = 50.0       # reference card side length (mm)

# SDNET2018 empirical: images taken ~0.5 m from surface, typical lens.
# Calibrated by measuring known features in several dataset images.
SDNET_DEFAULT_PX_PER_MM  = 21.3
SDNET_DEFAULT_UNCERTAINTY = 0.20    # fractional (±20 %)


# ── Result ────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    px_per_mm:        float
    method:           str           # "manual" | "lidar" | "aruco" | "exif" | "default"
    uncertainty:      float         # fractional 1-sigma, 0–1
    marker_detected:  bool  = False
    notes:            str   = ""


# ── ArUco ─────────────────────────────────────────────────────

def _detect_aruco(img_bgr: np.ndarray) -> Optional[float]:
    """
    Detect a single DICT_4X4_50 ArUco marker and return px_per_mm.
    Compatible with OpenCV ≥ 4.7 (ArucoDetector) and the older API.
    Returns None if no marker found.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        # OpenCV ≥ 4.7
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params     = cv2.aruco.DetectorParameters()
        detector   = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        # OpenCV < 4.7 legacy fallback
        try:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            params     = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        except Exception:
            return None

    if ids is None or len(ids) == 0:
        return None

    # Use first detected marker; average all four side lengths for robustness
    c = corners[0][0]   # shape (4, 2) — corner coordinates
    side_px = float(np.mean([
        np.linalg.norm(c[1] - c[0]),
        np.linalg.norm(c[2] - c[1]),
        np.linalg.norm(c[3] - c[2]),
        np.linalg.norm(c[0] - c[3]),
    ]))

    if side_px < 5:     # implausibly small detection
        return None

    return side_px / ARUCO_CARD_SIZE_MM


# ── Minimal EXIF parser ───────────────────────────────────────
#
# We parse raw bytes to avoid a hard Pillow/piexif dependency.
# Only FocalLength and FocalLengthIn35mmFilm are needed.

def _read_u16(data: bytes, offset: int, big_endian: bool) -> int:
    fmt = ">H" if big_endian else "<H"
    return struct.unpack_from(fmt, data, offset)[0]


def _read_u32(data: bytes, offset: int, big_endian: bool) -> int:
    fmt = ">I" if big_endian else "<I"
    return struct.unpack_from(fmt, data, offset)[0]


def _read_rational(data: bytes, offset: int, big_endian: bool) -> float:
    fmt = ">II" if big_endian else "<II"
    num, den = struct.unpack_from(fmt, data, offset)
    return (num / den) if den else 0.0


def _parse_exif(raw: bytes) -> dict:
    """
    Minimal TIFF/EXIF parser. Extracts focal_length_mm and focal_35mm.
    Returns empty dict on any parse error.
    """
    # EXIF block starts with "Exif\x00\x00" then TIFF header
    if len(raw) < 8 or raw[:6] != b"Exif\x00\x00":
        return {}
    t = 6                           # TIFF header starts here
    be = raw[t:t+2] == b"MM"
    if raw[t:t+2] not in (b"MM", b"II"):
        return {}
    ifd0_off = _read_u32(raw, t + 4, be)

    def read_ifd(off: int) -> dict:
        tags: dict = {}
        try:
            n = _read_u16(raw, t + off, be)
        except Exception:
            return tags
        for i in range(n):
            base = t + off + 2 + i * 12
            try:
                tag_id  = _read_u16(raw, base,     be)
                typ     = _read_u16(raw, base + 2, be)
                count   = _read_u32(raw, base + 4, be)
                val_raw = raw[base + 8: base + 12]
                tags[tag_id] = (typ, count, val_raw)
            except Exception:
                continue
        return tags

    ifd = read_ifd(ifd0_off)

    # Resolve Exif SubIFD
    if 0x8769 in ifd:
        _, _, val_raw = ifd[0x8769]
        sub_off = _read_u32(val_raw, 0, be)
        ifd.update(read_ifd(sub_off))

    result: dict = {}

    # FocalLength (tag 0x920A, RATIONAL)
    if 0x920A in ifd:
        typ, cnt, val_raw = ifd[0x920A]
        if typ == 5:    # RATIONAL
            try:
                off = _read_u32(val_raw, 0, be)
                result["focal_length_mm"] = _read_rational(raw, t + off, be)
            except Exception:
                pass

    # FocalLengthIn35mmFilm (tag 0xA405, SHORT)
    if 0xA405 in ifd:
        typ, cnt, val_raw = ifd[0xA405]
        if typ == 3:    # SHORT
            try:
                result["focal_35mm"] = _read_u16(val_raw, 0, be)
            except Exception:
                pass

    # Image width for GSD formula
    for tag in (0xA002, 0x0100):
        if tag in ifd and "img_width" not in result:
            typ, cnt, val_raw = ifd[tag]
            try:
                result["img_width"] = (
                    _read_u32(val_raw, 0, be) if typ == 4
                    else _read_u16(val_raw, 0, be)
                )
            except Exception:
                pass

    return result


def _estimate_from_exif(
    exif_bytes: Optional[bytes],
    assumed_distance_mm: float = 500.0,
) -> Optional[float]:
    """
    Estimate px_per_mm using focal length from EXIF.

    Uses the thin lens equation and assumes the inspector held the phone
    approximately `assumed_distance_mm` (default 500 mm) from the surface.

    Formula:
        px_per_mm = (f_35 × img_width_px) / (36 mm × distance_mm)

    where 36 mm is the standard 35 mm film frame width and f_35 is the
    35mm-equivalent focal length.
    Returns None if EXIF lacks the required fields.
    """
    if not exif_bytes:
        return None
    try:
        meta = _parse_exif(exif_bytes)
    except Exception:
        return None

    focal_35  = meta.get("focal_35mm")
    focal_raw = meta.get("focal_length_mm")
    img_w_px  = meta.get("img_width", 4000)   # typical modern phone

    if focal_35:
        f = float(focal_35)
    elif focal_raw:
        # Rough conversion assuming 1/2.3" sensor (7.2× crop factor)
        f = float(focal_raw) * 7.2
    else:
        return None

    if f <= 0 or img_w_px <= 0:
        return None

    px_per_mm = (f * img_w_px) / (36.0 * assumed_distance_mm)
    # Sanity check: values outside [2, 300] px/mm are implausible
    if not (2.0 < px_per_mm < 300.0):
        return None

    return px_per_mm


# ── Public API ────────────────────────────────────────────────

def calibrate(
    img_bgr: np.ndarray,
    exif_bytes:        Optional[bytes] = None,
    lidar_distance_m:  Optional[float] = None,
    manual_px_per_mm:  Optional[float] = None,
) -> CalibrationResult:
    """
    Determine px_per_mm by the best available method.

    Priority: manual > lidar > aruco > exif > sdnet_default
    """

    # 1. Manual override (unit tests, CLI, known calibration jig)
    if manual_px_per_mm and manual_px_per_mm > 0:
        return CalibrationResult(
            px_per_mm=manual_px_per_mm,
            method="manual",
            uncertainty=0.02,
            notes="Manual calibration supplied by caller.",
        )

    # 2. LiDAR depth (injected client-side from iOS TrueDepth / iPhone 12 Pro+)
    if lidar_distance_m and lidar_distance_m > 0.05:
        # Use 35mm-equiv wide lens (26 mm) and standard phone sensor (36 mm frame)
        focal_35  = 26.0
        img_w_px  = float(img_bgr.shape[1])
        dist_mm   = lidar_distance_m * 1000.0
        px_per_mm = (focal_35 * img_w_px) / (36.0 * dist_mm)
        return CalibrationResult(
            px_per_mm=px_per_mm,
            method="lidar",
            uncertainty=0.05,
            marker_detected=True,
            notes=f"LiDAR distance: {lidar_distance_m:.3f} m  "
                  f"→ {px_per_mm:.1f} px/mm.",
        )

    # 3. ArUco reference card
    aruco_px = _detect_aruco(img_bgr)
    if aruco_px and aruco_px > 0:
        return CalibrationResult(
            px_per_mm=aruco_px,
            method="aruco",
            uncertainty=0.05,
            marker_detected=True,
            notes=(
                f"ArUco DICT_4X4_50 marker detected. "
                f"Card = {ARUCO_CARD_SIZE_MM} mm → {aruco_px:.2f} px/mm."
            ),
        )

    # 4. EXIF focal length
    exif_px = _estimate_from_exif(exif_bytes)
    if exif_px:
        return CalibrationResult(
            px_per_mm=exif_px,
            method="exif",
            uncertainty=0.15,
            notes=(
                f"Estimated from EXIF focal length → {exif_px:.2f} px/mm. "
                "Assumes 500 mm working distance. Place an ArUco card for "
                "higher accuracy."
            ),
        )

    # 5. SDNET dataset empirical constant
    return CalibrationResult(
        px_per_mm=SDNET_DEFAULT_PX_PER_MM,
        method="default",
        uncertainty=SDNET_DEFAULT_UNCERTAINTY,
        notes=(
            "Using SDNET2018 empirical constant (21.3 px/mm ≈ 0.5 m working "
            "distance). Attach an ArUco card or enable LiDAR for traceable "
            "measurements."
        ),
    )
