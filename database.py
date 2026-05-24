"""
CrackScan — Inspection Log  (Phase 3 upgrade)
=============================================
New columns vs Phase 2:
  max_width_mm          REAL   — EN 206 measured maximum crack width
  mean_width_mm         REAL   — mean width across crack skeleton
  total_length_mm       REAL   — sum of skeleton lengths across all cracks
  total_area_mm2        REAL   — total crack area in mm²
  px_per_mm             REAL   — calibration factor used
  calibration_method    TEXT   — "aruco" | "lidar" | "exif" | "default" | "manual"
  calibration_uncertainty REAL — fractional 1-sigma
  measurement_aggregate TEXT   — JSON: full aggregate dict

Backward compatible: all new columns are added via ALTER TABLE migration
if they are missing (same pattern as the Phase 1 → 2 migration).

Temporal tracking
-----------------
get_crack_growth(lat, lon, radius_m) returns the chronological series of
max_width_mm measurements for inspections near a GPS location, together
with a simple linear trend ("growing" / "stable" / "closing").
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "inspections.db"


# ── Connection ────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema ────────────────────────────────────────────────────

_PHASE3_MIGRATIONS = {
    "max_width_mm":             "REAL    NOT NULL DEFAULT 0.0",
    "mean_width_mm":            "REAL    NOT NULL DEFAULT 0.0",
    "total_length_mm":          "REAL    NOT NULL DEFAULT 0.0",
    "total_area_mm2":           "REAL    NOT NULL DEFAULT 0.0",
    "px_per_mm":                "REAL    NOT NULL DEFAULT 0.0",
    "calibration_method":       "TEXT    NOT NULL DEFAULT 'none'",
    "calibration_uncertainty":  "REAL    NOT NULL DEFAULT 0.0",
    "measurement_aggregate":    "TEXT    NOT NULL DEFAULT '{}'",
}


def init_db() -> None:
    """Create / migrate table. Safe to call repeatedly at startup."""
    with get_connection() as conn:
        # Core table (Phase 1 schema)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT    NOT NULL,
                image_path      TEXT,
                is_cracked      INTEGER NOT NULL,
                confidence      REAL    NOT NULL,
                severity        TEXT    NOT NULL,
                action          TEXT    NOT NULL,
                class_probs     TEXT    NOT NULL,
                infer_ms        REAL    NOT NULL,
                latitude        REAL,
                longitude       REAL,
                location_name   TEXT,
                user_note       TEXT,
                model_ver       TEXT,
                bounding_boxes  TEXT    NOT NULL DEFAULT '[]',
                box_count       INTEGER NOT NULL DEFAULT 0,
                detection_mode  TEXT    NOT NULL DEFAULT 'classification'
            )
        """)

        # Migrate Phase 2 columns
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
        }
        for col, typedef in {
            "bounding_boxes": "TEXT    NOT NULL DEFAULT '[]'",
            "box_count":      "INTEGER NOT NULL DEFAULT 0",
            "detection_mode": "TEXT    NOT NULL DEFAULT 'classification'",
            **_PHASE3_MIGRATIONS,
        }.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE inspections ADD COLUMN {col} {typedef}")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_severity  ON inspections(severity)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created   ON inspections(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mode      ON inspections(detection_mode)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_location  ON inspections(latitude, longitude)")
        conn.commit()


# ── Write ────────────────────────────────────────────────────

def save_inspection(
    is_cracked:              bool,
    confidence:              float,
    severity:                str,
    action:                  str,
    class_probs:             dict,
    infer_ms:                float,
    image_path:              Optional[str]   = None,
    latitude:                Optional[float] = None,
    longitude:               Optional[float] = None,
    location_name:           Optional[str]   = None,
    user_note:               Optional[str]   = None,
    model_ver:               str             = "unknown",
    bounding_boxes:          list            = None,
    detection_mode:          str             = "classification",
    # Phase 3
    measurement_aggregate:   dict            = None,
    calibration_method:      str             = "none",
    calibration_uncertainty: float           = 0.0,
    px_per_mm:               float           = 0.0,
) -> int:
    """Insert an inspection record. Returns the new row ID."""
    boxes = bounding_boxes or []
    agg   = measurement_aggregate or {}

    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO inspections
              (created_at, image_path, is_cracked, confidence, severity,
               action, class_probs, infer_ms,
               latitude, longitude, location_name, user_note, model_ver,
               bounding_boxes, box_count, detection_mode,
               max_width_mm, mean_width_mm, total_length_mm, total_area_mm2,
               px_per_mm, calibration_method, calibration_uncertainty,
               measurement_aggregate)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            image_path,
            int(is_cracked),
            confidence,
            severity,
            action,
            json.dumps(class_probs),
            infer_ms,
            latitude,
            longitude,
            location_name,
            user_note,
            model_ver,
            json.dumps(boxes),
            len(boxes),
            detection_mode,
            agg.get("max_width_mm", 0.0),
            agg.get("mean_width_mm", 0.0) if "mean_width_mm" in agg else 0.0,
            agg.get("total_length_mm", 0.0),
            agg.get("total_area_mm2", 0.0),
            px_per_mm,
            calibration_method,
            calibration_uncertainty,
            json.dumps(agg),
        ))
        conn.commit()
        return cur.lastrowid


# ── Read ─────────────────────────────────────────────────────

def _deserialize(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["class_probs"]           = json.loads(d["class_probs"])
    d["bounding_boxes"]        = json.loads(d.get("bounding_boxes") or "[]")
    d["measurement_aggregate"] = json.loads(d.get("measurement_aggregate") or "{}")
    d["is_cracked"]            = bool(d["is_cracked"])
    return d


def get_inspections(
    limit:    int           = 50,
    offset:   int           = 0,
    severity: Optional[str] = None,
    mode:     Optional[str] = None,
) -> list[dict]:
    """Fetch inspection records, newest first. Supports severity + mode filters."""
    clauses: list[str] = []
    params:  list      = []
    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    if mode:
        clauses.append("detection_mode = ?")
        params.append(mode)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM inspections {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
    return [_deserialize(r) for r in rows]


def get_stats() -> dict:
    """Aggregate statistics for the dashboard."""
    with get_connection() as conn:
        total   = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        cracked = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE is_cracked = 1"
        ).fetchone()[0]
        by_severity = dict(conn.execute(
            "SELECT severity, COUNT(*) FROM inspections GROUP BY severity"
        ).fetchall())
        by_mode = dict(conn.execute(
            "SELECT detection_mode, COUNT(*) FROM inspections GROUP BY detection_mode"
        ).fetchall())
        avg_boxes = conn.execute(
            "SELECT AVG(box_count) FROM inspections WHERE is_cracked = 1"
        ).fetchone()[0]
        # Phase 3 aggregate stats
        max_ever = conn.execute(
            "SELECT MAX(max_width_mm) FROM inspections"
        ).fetchone()[0] or 0.0
        avg_width = conn.execute(
            "SELECT AVG(max_width_mm) FROM inspections WHERE is_cracked = 1"
        ).fetchone()[0] or 0.0

    return {
        "total":                  total,
        "cracked":                cracked,
        "uncracked":              total - cracked,
        "by_severity":            by_severity,
        "by_mode":                by_mode,
        "crack_rate_pct":         round(100 * cracked / total, 1) if total else 0,
        "avg_boxes_per_crack":    round(avg_boxes or 0, 1),
        "max_recorded_width_mm":  round(max_ever, 3),
        "avg_crack_width_mm":     round(avg_width, 3),
    }


def delete_inspection(inspection_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM inspections WHERE id = ?", (inspection_id,)
        )
        conn.commit()
        return cur.rowcount > 0


# ── Temporal tracking ─────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two GPS coordinates."""
    R = 6_371_000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ/2)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_inspections_near(
    lat: float,
    lon: float,
    radius_m: float = 50.0,
    limit: int = 200,
) -> list[dict]:
    """
    Return inspections within `radius_m` metres of the given GPS point.
    SQLite lacks Haversine, so we do a bounding-box pre-filter then
    exact distance check in Python (fast enough at inspection scales).
    """
    # 1° latitude ≈ 111 km; 1° longitude ≈ 111 km × cos(lat)
    deg_lat = radius_m / 111_000.0
    deg_lon = radius_m / (111_000.0 * max(math.cos(math.radians(lat)), 0.001))

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM inspections
            WHERE latitude  BETWEEN ? AND ?
              AND longitude BETWEEN ? AND ?
            ORDER BY created_at ASC
            LIMIT ?
        """, (lat - deg_lat, lat + deg_lat,
               lon - deg_lon, lon + deg_lon,
               limit)).fetchall()

    result = []
    for row in rows:
        d = _deserialize(row)
        if d["latitude"] and d["longitude"]:
            dist = _haversine_m(lat, lon, d["latitude"], d["longitude"])
            if dist <= radius_m:
                d["distance_m"] = round(dist, 1)
                result.append(d)
    return result


def get_crack_growth(
    lat: float,
    lon: float,
    radius_m: float = 50.0,
) -> dict:
    """
    Chronological crack width measurements for a GPS location.

    Returns a time-series of max_width_mm values and a simple linear trend
    assessment so the inspector can see whether the crack is growing.

    Trend logic:
      width_last − width_first > +0.05 mm → "growing"
      width_last − width_first < −0.05 mm → "closing"
      otherwise                           → "stable"
    Only inspections with a Phase 3 measurement (max_width_mm > 0) and
    a reliable calibration method (aruco / lidar / manual) are included
    in the trend calculation to avoid noise from the default fallback.
    """
    nearby = get_inspections_near(lat, lon, radius_m)

    series = []
    for insp in nearby:
        agg = insp.get("measurement_aggregate") or {}
        series.append({
            "inspection_id":      insp["id"],
            "created_at":         insp["created_at"],
            "max_width_mm":       insp.get("max_width_mm", 0.0),
            "total_length_mm":    insp.get("total_length_mm", 0.0),
            "severity":           insp["severity"],
            "calibration_method": insp.get("calibration_method", "none"),
            "detection_mode":     insp.get("detection_mode", "unknown"),
            "distance_m":         insp.get("distance_m"),
        })

    # Trend: use only reliable calibration points
    reliable = [
        s for s in series
        if s["max_width_mm"] > 0
        and s["calibration_method"] in ("aruco", "lidar", "manual")
    ]

    trend: Optional[str] = None
    delta_mm: Optional[float] = None
    if len(reliable) >= 2:
        delta_mm = reliable[-1]["max_width_mm"] - reliable[0]["max_width_mm"]
        trend = (
            "growing" if delta_mm >  0.05 else
            "closing" if delta_mm < -0.05 else
            "stable"
        )

    return {
        "count":           len(series),
        "series":          series,
        "trend":           trend,
        "delta_mm":        round(delta_mm, 3) if delta_mm is not None else None,
        "reliable_points": len(reliable),
        "radius_m":        radius_m,
    }
