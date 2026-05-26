"""
Inspection Log — SQLite database layer  (Phase 3 upgrade)
==========================================================
New columns vs Phase 2:
  scale_mm_per_px  REAL  — Ground Sampling Distance used for width measurement

Phase 2 columns:
  bounding_boxes  TEXT  — JSON array of box dicts [{x1,y1,x2,y2,confidence,severity,crack_width_px,crack_width_mm}]
  box_count       INT   — pre-computed length of bounding_boxes list
  detection_mode  TEXT  — "detection" | "classification" | "mock"

Backward compatible: earlier rows without new columns get defaults.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "inspections.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create / migrate tables. Called at startup — safe to run repeatedly."""
    with get_connection() as conn:
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
                detection_mode  TEXT    NOT NULL DEFAULT 'classification',
                scale_mm_per_px REAL
            )
        """)

        # ── Migrate Phase 1 DBs that are missing the new columns ──────────
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
        }
        migrations = {
            "bounding_boxes":  "TEXT NOT NULL DEFAULT '[]'",
            "box_count":       "INTEGER NOT NULL DEFAULT 0",
            "detection_mode":  "TEXT NOT NULL DEFAULT 'classification'",
            "scale_mm_per_px": "REAL",
        }
        for col, typedef in migrations.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE inspections ADD COLUMN {col} {typedef}")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON inspections(severity)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created  ON inspections(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mode     ON inspections(detection_mode)")
        conn.commit()


def save_inspection(
    is_cracked:     bool,
    confidence:     float,
    severity:       str,
    action:         str,
    class_probs:    dict,
    infer_ms:       float,
    image_path:     Optional[str]  = None,
    latitude:       Optional[float] = None,
    longitude:      Optional[float] = None,
    location_name:  Optional[str]  = None,
    user_note:      Optional[str]  = None,
    model_ver:      str            = "unknown",
    bounding_boxes: list           = None,
    detection_mode: str            = "classification",
    scale_mm_per_px: Optional[float] = None,
) -> int:
    """Insert an inspection record. Returns the new row ID."""
    boxes = bounding_boxes or []
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO inspections
              (created_at, image_path, is_cracked, confidence, severity,
               action, class_probs, infer_ms,
               latitude, longitude, location_name, user_note, model_ver,
               bounding_boxes, box_count, detection_mode, scale_mm_per_px)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            scale_mm_per_px,
        ))
        conn.commit()
        return cur.lastrowid


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
        rows = conn.execute(f"""
            SELECT * FROM inspections
            {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (*params, limit, offset)).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d["class_probs"]    = json.loads(d["class_probs"])
        d["bounding_boxes"] = json.loads(d.get("bounding_boxes") or "[]")
        d["is_cracked"]     = bool(d["is_cracked"])
        result.append(d)
    return result


def get_stats() -> dict:
    """Summary statistics for the dashboard."""
    with get_connection() as conn:
        total   = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        cracked = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE is_cracked = 1"
        ).fetchone()[0]
        by_severity = dict(conn.execute("""
            SELECT severity, COUNT(*) FROM inspections GROUP BY severity
        """).fetchall())
        by_mode = dict(conn.execute("""
            SELECT detection_mode, COUNT(*) FROM inspections GROUP BY detection_mode
        """).fetchall())
        avg_boxes = conn.execute(
            "SELECT AVG(box_count) FROM inspections WHERE is_cracked = 1"
        ).fetchone()[0]

    return {
        "total":           total,
        "cracked":         cracked,
        "uncracked":       total - cracked,
        "by_severity":     by_severity,
        "by_mode":         by_mode,
        "crack_rate_pct":  round(100 * cracked / total, 1) if total else 0,
        "avg_boxes_per_crack": round(avg_boxes or 0, 1),
    }


def delete_inspection(inspection_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM inspections WHERE id = ?", (inspection_id,)
        )
        conn.commit()
        return cur.rowcount > 0
