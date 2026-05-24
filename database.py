"""
Inspection Log — SQLite database layer
Stores every inspection: image path, detection result, GPS, timestamp.
Uses aiosqlite for async access compatible with FastAPI.
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
    """Create tables if they don't exist. Called at startup."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                image_path  TEXT,
                is_cracked  INTEGER NOT NULL,
                confidence  REAL    NOT NULL,
                severity    TEXT    NOT NULL,
                action      TEXT    NOT NULL,
                class_probs TEXT    NOT NULL,
                infer_ms    REAL    NOT NULL,
                latitude    REAL,
                longitude   REAL,
                location_name TEXT,
                user_note   TEXT,
                model_ver   TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_severity ON inspections(severity)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created ON inspections(created_at)
        """)
        conn.commit()


def save_inspection(
    is_cracked: bool,
    confidence: float,
    severity: str,
    action: str,
    class_probs: dict,
    infer_ms: float,
    image_path: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
    user_note: Optional[str] = None,
    model_ver: str = "unknown",
) -> int:
    """Insert an inspection record. Returns the new row ID."""
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO inspections
              (created_at, image_path, is_cracked, confidence, severity,
               action, class_probs, infer_ms,
               latitude, longitude, location_name, user_note, model_ver)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
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
        ))
        conn.commit()
        return cur.lastrowid


def get_inspections(
    limit: int = 50,
    offset: int = 0,
    severity: Optional[str] = None,
) -> list[dict]:
    """Fetch inspection records, newest first."""
    with get_connection() as conn:
        if severity:
            rows = conn.execute("""
                SELECT * FROM inspections
                WHERE severity = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (severity, limit, offset)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM inspections
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d["class_probs"] = json.loads(d["class_probs"])
        d["is_cracked"] = bool(d["is_cracked"])
        result.append(d)
    return result


def get_stats() -> dict:
    """Summary statistics for the dashboard."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        cracked = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE is_cracked = 1"
        ).fetchone()[0]
        by_severity = dict(conn.execute("""
            SELECT severity, COUNT(*) FROM inspections GROUP BY severity
        """).fetchall())

    return {
        "total": total,
        "cracked": cracked,
        "uncracked": total - cracked,
        "by_severity": by_severity,
        "crack_rate_pct": round(100 * cracked / total, 1) if total else 0,
    }


def delete_inspection(inspection_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM inspections WHERE id = ?", (inspection_id,)
        )
        conn.commit()
        return cur.rowcount > 0
