import json
import os
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

DB_PATH = Path(
    os.environ.get(
        "SPEECHSENSE_DB_PATH",
        str(Path(__file__).resolve().parent / "speechsense.db"),
    )
).resolve()

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                predicted_emotion TEXT NOT NULL,
                probabilities TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                audio_path TEXT,
                created_at TEXT NOT NULL,
                model_version TEXT,
                processing_time_ms REAL,
                is_uncertain INTEGER
            )
        """)
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        migrations = {
            "model_version": "TEXT",
            "processing_time_ms": "REAL",
            "is_uncertain": "INTEGER",
        }
        for column, sql_type in migrations.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {column} {sql_type}")
        conn.commit()

def save_prediction(
    prediction_id: str,
    filename: str,
    predicted_emotion: str,
    probabilities: dict,
    duration_seconds: float,
    file_size_bytes: int,
    audio_path: Optional[str],
    created_at: str,
    model_version: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
    is_uncertain: Optional[bool] = None,
) -> None:
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions (
                id, filename, predicted_emotion, probabilities,
                duration_seconds, file_size_bytes, audio_path, created_at,
                model_version, processing_time_ms, is_uncertain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id,
            filename,
            predicted_emotion,
            json.dumps(probabilities),
            duration_seconds,
            file_size_bytes,
            audio_path,
            created_at,
            model_version,
            processing_time_ms,
            None if is_uncertain is None else int(is_uncertain),
        ))
        conn.commit()

def get_predictions(
    search: Optional[str] = None,
    emotion: Optional[str] = None,
    page: int = 1,
    per_page: int = 20
) -> Tuple[List[Dict[str, Any]], int]:
    query = "SELECT * FROM predictions WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM predictions WHERE 1=1"
    params: List[Any] = []

    if search and search.strip():
        term = f"%{search.strip().lower()}%"
        query += " AND LOWER(filename) LIKE ?"
        count_query += " AND LOWER(filename) LIKE ?"
        params.append(term)

    if emotion and emotion.strip():
        em = emotion.strip().lower()
        query += " AND LOWER(predicted_emotion) = ?"
        count_query += " AND LOWER(predicted_emotion) = ?"
        params.append(em)

    query += " ORDER BY datetime(created_at) DESC LIMIT ? OFFSET ?"
    offset = (page - 1) * per_page
    pagination_params = params + [per_page, offset]

    with get_connection() as conn:
        total = conn.execute(count_query, params).fetchone()[0]
        rows = conn.execute(query, pagination_params).fetchall()

    results = []
    for r in rows:
        row_dict = dict(r)
        try:
            row_dict["probabilities"] = json.loads(row_dict["probabilities"])
        except Exception:
            pass
        results.append(row_dict)

    return results, total

def get_prediction_by_id(prediction_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["probabilities"] = json.loads(res["probabilities"])
        except Exception:
            pass
        return res


def get_prediction_stats() -> Dict[str, Any]:
    with get_connection() as conn:
        summary = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(AVG(duration_seconds), 0) AS average_duration_seconds,
                COALESCE(SUM(CASE WHEN is_uncertain = 1 THEN 1 ELSE 0 END), 0)
                    AS uncertain_count,
                COALESCE(SUM(
                    CASE WHEN datetime(created_at) >= datetime('now', '-7 days')
                    THEN 1 ELSE 0 END
                ), 0) AS recent_count
            FROM predictions
            """
        ).fetchone()
        emotion_rows = conn.execute(
            """
            SELECT LOWER(predicted_emotion) AS emotion, COUNT(*) AS count
            FROM predictions
            GROUP BY LOWER(predicted_emotion)
            ORDER BY count DESC, emotion ASC
            """
        ).fetchall()
        model_rows = conn.execute(
            """
            SELECT COALESCE(model_version, 'historical-ravdess-8class-v1') AS version,
                   COUNT(*) AS count
            FROM predictions
            GROUP BY COALESCE(model_version, 'historical-ravdess-8class-v1')
            ORDER BY count DESC, version ASC
            """
        ).fetchall()

    return {
        "total": int(summary["total"]),
        "recent_count": int(summary["recent_count"]),
        "average_duration_seconds": round(
            float(summary["average_duration_seconds"]), 2
        ),
        "uncertain_count": int(summary["uncertain_count"]),
        "emotion_distribution": {
            str(row["emotion"]): int(row["count"]) for row in emotion_rows
        },
        "model_versions": {
            str(row["version"]): int(row["count"]) for row in model_rows
        },
    }


def delete_prediction(prediction_id: str) -> Optional[Dict[str, Any]]:
    """Delete one prediction and return its former database row."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        conn.commit()
        return dict(row)

def delete_all_predictions() -> int:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.execute("DELETE FROM predictions")
        conn.commit()
        return total
