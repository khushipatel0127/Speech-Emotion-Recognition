import json
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

DB_PATH = Path(__file__).resolve().parent / "speechsense.db"

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
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
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

def save_prediction(
    prediction_id: str,
    filename: str,
    predicted_emotion: str,
    probabilities: dict,
    duration_seconds: float,
    file_size_bytes: int,
    audio_path: Optional[str],
    created_at: str
) -> None:
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions (
                id, filename, predicted_emotion, probabilities,
                duration_seconds, file_size_bytes, audio_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id,
            filename,
            predicted_emotion,
            json.dumps(probabilities),
            duration_seconds,
            file_size_bytes,
            audio_path,
            created_at
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

def delete_all_predictions() -> int:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.execute("DELETE FROM predictions")
        conn.commit()
        return total
