
import json
import sqlite3
from datetime import datetime
from pathlib import Path


# Thư mục gốc của dự án AI_Traffic_Vision_Web
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "history.db"


def get_connection():
    """Tạo kết nối SQLite và bảo đảm thư mục data tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Khởi tạo bảng lưu lịch sử phân tích video."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                video_name TEXT NOT NULL,
                source_video_path TEXT,
                result_video_path TEXT,
                motorbike INTEGER NOT NULL DEFAULT 0,
                helmet INTEGER NOT NULL DEFAULT 0,
                no_helmet INTEGER NOT NULL DEFAULT 0,
                avg_conf REAL NOT NULL DEFAULT 0,
                elapsed REAL NOT NULL DEFAULT 0,
                stats_json TEXT,
                config_json TEXT,
                recent_detections_json TEXT
            )
            """
        )
        conn.commit()


def add_history(
    video_name,
    source_video_path,
    result_video_path,
    stats,
    config=None,
    recent_detections=None,
    elapsed=0.0,
):
    """Lưu một lần phân tích video và trả về ID bản ghi."""

    stats = stats or {}
    config = config or {}
    recent_detections = recent_detections or []

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO detection_history (
                created_at,
                video_name,
                source_video_path,
                result_video_path,
                motorbike,
                helmet,
                no_helmet,
                avg_conf,
                elapsed,
                stats_json,
                config_json,
                recent_detections_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(video_name or "video"),
                str(source_video_path or ""),
                str(result_video_path or ""),
                int(stats.get("motorbike", 0)),
                int(stats.get("helmet", 0)),
                int(stats.get("no_helmet", 0)),
                float(stats.get("avg_conf", 0.0)),
                float(elapsed or 0.0),
                json.dumps(stats, ensure_ascii=False),
                json.dumps(config, ensure_ascii=False, default=str),
                json.dumps(recent_detections, ensure_ascii=False, default=str),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_history(limit=100):
    """Lấy danh sách lịch sử mới nhất."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM detection_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    return [dict(row) for row in rows]


def get_history_by_id(history_id):
    """Lấy đầy đủ thông tin của một lần phân tích."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM detection_history
            WHERE id = ?
            """,
            (int(history_id),),
        ).fetchone()

    if row is None:
        return None

    item = dict(row)

    for column, key in (
        ("stats_json", "stats"),
        ("config_json", "config"),
        ("recent_detections_json", "recent_detections"),
    ):
        try:
            item[key] = json.loads(item.get(column) or "{}")
        except (TypeError, json.JSONDecodeError):
            item[key] = {}

    return item


def delete_history(history_id):
    """Xóa một bản ghi lịch sử, không xóa file video."""
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM detection_history WHERE id = ?",
            (int(history_id),),
        )
        conn.commit()
        return cursor.rowcount > 0
