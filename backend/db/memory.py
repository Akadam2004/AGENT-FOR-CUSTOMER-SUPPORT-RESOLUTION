"""
SQLite storage for support ticket history (memory.db).
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

_DB_PATH = Path(__file__).resolve().parent / "memory.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create database file and tickets table if they do not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                category TEXT,
                decision TEXT,
                confidence REAL,
                "timestamp" TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()


def save_ticket(
    user_id: str,
    message: str,
    category: str,
    decision: str,
    confidence: float,
) -> int:
    """
    Persist a ticket row. Returns the new row id.
    """
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO tickets (user_id, message, category, decision, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, message, category, decision, float(confidence)),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_user_history(user_id: str) -> List[Dict[str, Any]]:
    """
    Return all tickets for a user, newest first.
    """
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT id, user_id, message, category, decision, confidence, "timestamp"
            FROM tickets
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        )
        rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}
