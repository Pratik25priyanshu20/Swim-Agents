# swim/shared/database/connection.py

"""SQLite-backed persistent storage for agent task history and memory."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from swim.shared.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "swim.db"


class SWIMDatabase:
    """Lightweight SQLite database for agent memory persistence."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                status TEXT NOT NULL,
                query TEXT,
                result TEXT,
                duration_seconds REAL,
                trace_id TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT UNIQUE,
                location_name TEXT,
                latitude REAL,
                longitude REAL,
                risk_level TEXT,
                risk_score REAL,
                errors TEXT,
                total_time_seconds REAL,
                result_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(agent, key)
            );

            CREATE INDEX IF NOT EXISTS idx_task_agent ON task_history(agent);
            CREATE INDEX IF NOT EXISTS idx_pipeline_location ON pipeline_runs(location_name);
        """)
        conn.commit()

    def record_task(
        self,
        task_id: str,
        agent: str,
        status: str,
        query: str = "",
        result: str = "",
        duration: float = 0.0,
        trace_id: str = "",
    ):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO task_history (task_id, agent, status, query, result, duration_seconds, trace_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, agent, status, query[:2000], result[:5000], duration, trace_id),
        )
        conn.commit()

    def record_pipeline_run(self, result: Dict[str, Any]):
        conn = self._get_conn()
        meta = result.get("metadata", {})
        risk = result.get("risk_assessment", {})
        exec_sum = result.get("execution_summary", {})
        location = meta.get("location", {})
        conn.execute(
            "INSERT OR REPLACE INTO pipeline_runs "
            "(trace_id, location_name, latitude, longitude, risk_level, risk_score, errors, total_time_seconds, result_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exec_sum.get("trace_id", ""),
                location.get("name", ""),
                location.get("latitude"),
                location.get("longitude"),
                risk.get("level", ""),
                risk.get("score", 0),
                json.dumps(exec_sum.get("errors", [])),
                exec_sum.get("time", 0),
                json.dumps(result, default=str)[:50000],
            ),
        )
        conn.commit()

    def get_recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def set_memory(self, agent: str, key: str, value: str):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO agent_memory (agent, key, value, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (agent, key, value, datetime.now().isoformat()),
        )
        conn.commit()

    def get_memory(self, agent: str, key: str) -> Optional[str]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM agent_memory WHERE agent = ? AND key = ?", (agent, key)
        ).fetchone()
        return row["value"] if row else None

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# Global singleton
db = SWIMDatabase()
