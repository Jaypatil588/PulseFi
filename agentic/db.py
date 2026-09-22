"""SQLite persistence for observations and agent memories."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agentic.llm import AgentDecision
from agentic.models import Reading


PLAYBOOK = [
    "ingest real inference row",
    "load bounded working, episodic, and semantic memory",
    "call the configured language model",
    "validate structured model output",
    "persist decision and memory",
]


class MemoryDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY,
                ts_us INTEGER NOT NULL,
                human INTEGER NOT NULL,
                presence REAL NOT NULL,
                bpm REAL,
                sensor_bpm REAL,
                sensor_valid INTEGER NOT NULL,
                source TEXT NOT NULL,
                UNIQUE(ts_us, source)
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY,
                observation_id INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL,
                trend TEXT NOT NULL,
                prediction TEXT NOT NULL,
                feedback TEXT NOT NULL,
                suggestion TEXT NOT NULL,
                confidence REAL NOT NULL,
                memory_summary TEXT NOT NULL,
                model TEXT NOT NULL,
                created_us INTEGER NOT NULL,
                FOREIGN KEY(observation_id) REFERENCES observations(id)
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY,
                opened_us INTEGER NOT NULL,
                closed_us INTEGER,
                status TEXT NOT NULL,
                summary TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS semantic_memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_us INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS procedural_memory (
                name TEXT PRIMARY KEY,
                steps_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_memory (
                name TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                description TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stream_cursor (
                path TEXT PRIMARY KEY,
                offset INTEGER NOT NULL
            );
            """
        )
        self.conn.execute(
            """
            INSERT INTO procedural_memory(name, steps_json) VALUES('agent_tick', ?)
            ON CONFLICT(name) DO UPDATE SET steps_json=excluded.steps_json
            """,
            (json.dumps(PLAYBOOK),),
        )
        self.conn.execute(
            """
            INSERT INTO model_memory(name, kind, description)
            VALUES('upstream_lstm', 'parametric', ?)
            ON CONFLICT(name) DO UPDATE SET
                kind=excluded.kind, description=excluded.description
            """,
            (
                "The upstream trained LSTM weights encode CSI patterns for "
                "presence and heart-rate inference. This worker consumes its "
                "CSV outputs; it does not claim the weights are present.",
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def observation_exists(self, reading: Reading) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM observations WHERE ts_us=? AND source=?",
            (reading.ts_us, reading.source),
        ).fetchone()
        return row is not None

    def recent_readings(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT ts_us, human, presence, bpm, sensor_bpm, sensor_valid, source
            FROM observations ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        result = [dict(row) for row in rows]
        result.reverse()
        return result

    def recent_episodes(self, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def semantic_memory(self) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM semantic_memory ORDER BY key"
        ).fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def model_memory(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT name, kind, description FROM model_memory ORDER BY name"
        ).fetchall()
        return [dict(row) for row in rows]

    def latest_decision(self) -> dict | None:
        row = self.conn.execute(
            """
            SELECT d.*, o.ts_us, o.bpm, o.source
            FROM decisions d JOIN observations o ON o.id=d.observation_id
            ORDER BY d.id DESC LIMIT 1
            """
        ).fetchone()
        return dict(row) if row else None

    def _update_episode(self, decision: AgentDecision, ts_us: int) -> None:
        open_row = self.conn.execute(
            "SELECT * FROM episodes WHERE closed_us IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        concerning = decision.status in {"watch", "urgent"}
        if concerning and open_row is None:
            self.conn.execute(
                """
                INSERT INTO episodes(opened_us, status, summary)
                VALUES(?, ?, ?)
                """,
                (ts_us, decision.status, decision.memory_summary),
            )
        elif concerning and open_row is not None:
            self.conn.execute(
                "UPDATE episodes SET status=?, summary=? WHERE id=?",
                (decision.status, decision.memory_summary, open_row["id"]),
            )
        elif not concerning and open_row is not None:
            self.conn.execute(
                "UPDATE episodes SET closed_us=? WHERE id=?",
                (ts_us, open_row["id"]),
            )

    def commit_tick(
        self,
        reading: Reading,
        decision: AgentDecision,
        model: str,
        created_us: int,
    ) -> None:
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO observations
                    (ts_us, human, presence, bpm, sensor_bpm, sensor_valid, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reading.ts_us,
                    int(reading.human),
                    reading.presence,
                    reading.bpm,
                    reading.sensor_bpm,
                    int(reading.sensor_valid),
                    reading.source,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO decisions
                    (observation_id, status, trend, prediction, feedback,
                     suggestion, confidence, memory_summary, model, created_us)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cursor.lastrowid,
                    decision.status,
                    decision.trend,
                    decision.prediction,
                    decision.feedback,
                    decision.suggestion,
                    decision.confidence,
                    decision.memory_summary,
                    model,
                    created_us,
                ),
            )
            self._update_episode(decision, reading.ts_us)
            self.conn.execute(
                """
                INSERT INTO semantic_memory(key, value, updated_us)
                VALUES('latest_agent_memory', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value, updated_us=excluded.updated_us
                """,
                (decision.memory_summary, reading.ts_us),
            )

    def cursor_get(self, path: str) -> int:
        row = self.conn.execute(
            "SELECT offset FROM stream_cursor WHERE path=?", (path,)
        ).fetchone()
        return int(row["offset"]) if row else 0

    def cursor_set(self, path: str, offset: int) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO stream_cursor(path, offset) VALUES(?, ?)
                ON CONFLICT(path) DO UPDATE SET offset=excluded.offset
                """,
                (path, offset),
            )
