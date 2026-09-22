"""SQLite persistence for observations and agent memories."""

from __future__ import annotations

import json
import sqlite3
from math import sqrt
from pathlib import Path

from agentic.llm import AgentResponse
from agentic.models import Reading
from agentic.policy import GuardResult


PLAYBOOK = [
    "ingest and validate one inference row",
    "compute deterministic signal features and guard state",
    "load bounded working, episodic, and semantic memory",
    "call Groq for an advisory structured interpretation",
    "validate structured model output",
    "merge guard and model state without allowing a guard downgrade",
    "persist observation, features, decision, episode, trace, and message atomically",
    "surface abnormality messages in the dashboard until acknowledged",
]


class MemoryDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
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
            CREATE TABLE IF NOT EXISTS features (
                observation_id INTEGER PRIMARY KEY,
                guard_state TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                features_json TEXT NOT NULL,
                measurement_valid INTEGER NOT NULL,
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
            CREATE TABLE IF NOT EXISTS stream_events (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                offset INTEGER NOT NULL,
                next_offset INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                detail TEXT NOT NULL,
                row_sha256 TEXT,
                created_us INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                ts_us INTEGER NOT NULL,
                source TEXT NOT NULL,
                bpm REAL,
                model TEXT NOT NULL,
                context_bytes INTEGER NOT NULL,
                working_memory_count INTEGER NOT NULL,
                episode_count INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                outcome TEXT NOT NULL,
                status TEXT,
                confidence REAL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY,
                observation_id INTEGER NOT NULL,
                episode_id INTEGER,
                action_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                created_us INTEGER NOT NULL,
                acknowledged_us INTEGER,
                delivery_status TEXT NOT NULL DEFAULT 'in_app',
                external_delivery_status TEXT NOT NULL DEFAULT 'not_configured',
                FOREIGN KEY(observation_id) REFERENCES observations(id),
                FOREIGN KEY(episode_id) REFERENCES episodes(id)
            );
            CREATE TABLE IF NOT EXISTS delivery_attempts (
                id INTEGER PRIMARY KEY,
                action_id INTEGER NOT NULL,
                channel TEXT NOT NULL,
                attempted_us INTEGER NOT NULL,
                status TEXT NOT NULL,
                detail TEXT,
                FOREIGN KEY(action_id) REFERENCES actions(id)
            );
            CREATE TABLE IF NOT EXISTS integration_outbox (
                id INTEGER PRIMARY KEY,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_us INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_us INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                delivered_us INTEGER
            );
            """
        )
        self._migrate()
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

    def _columns(self, table: str) -> set[str]:
        return {
            str(row["name"])
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _ensure_column(self, table: str, definition: str) -> None:
        name = definition.split()[0]
        if name not in self._columns(table):
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    def _migrate(self) -> None:
        self._ensure_column(
            "observations", "activity TEXT NOT NULL DEFAULT 'unknown'"
        )
        self._ensure_column(
            "observations", "symptoms_json TEXT NOT NULL DEFAULT '[]'"
        )
        self._ensure_column("decisions", "llm_state TEXT")
        self._ensure_column("decisions", "final_state TEXT")
        self._ensure_column("decisions", "evidence_json TEXT")
        self._ensure_column(
            "decisions", "reasoning_available INTEGER NOT NULL DEFAULT 1"
        )
        self._ensure_column(
            "decisions", "reasoning_status TEXT NOT NULL DEFAULT 'success'"
        )
        self._ensure_column(
            "decisions", "prompt_tokens INTEGER NOT NULL DEFAULT 0"
        )
        self._ensure_column(
            "decisions", "completion_tokens INTEGER NOT NULL DEFAULT 0"
        )
        self._ensure_column(
            "decisions", "total_tokens INTEGER NOT NULL DEFAULT 0"
        )
        self._ensure_column("decisions", "request_id TEXT")
        self._ensure_column("decisions", "system_fingerprint TEXT")
        self._ensure_column("episodes", "last_update_us INTEGER")
        self._ensure_column("episodes", "peak_bpm REAL")
        self._ensure_column("episodes", "max_severity TEXT")
        self._ensure_column(
            "episodes", "evidence_json TEXT NOT NULL DEFAULT '[]'"
        )
        self._ensure_column("episodes", "acknowledged_us INTEGER")
        self._ensure_column("episodes", "outcome TEXT")
        self._ensure_column(
            "actions",
            "external_delivery_status TEXT NOT NULL DEFAULT 'not_configured'",
        )
        self._ensure_column(
            "integration_outbox",
            "next_attempt_us INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column("traces", "guard_state TEXT")
        self._ensure_column("traces", "final_state TEXT")
        self._ensure_column(
            "traces", "prompt_tokens INTEGER NOT NULL DEFAULT 0"
        )
        self._ensure_column(
            "traces", "completion_tokens INTEGER NOT NULL DEFAULT 0"
        )
        self._ensure_column(
            "traces", "total_tokens INTEGER NOT NULL DEFAULT 0"
        )
        self._ensure_column("traces", "request_id TEXT")

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
            SELECT ts_us, human, presence, bpm, sensor_bpm, sensor_valid,
                   source, activity, symptoms_json
            FROM observations ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["symptoms"] = json.loads(value.pop("symptoms_json"))
            result.append(value)
        result.reverse()
        return result

    def recent_reading_models(self, limit: int = 60) -> list[Reading]:
        return [
            Reading(
                ts_us=int(value["ts_us"]),
                human=bool(value["human"]),
                presence=float(value["presence"]),
                bpm=None if value["bpm"] is None else float(value["bpm"]),
                sensor_bpm=(
                    None
                    if value["sensor_bpm"] is None
                    else float(value["sensor_bpm"])
                ),
                sensor_valid=bool(value["sensor_valid"]),
                source=str(value["source"]),
                activity=str(value["activity"]),
                symptoms=tuple(value["symptoms"]),
            )
            for value in self.recent_readings(limit)
        ]

    def recent_episodes(self, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def similar_episodes(
        self, bpm: float | None, limit: int = 3
    ) -> list[dict]:
        if bpm is None:
            return []
        rows = self.conn.execute(
            """
            SELECT *, ABS(peak_bpm - ?) AS bpm_distance
            FROM episodes
            WHERE closed_us IS NOT NULL AND peak_bpm IS NOT NULL
            ORDER BY bpm_distance, id DESC
            LIMIT ?
            """,
            (bpm, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def active_episode(self) -> dict | None:
        row = self.conn.execute(
            """
            SELECT * FROM episodes
            WHERE closed_us IS NULL
            ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
        return dict(row) if row else None

    def resting_baseline(self) -> float | None:
        raw = self.semantic_memory().get("resting_bpm")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

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

    def last_successful_reasoning_ts_us(self) -> int | None:
        row = self.conn.execute(
            """
            SELECT o.ts_us
            FROM decisions d
            JOIN observations o ON o.id=d.observation_id
            WHERE d.reasoning_available=1
            ORDER BY d.id DESC LIMIT 1
            """
        ).fetchone()
        return None if row is None else int(row["ts_us"])

    @staticmethod
    def _decode_json_fields(value: dict, fields: tuple[str, ...]) -> dict:
        result = dict(value)
        for field in fields:
            raw = result.get(field)
            if isinstance(raw, str):
                try:
                    result[field] = json.loads(raw)
                except json.JSONDecodeError:
                    result[field] = []
        return result

    def dashboard_snapshot(self) -> dict:
        latest = self.conn.execute(
            """
            SELECT
                o.id AS observation_id, o.ts_us, o.human, o.presence, o.bpm,
                o.sensor_bpm, o.sensor_valid, o.source, o.activity,
                o.symptoms_json,
                f.guard_state, f.reasons_json, f.features_json,
                f.measurement_valid,
                d.llm_state, d.final_state, d.trend, d.prediction AS forecast,
                d.evidence_json, d.feedback, d.suggestion, d.confidence,
                d.memory_summary AS memory_update, d.model, d.created_us,
                d.reasoning_available, d.prompt_tokens, d.completion_tokens,
                d.total_tokens, d.request_id, d.reasoning_status
            FROM observations o
            JOIN features f ON f.observation_id=o.id
            JOIN decisions d ON d.observation_id=o.id
            ORDER BY o.id DESC LIMIT 1
            """
        ).fetchone()
        active = self.active_episode()
        actions = self.recent_actions(limit=20)
        episodes = self.recent_episodes(limit=20)
        return {
            "latest": (
                None
                if latest is None
                else self._decode_json_fields(
                    dict(latest),
                    (
                        "symptoms_json",
                        "reasons_json",
                        "features_json",
                        "evidence_json",
                    ),
                )
            ),
            "active_episode": (
                None
                if active is None
                else self._decode_json_fields(active, ("evidence_json",))
            ),
            "actions": actions,
            "episodes": [
                self._decode_json_fields(item, ("evidence_json",))
                for item in episodes
            ],
            "semantic_memory": self.semantic_memory(),
            "integration_events": self.recent_integration_events(limit=20),
            "delivery_attempts": self.recent_delivery_attempts(limit=20),
            "stream_events": self.recent_stream_events(limit=20),
        }

    def recent_actions(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT * FROM actions ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_integration_events(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT id, event_type, created_us, status, attempt_count,
                   last_error, delivered_us
            FROM integration_outbox ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_stream_events(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT id, path, offset, next_offset, event_type, detail,
                   row_sha256, created_us
            FROM stream_events ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_delivery_attempts(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT id, action_id, channel, attempted_us, status, detail
            FROM delivery_attempts ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def dashboard_history(self, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT o.ts_us, o.bpm, o.presence, o.source,
                   f.guard_state, d.final_state, d.confidence
            FROM observations o
            JOIN features f ON f.observation_id=o.id
            JOIN decisions d ON d.observation_id=o.id
            ORDER BY o.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        values = [dict(row) for row in rows]
        values.reverse()
        return values

    def acknowledge_action(self, action_id: int, ts_us: int) -> bool:
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE actions SET acknowledged_us=?
                WHERE id=? AND acknowledged_us IS NULL
                """,
                (ts_us, action_id),
            )
            if cursor.rowcount == 1:
                self.conn.execute(
                    """
                    UPDATE episodes SET acknowledged_us=?
                    WHERE id=(
                        SELECT episode_id FROM actions WHERE id=?
                    )
                    """,
                    (ts_us, action_id),
                )
        return cursor.rowcount == 1

    def _update_episode(
        self,
        *,
        final_state: str,
        response: AgentResponse,
        guard: GuardResult,
        reading: Reading,
    ) -> dict | None:
        decision = response.decision
        open_row = self.conn.execute(
            "SELECT * FROM episodes WHERE closed_us IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        summary = (
            decision.memory_update
            if response.reasoning_available
            else (
                str(open_row["summary"])
                if open_row is not None
                else "Deterministic guard event; no LLM interpretation."
            )
        )
        evidence = list(dict.fromkeys([*guard.reasons, *decision.evidence]))
        evidence_json = json.dumps(evidence)
        if final_state in {"watch", "urgent"} and open_row is None:
            cursor = self.conn.execute(
                """
                INSERT INTO episodes
                    (opened_us, status, summary, last_update_us, peak_bpm,
                     evidence_json, max_severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reading.ts_us,
                    final_state,
                    summary,
                    reading.ts_us,
                    reading.bpm,
                    evidence_json,
                    final_state,
                ),
            )
            return {
                "episode_id": int(cursor.lastrowid),
                "action_type": "alert_opened",
                "severity": final_state,
            }
        if final_state in {"watch", "urgent"} and open_row is not None:
            previous = str(open_row["status"])
            max_severity = (
                "urgent"
                if final_state == "urgent"
                or str(open_row["max_severity"]) == "urgent"
                else "watch"
            )
            peak = open_row["peak_bpm"]
            if reading.bpm is not None:
                peak = (
                    reading.bpm
                    if peak is None
                    else max(float(peak), float(reading.bpm))
                )
            self.conn.execute(
                """
                UPDATE episodes
                SET status=?, summary=?, last_update_us=?, peak_bpm=?,
                    evidence_json=?, max_severity=?
                WHERE id=?
                """,
                (
                    final_state,
                    summary,
                    reading.ts_us,
                    peak,
                    evidence_json,
                    max_severity,
                    open_row["id"],
                ),
            )
            if previous != "urgent" and final_state == "urgent":
                return {
                    "episode_id": int(open_row["id"]),
                    "action_type": "alert_escalated",
                    "severity": "urgent",
                }
            return None
        if final_state == "recovering" and open_row is not None:
            previous = str(open_row["status"])
            self.conn.execute(
                """
                UPDATE episodes
                SET status='recovering', summary=?, last_update_us=?,
                    evidence_json=?
                WHERE id=?
                """,
                (
                    summary,
                    reading.ts_us,
                    evidence_json,
                    open_row["id"],
                ),
            )
            if previous != "recovering":
                return {
                    "episode_id": int(open_row["id"]),
                    "action_type": "recovery_started",
                    "severity": "recovering",
                }
            return None
        if final_state == "normal" and open_row is not None:
            duration_s = max(
                0.0,
                (reading.ts_us - int(open_row["opened_us"])) / 1_000_000.0,
            )
            memory = self.semantic_memory()
            count = int(memory.get("resolved_episode_count", "0")) + 1
            previous_average = float(
                memory.get("average_recovery_duration_s", str(duration_s))
            )
            average = previous_average + (duration_s - previous_average) / count
            severity = str(open_row["max_severity"] or "watch")
            self.conn.execute(
                """
                UPDATE episodes
                SET status='closed', closed_us=?, last_update_us=?,
                    summary=?, outcome='recovered', evidence_json=?
                WHERE id=?
                """,
                (
                    reading.ts_us,
                    reading.ts_us,
                    summary,
                    evidence_json,
                    open_row["id"],
                ),
            )
            self._set_semantic(
                "resolved_episode_count", str(count), reading.ts_us
            )
            self._set_semantic(
                "average_recovery_duration_s",
                f"{average:.2f}",
                reading.ts_us,
            )
            severity_key = f"resolved_{severity}_episode_count"
            severity_count = int(memory.get(severity_key, "0")) + 1
            self._set_semantic(
                severity_key, str(severity_count), reading.ts_us
            )
            return {
                "episode_id": int(open_row["id"]),
                "action_type": "alert_resolved",
                "severity": "normal",
            }
        return None

    def _set_semantic(self, key: str, value: str, ts_us: int) -> None:
        self.conn.execute(
            """
            INSERT INTO semantic_memory(key, value, updated_us)
            VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value, updated_us=excluded.updated_us
            """,
            (key, value, ts_us),
        )

    def _update_baseline(
        self,
        *,
        reading: Reading,
        final_state: str,
        guard: GuardResult,
        had_active_episode: bool,
    ) -> None:
        if (
            final_state != "normal"
            or not guard.measurement_valid
            or reading.bpm is None
            or reading.activity != "rest"
            or had_active_episode
        ):
            return
        memory = self.semantic_memory()
        count = int(memory.get("baseline_candidate_count", "0")) + 1
        previous_mean = float(
            memory.get("baseline_candidate_mean", str(reading.bpm))
        )
        previous_m2 = float(memory.get("baseline_candidate_m2", "0"))
        delta = reading.bpm - previous_mean
        mean = previous_mean + delta / count
        m2 = previous_m2 + delta * (reading.bpm - mean)
        values = {
            "baseline_candidate_count": str(count),
            "baseline_candidate_mean": f"{mean:.4f}",
            "baseline_candidate_m2": f"{m2:.4f}",
        }
        if count >= 60:
            values["resting_bpm"] = f"{mean:.2f}"
            values["resting_bpm_variability"] = (
                f"{sqrt(m2 / max(1, count - 1)):.2f}"
            )
            values["resting_bpm_sample_count"] = str(count)
        for key, value in values.items():
            self.conn.execute(
                """
                INSERT INTO semantic_memory(key, value, updated_us)
                VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value, updated_us=excluded.updated_us
                """,
                (key, value, reading.ts_us),
            )

    @staticmethod
    def _action_message(
        transition: dict,
        response: AgentResponse,
        guard: GuardResult,
        reading: Reading,
    ) -> str:
        bpm = "unavailable" if reading.bpm is None else f"{reading.bpm:.0f} BPM"
        action_type = transition["action_type"]
        evidence_parts = list(guard.reasons)
        if (
            action_type in {"alert_opened", "alert_escalated"}
            and transition["severity"] != guard.state
        ):
            evidence_parts.extend(response.decision.evidence[:2])
        reasons = ", ".join(dict.fromkeys(evidence_parts)) or (
            "agent interpretation"
        )
        confirmation_count = max(
            1,
            guard.features.consecutive_outside_normal,
            guard.features.consecutive_high,
            guard.features.consecutive_low,
            guard.features.consecutive_urgent_high,
            guard.features.consecutive_urgent_low,
        )
        if action_type == "alert_resolved":
            return (
                f"PulseFi recovery confirmed at {bpm} after "
                f"{guard.features.consecutive_recovery} stable readings. "
                f"{response.decision.feedback}"
            )
        if action_type == "recovery_started":
            return (
                f"PulseFi recovery started at {bpm}; continued confirmation "
                f"is required because this is an experimental estimate. "
                f"{response.decision.suggestion}"
            )
        return (
            f"PulseFi {transition['severity']} alert at {bpm}. "
            f"Evidence: {reasons}; confirmation count {confirmation_count}. "
            f"This is an experimental estimate. "
            f"{response.decision.suggestion}"
        )

    def commit_tick(
        self,
        reading: Reading,
        response: AgentResponse,
        guard: GuardResult,
        final_state: str,
        model: str,
        created_us: int,
        trace: dict | None = None,
    ) -> dict | None:
        transition = None
        with self.conn:
            had_active_episode = self.active_episode() is not None
            cursor = self.conn.execute(
                """
                INSERT INTO observations
                    (ts_us, human, presence, bpm, sensor_bpm, sensor_valid,
                     source, activity, symptoms_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reading.ts_us,
                    int(reading.human),
                    reading.presence,
                    reading.bpm,
                    reading.sensor_bpm,
                    int(reading.sensor_valid),
                    reading.source,
                    reading.activity,
                    json.dumps(reading.symptoms),
                ),
            )
            observation_id = int(cursor.lastrowid)
            self.conn.execute(
                """
                INSERT INTO features
                    (observation_id, guard_state, reasons_json, features_json,
                     measurement_valid)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    guard.state,
                    json.dumps(guard.reasons),
                    json.dumps(guard.features.to_dict()),
                    int(guard.measurement_valid),
                ),
            )
            decision = response.decision
            self.conn.execute(
                """
                INSERT INTO decisions
                    (observation_id, status, trend, prediction, feedback,
                     suggestion, confidence, memory_summary, model, created_us,
                     llm_state, final_state, evidence_json,
                     reasoning_available, reasoning_status,
                     prompt_tokens, completion_tokens,
                     total_tokens, request_id, system_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    final_state,
                    decision.trend,
                    decision.forecast,
                    decision.feedback,
                    decision.suggestion,
                    decision.confidence,
                    decision.memory_update,
                    model,
                    created_us,
                    (
                        decision.state
                        if response.reasoning_available
                        else None
                    ),
                    final_state,
                    json.dumps(decision.evidence),
                    int(response.reasoning_available),
                    response.reasoning_status,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                    response.request_id,
                    response.system_fingerprint,
                ),
            )
            transition = self._update_episode(
                final_state=final_state,
                response=response,
                guard=guard,
                reading=reading,
            )
            if response.reasoning_available:
                self.conn.execute(
                    """
                    INSERT INTO semantic_memory(key, value, updated_us)
                    VALUES('latest_agent_memory', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value=excluded.value, updated_us=excluded.updated_us
                    """,
                    (decision.memory_update, reading.ts_us),
                )
            self._update_baseline(
                reading=reading,
                final_state=final_state,
                guard=guard,
                had_active_episode=had_active_episode,
            )
            if transition is not None:
                message = self._action_message(
                    transition, response, guard, reading
                )
                action_cursor = self.conn.execute(
                    """
                    INSERT INTO actions
                        (observation_id, episode_id, action_type, severity,
                         message, created_us)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation_id,
                        transition["episode_id"],
                        transition["action_type"],
                        transition["severity"],
                        message,
                        created_us,
                    ),
                )
                transition = {
                    **transition,
                    "action_id": int(action_cursor.lastrowid),
                    "message": message,
                }
                self.conn.execute(
                    """
                    INSERT INTO integration_outbox
                        (event_type, payload_json, created_us)
                    VALUES ('pulsefi.health_alert.v1', ?, ?)
                    """,
                    (
                        json.dumps(
                            {
                                "action_id": transition["action_id"],
                                "episode_id": transition["episode_id"],
                                "type": transition["action_type"],
                                "severity": transition["severity"],
                                "message": message,
                                "reading": reading.to_dict(),
                                "source": reading.source,
                            }
                        ),
                        created_us,
                    ),
                )
            if trace is not None:
                self._insert_trace(trace)
        return transition

    def pending_integration_events(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT * FROM integration_outbox
            WHERE status='pending'
            ORDER BY id LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            self._decode_json_fields(dict(row), ("payload_json",))
            for row in rows
        ]

    def due_integration_events(
        self, now_us: int, limit: int = 20
    ) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT * FROM integration_outbox
            WHERE status='pending' AND next_attempt_us<=?
            ORDER BY id LIMIT ?
            """,
            (now_us, limit),
        ).fetchall()
        return [
            self._decode_json_fields(dict(row), ("payload_json",))
            for row in rows
        ]

    def record_integration_attempt(
        self,
        event_id: int,
        *,
        channel: str,
        attempted_us: int,
        delivered: bool,
        error: str | None = None,
        next_attempt_us: int = 0,
        max_attempts: int = 5,
    ) -> str:
        with self.conn:
            row = self.conn.execute(
                """
                SELECT payload_json, attempt_count
                FROM integration_outbox WHERE id=?
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Integration event {event_id} does not exist")
            payload = json.loads(str(row["payload_json"]))
            action_id = int(payload["action_id"])
            attempt_count = int(row["attempt_count"]) + 1
            if delivered:
                status = "delivered"
                detail = None
                delivered_us = attempted_us
                external_status = f"{channel}_delivered"
            else:
                status = (
                    "failed"
                    if attempt_count >= max_attempts
                    else "pending"
                )
                detail = (error or "unknown integration error")[:500]
                delivered_us = None
                external_status = (
                    f"{channel}_failed"
                    if status == "failed"
                    else f"{channel}_retrying"
                )
            self.conn.execute(
                """
                INSERT INTO delivery_attempts
                    (action_id, channel, attempted_us, status, detail)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    channel,
                    attempted_us,
                    "delivered" if delivered else "failed",
                    detail,
                ),
            )
            self.conn.execute(
                """
                UPDATE integration_outbox
                SET status=?, attempt_count=?, next_attempt_us=?,
                    last_error=?, delivered_us=?
                WHERE id=?
                """,
                (
                    status,
                    attempt_count,
                    0 if delivered else next_attempt_us,
                    detail,
                    delivered_us,
                    event_id,
                ),
            )
            self.conn.execute(
                """
                UPDATE actions SET external_delivery_status=? WHERE id=?
                """,
                (external_status, action_id),
            )
        return status

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

    def reject_stream_row(
        self,
        *,
        path: str,
        offset: int,
        next_offset: int,
        event_type: str,
        detail: str,
        row_sha256: str,
        created_us: int,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO stream_events
                    (path, offset, next_offset, event_type, detail,
                     row_sha256, created_us)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path,
                    offset,
                    next_offset,
                    event_type,
                    detail[:500],
                    row_sha256,
                    created_us,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO stream_cursor(path, offset) VALUES(?, ?)
                ON CONFLICT(path) DO UPDATE SET offset=excluded.offset
                """,
                (path, next_offset),
            )

    def _insert_trace(self, trace: dict) -> None:
        self.conn.execute(
            """
            INSERT INTO traces
                (run_id, ts_us, source, bpm, model, context_bytes,
                 working_memory_count, episode_count, latency_ms, outcome,
                 status, confidence, error, guard_state, final_state,
                 prompt_tokens, completion_tokens, total_tokens, request_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace["run_id"],
                trace["ts_us"],
                trace["source"],
                trace.get("bpm"),
                trace["model"],
                trace["context_bytes"],
                trace["working_memory_count"],
                trace["episode_count"],
                trace["latency_ms"],
                trace["outcome"],
                trace.get("status"),
                trace.get("confidence"),
                trace.get("error"),
                trace.get("guard_state"),
                trace.get("final_state"),
                trace.get("prompt_tokens", 0),
                trace.get("completion_tokens", 0),
                trace.get("total_tokens", 0),
                trace.get("request_id"),
            ),
        )

    def add_trace(self, trace: dict) -> None:
        with self.conn:
            self._insert_trace(trace)

    def traces(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM traces WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]
