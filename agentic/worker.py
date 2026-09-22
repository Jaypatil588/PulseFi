"""Continuously tail PulseFi inference and run the LLM agent."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

from agentic.db import MemoryDB
from agentic.integration import JsonlTransport, OutboxDispatcher
from agentic.llm import AgentClientError, GroqClient
from agentic.loop import HeartbeatAgent
from agentic.models import (
    DISCLAIMER,
    SUPPORTED_PREDICTION_HEADERS,
    Reading,
    reading_from_row,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "runtime" / "live_predictions.csv"
DEFAULT_DB = ROOT / "runtime" / "pulsefi_agent.db"


def load_local_env(path: Path = ROOT / ".env") -> None:
    """Load simple KEY=VALUE entries without overriding shell variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    for key in (
        "GROQ_API_KEY",
        "PULSEFI_AGENT_MODEL",
        "PULSEFI_AGENT_BASE_URL",
        "PULSEFI_ALERT_EXPORT_JSONL",
    ):
        if key in os.environ:
            os.environ[key] = os.environ[key].strip().strip("\"'")


@dataclass(frozen=True)
class PendingRow:
    reading: Reading
    next_offset: int


@dataclass(frozen=True)
class RejectedRow:
    offset: int
    next_offset: int
    event_type: str
    detail: str
    row_sha256: str


class StreamSchemaError(RuntimeError):
    def __init__(self, offset: int, detail: str, header_sha256: str):
        super().__init__(detail)
        self.offset = offset
        self.detail = detail
        self.header_sha256 = header_sha256


class WorkerLock:
    def __init__(self, db_path: Path):
        resolved = db_path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(f"{resolved}.worker.lock")
        self.handle = None

    def __enter__(self) -> "WorkerLock":
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(
                self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                f"Another PulseFi worker owns {self.path}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"{os.getpid()}\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


class CsvTail:
    def __init__(self, path: Path, db: MemoryDB, source: str):
        self.path = path
        self.db = db
        self.source = source
        self.header: list[str] | None = None
        self.offset = db.cursor_get(str(path))
        self.file_identity: tuple[int, int] | None = None

    def poll(self) -> PendingRow | RejectedRow | None:
        if not self.path.exists():
            return None
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.file_identity is not None and identity != self.file_identity:
            self.offset = 0
            self.header = None
        self.file_identity = identity
        size = stat.st_size
        if size < self.offset:
            self.offset = 0
            self.header = None
        with self.path.open("r", newline="") as handle:
            header_line = handle.readline()
            if not header_line.endswith("\n"):
                return None
            header_sha256 = hashlib.sha256(
                header_line.encode("utf-8")
            ).hexdigest()
            try:
                candidate_header = next(csv.reader([header_line]))
            except csv.Error as exc:
                raise StreamSchemaError(
                    self.offset,
                    f"CSV header could not be parsed: {exc}",
                    header_sha256,
                ) from exc
            if candidate_header:
                candidate_header[0] = candidate_header[0].lstrip("\ufeff")
            candidate = tuple(candidate_header)
            if candidate not in SUPPORTED_PREDICTION_HEADERS:
                raise StreamSchemaError(
                    self.offset,
                    (
                        "Unsupported prediction CSV header; use the canonical "
                        "live_predictions.csv schema"
                    ),
                    header_sha256,
                )
            if self.header is not None and candidate_header != self.header:
                raise StreamSchemaError(
                    self.offset,
                    "Prediction CSV header changed without stream reset",
                    header_sha256,
                )
            self.header = candidate_header
            header_end = handle.tell()
            if self.offset < header_end:
                self.offset = header_end
            if self.offset >= size:
                return None
            handle.seek(self.offset)
            row_offset = self.offset
            line = handle.readline()
            if not line.endswith("\n"):
                return None
            next_offset = handle.tell()
        row_sha256 = hashlib.sha256(line.encode("utf-8")).hexdigest()
        try:
            cells = next(csv.reader([line]))
        except csv.Error as exc:
            return RejectedRow(
                offset=row_offset,
                next_offset=next_offset,
                event_type="malformed_csv",
                detail=f"CSV parser rejected row: {exc}",
                row_sha256=row_sha256,
            )
        if len(cells) != len(self.header):
            return RejectedRow(
                offset=row_offset,
                next_offset=next_offset,
                event_type="column_count_mismatch",
                detail=(
                    f"Expected {len(self.header)} columns, received "
                    f"{len(cells)}"
                ),
                row_sha256=row_sha256,
            )
        reading = reading_from_row(
            dict(zip(self.header, cells)), source=self.source
        )
        if reading is None:
            return RejectedRow(
                offset=row_offset,
                next_offset=next_offset,
                event_type="invalid_reading",
                detail="Required timestamp or presence fields are invalid",
                row_sha256=row_sha256,
            )
        return PendingRow(reading=reading, next_offset=next_offset)

    def commit(self, offset: int) -> None:
        self.offset = offset
        self.db.cursor_set(str(self.path), offset)

    def reject(self, rejected: RejectedRow) -> None:
        self.db.reject_stream_row(
            path=str(self.path),
            offset=rejected.offset,
            next_offset=rejected.next_offset,
            event_type=rejected.event_type,
            detail=rejected.detail,
            row_sha256=rejected.row_sha256,
            created_us=int(time.time() * 1_000_000),
        )
        self.offset = rejected.next_offset

    def record_schema_error(self, error: StreamSchemaError) -> None:
        self.db.reject_stream_row(
            path=str(self.path),
            offset=error.offset,
            next_offset=error.offset,
            event_type="incompatible_schema",
            detail=error.detail,
            row_sha256=error.header_sha256,
            created_us=int(time.time() * 1_000_000),
        )


def serve(
    csv_path: Path,
    db_path: Path,
    poll_s: float,
    activity: str,
    source: str,
    llm_interval_s: float,
) -> None:
    with WorkerLock(db_path):
        client = (
            GroqClient.from_env()
            if os.environ.get("GROQ_API_KEY")
            else None
        )
        model_name = (
            client.model if client is not None else "deterministic-guard-only"
        )
        db = MemoryDB(db_path)
        export_path = os.environ.get("PULSEFI_ALERT_EXPORT_JSONL", "").strip()
        dispatcher = (
            OutboxDispatcher(db, JsonlTransport(Path(export_path)))
            if export_path
            else None
        )
        tail = CsvTail(csv_path, db, source)
        agent = HeartbeatAgent(
            db,
            client,
            model_name,
            llm_interval_s=llm_interval_s,
        )
        print("PulseFi heartbeat worker")
        print(DISCLAIMER)
        print(f"Model: {model_name}")
        print(f"Watching: {csv_path}")
        print(f"Memory DB: {db_path}")
        print(
            f"Alert export: {export_path}"
            if dispatcher is not None
            else "Alert export: not configured; outbox remains pending"
        )
        try:
            while True:
                if dispatcher is not None:
                    dispatcher.poll()
                try:
                    pending = tail.poll()
                except StreamSchemaError as exc:
                    tail.record_schema_error(exc)
                    raise RuntimeError(
                        f"Input schema error: {exc.detail}"
                    ) from exc
                if pending is None:
                    time.sleep(poll_s)
                    continue
                if isinstance(pending, RejectedRow):
                    tail.reject(pending)
                    print(
                        f"Rejected input row at byte {pending.offset}: "
                        f"{pending.event_type}"
                    )
                    continue
                if pending.reading.activity == "unknown":
                    pending.reading.activity = activity
                if db.observation_exists(pending.reading):
                    tail.commit(pending.next_offset)
                    continue
                try:
                    result = agent.tick(pending.reading)
                except AgentClientError as exc:
                    print(f"Agent API error; row retained for retry: {exc}")
                    time.sleep(min(10.0, max(1.0, poll_s * 5)))
                    continue
                tail.commit(pending.next_offset)
                decision = result.decision
                bpm = (
                    "none"
                    if result.reading.bpm is None
                    else f"{result.reading.bpm:.1f}"
                )
                llm_state = (
                    decision.state
                    if result.response.reasoning_available
                    else result.response.reasoning_status
                )
                print(
                    f"bpm={bpm} guard={result.guard.state} "
                    f"llm={llm_state} final={result.final_state} "
                    f"confidence={decision.confidence:.2f}"
                )
                print(f"  forecast: {decision.forecast}")
                print(f"  feedback: {decision.feedback}")
                print(f"  suggestion: {decision.suggestion}")
                if result.action is not None:
                    print(f"  message: {result.action['message']}")
        except KeyboardInterrupt:
            print("Worker stopped.")
        finally:
            db.close()


def main() -> int:
    load_local_env()
    parser = argparse.ArgumentParser(description="Run the PulseFi LLM agent worker.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument(
        "--activity",
        choices=["rest", "exercise", "sleep", "unknown"],
        default="rest",
    )
    parser.add_argument(
        "--source",
        choices=["lstm_live", "generated_stream", "recorded_replay"],
        default="lstm_live",
    )
    parser.add_argument(
        "--llm-interval-seconds",
        type=float,
        default=float(os.getenv("PULSEFI_LLM_INTERVAL_SECONDS", "30")),
        help="Periodic Groq cadence; watch/urgent transitions still call immediately.",
    )
    args = parser.parse_args()
    try:
        serve(
            Path(args.csv),
            Path(args.db),
            max(0.1, args.poll_seconds),
            args.activity,
            args.source,
            max(1.0, args.llm_interval_seconds),
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
