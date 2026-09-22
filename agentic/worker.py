"""Continuously tail PulseFi inference and run the LLM agent."""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

from agentic.db import MemoryDB
from agentic.llm import AgentClientError, OpenAICompatibleClient
from agentic.loop import HeartbeatAgent
from agentic.models import DISCLAIMER, Reading, reading_from_row


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "runtime" / "live_predictions.csv"
DEFAULT_DB = ROOT / "runtime" / "pulsefi_agent.db"


@dataclass(frozen=True)
class PendingRow:
    reading: Reading
    next_offset: int


class CsvTail:
    def __init__(self, path: Path, db: MemoryDB):
        self.path = path
        self.db = db
        self.header: list[str] | None = None
        self.offset = db.cursor_get(str(path))

    def poll(self) -> PendingRow | None:
        if not self.path.exists():
            return None
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.header = None
        with self.path.open("r", newline="") as handle:
            header_line = handle.readline()
            if not header_line.endswith("\n"):
                return None
            self.header = next(csv.reader([header_line]))
            header_end = handle.tell()
            if self.offset < header_end:
                self.offset = header_end
            if self.offset >= size:
                return None
            handle.seek(self.offset)
            line = handle.readline()
            if not line.endswith("\n"):
                return None
            next_offset = handle.tell()
        cells = next(csv.reader([line]))
        reading = reading_from_row(dict(zip(self.header, cells)), source="lstm_live")
        if reading is None:
            self.offset = next_offset
            self.db.cursor_set(str(self.path), self.offset)
            return None
        return PendingRow(reading=reading, next_offset=next_offset)

    def commit(self, offset: int) -> None:
        self.offset = offset
        self.db.cursor_set(str(self.path), offset)


def serve(csv_path: Path, db_path: Path, poll_s: float) -> None:
    client = OpenAICompatibleClient.from_env()
    db = MemoryDB(db_path)
    tail = CsvTail(csv_path, db)
    agent = HeartbeatAgent(db, client, client.model)
    print("PulseFi LLM heartbeat worker")
    print(DISCLAIMER)
    print(f"Model: {client.model}")
    print(f"Watching: {csv_path}")
    print(f"Memory DB: {db_path}")
    try:
        while True:
            pending = tail.poll()
            if pending is None:
                time.sleep(poll_s)
                continue
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
            bpm = "none" if result.reading.bpm is None else f"{result.reading.bpm:.1f}"
            print(
                f"bpm={bpm} status={decision.status} "
                f"confidence={decision.confidence:.2f}"
            )
            print(f"  prediction: {decision.prediction}")
            print(f"  feedback: {decision.feedback}")
            print(f"  suggestion: {decision.suggestion}")
    except KeyboardInterrupt:
        print("Worker stopped.")
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PulseFi LLM agent worker.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error(
            "OPENAI_API_KEY is missing. Set a newly created key in the environment."
        )
    serve(Path(args.csv), Path(args.db), max(0.1, args.poll_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
