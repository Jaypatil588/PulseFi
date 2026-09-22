"""Real outbox delivery interfaces for optional external adapters."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Protocol

from agentic.db import MemoryDB


class IntegrationTransport(Protocol):
    channel: str

    def deliver(self, event: dict) -> None: ...


class JsonlTransport:
    """Append versioned outbox events to a durable local JSONL sink."""

    channel = "jsonl"

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def deliver(self, event: dict) -> None:
        record = {
            "outbox_id": int(event["id"]),
            "event_type": str(event["event_type"]),
            "created_us": int(event["created_us"]),
            "payload": event["payload_json"],
        }
        encoded = (
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            written = 0
            while written < len(encoded):
                count = os.write(descriptor, encoded[written:])
                if count <= 0:
                    raise OSError("JSONL transport wrote zero bytes")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class OutboxDispatcher:
    def __init__(
        self,
        db: MemoryDB,
        transport: IntegrationTransport,
        *,
        poll_interval_s: float = 1.0,
        max_attempts: int = 5,
    ):
        self.db = db
        self.transport = transport
        self.poll_interval_s = max(0.1, poll_interval_s)
        self.max_attempts = max(1, max_attempts)
        self.next_poll_monotonic = 0.0

    def poll(self) -> int:
        now_monotonic = time.monotonic()
        if now_monotonic < self.next_poll_monotonic:
            return 0
        self.next_poll_monotonic = now_monotonic + self.poll_interval_s
        now_us = int(time.time() * 1_000_000)
        delivered_count = 0
        for event in self.db.due_integration_events(now_us):
            try:
                self.transport.deliver(event)
            except Exception as exc:
                attempt = int(event["attempt_count"]) + 1
                delay_s = min(300, 2 ** min(attempt, 8))
                self.db.record_integration_attempt(
                    int(event["id"]),
                    channel=self.transport.channel,
                    attempted_us=now_us,
                    delivered=False,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                    next_attempt_us=now_us + delay_s * 1_000_000,
                    max_attempts=self.max_attempts,
                )
            else:
                self.db.record_integration_attempt(
                    int(event["id"]),
                    channel=self.transport.channel,
                    attempted_us=now_us,
                    delivered=True,
                    max_attempts=self.max_attempts,
                )
                delivered_count += 1
        return delivered_count
