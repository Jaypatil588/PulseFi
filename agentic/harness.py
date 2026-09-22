"""Offline contract test for the agent loop.

This harness uses a recording fake client. It proves context, validation, and
database persistence only. It does not claim that an external model was called.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentic.db import MemoryDB
from agentic.llm import AgentDecision
from agentic.loop import HeartbeatAgent
from agentic.models import Reading


class RecordingFakeClient:
    def __init__(self):
        self.contexts: list[dict] = []

    def analyze(self, context: dict) -> AgentDecision:
        self.contexts.append(context)
        return AgentDecision(
            status="watch",
            trend="synthetic elevated-rate test",
            prediction="The synthetic test rate may remain elevated; uncertain.",
            feedback="This is output from the offline contract-test client.",
            suggestion="Do not treat this synthetic result as health advice.",
            confidence=0.5,
            memory_summary="Synthetic contract test produced one watch decision.",
        )


def main() -> int:
    print("OFFLINE CONTRACT TEST — no external AI API is called")
    with tempfile.TemporaryDirectory() as directory:
        db = MemoryDB(Path(directory) / "agent.db")
        client = RecordingFakeClient()
        agent = HeartbeatAgent(db, client, "recording-fake-client")
        reading = Reading(
            ts_us=1_700_000_000_000_000,
            human=True,
            presence=0.91,
            bpm=128.0,
            sensor_bpm=127.0,
            sensor_valid=True,
            source="synthetic_test",
        )
        result = agent.tick(reading)
        problems = []
        if len(client.contexts) != 1:
            problems.append("agent client was not invoked exactly once")
        if client.contexts[0]["current_reading"]["source"] != "synthetic_test":
            problems.append("synthetic source label was lost")
        saved = db.latest_decision()
        if saved is None:
            problems.append("decision was not persisted")
        elif saved["model"] != "recording-fake-client":
            problems.append("model identity was not persisted")
        if result.decision.status != "watch":
            problems.append("validated decision was not returned")
        db.close()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("PASS: local plumbing only; live API behavior remains unverified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
