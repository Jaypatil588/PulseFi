"""One real model-backed agent tick."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agentic.db import MemoryDB, PLAYBOOK
from agentic.llm import AgentClient, AgentDecision
from agentic.models import DISCLAIMER, Reading


@dataclass(frozen=True)
class TickResult:
    reading: Reading
    decision: AgentDecision


class HeartbeatAgent:
    def __init__(self, db: MemoryDB, client: AgentClient, model_name: str):
        self.db = db
        self.client = client
        self.model_name = model_name

    def build_context(self, reading: Reading) -> dict:
        return {
            "current_reading": reading.to_dict(),
            "working_memory": self.db.recent_readings(limit=20),
            "episodic_memory": self.db.recent_episodes(limit=5),
            "semantic_memory": self.db.semantic_memory(),
            "procedural_memory": PLAYBOOK,
            "parametric_memory": self.db.model_memory(),
            "measurement_limits": (
                "BPM is an experimental WiFi-LSTM estimate. Presence and BPM "
                "may be wrong. Sensor BPM is a reference only when sensor_valid is true."
            ),
            "required_safety": DISCLAIMER,
        }

    def tick(self, reading: Reading) -> TickResult:
        if self.db.observation_exists(reading):
            raise ValueError(
                f"Reading {reading.ts_us}/{reading.source} was already processed"
            )
        decision = self.client.analyze(self.build_context(reading))
        self.db.commit_tick(
            reading=reading,
            decision=decision,
            model=self.model_name,
            created_us=int(time.time() * 1_000_000),
        )
        return TickResult(reading=reading, decision=decision)
