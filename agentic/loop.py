"""One real model-backed agent tick."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

from agentic.db import MemoryDB, PLAYBOOK
from agentic.llm import (
    AgentClient,
    AgentClientError,
    AgentDecision,
    AgentResponse,
    AgentUsage,
)
from agentic.models import DISCLAIMER, Reading
from agentic.policy import GuardResult, evaluate_guard, merge_states


@dataclass(frozen=True)
class TickResult:
    reading: Reading
    decision: AgentDecision
    response: AgentResponse
    guard: GuardResult
    final_state: str
    action: dict | None


class HeartbeatAgent:
    def __init__(
        self,
        db: MemoryDB,
        client: AgentClient | None,
        model_name: str,
        run_id: str | None = None,
        llm_interval_s: float = 30.0,
    ):
        self.db = db
        self.client = client
        self.model_name = model_name
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.llm_interval_us = int(max(1.0, llm_interval_s) * 1_000_000)

    def build_context(self, reading: Reading) -> dict:
        history = self.db.recent_reading_models(limit=60)
        guard = evaluate_guard(
            history,
            reading,
            baseline_bpm=self.db.resting_baseline(),
            episode_active=self.db.active_episode() is not None,
        )
        return self._build_context(reading, guard)

    def _build_context(self, reading: Reading, guard: GuardResult) -> dict:
        return {
            "current_reading": reading.to_dict(),
            "deterministic_guard": guard.to_dict(),
            "working_memory": self.db.recent_readings(limit=12),
            "active_episode": self.db.active_episode(),
            "episodic_memory": self.db.recent_episodes(limit=3),
            "similar_episodes": self.db.similar_episodes(
                reading.bpm, limit=3
            ),
            "semantic_memory": self.db.semantic_memory(),
            "procedural_memory": PLAYBOOK,
            "parametric_memory": self.db.model_memory(),
            "measurement_limits": (
                "BPM is an experimental WiFi-LSTM estimate. Presence and BPM "
                "may be wrong. Sensor BPM is a reference only when sensor_valid is true."
            ),
            "required_safety": DISCLAIMER,
        }

    @staticmethod
    def _guard_only_response(
        guard: GuardResult,
        *,
        status: str,
        error: str | None = None,
    ) -> AgentResponse:
        suggestion = {
            "no_person": "Wait for a reliable presence reading.",
            "normal": "Continue monitoring; no action is indicated.",
            "recovering": "Continue resting while PulseFi confirms recovery.",
            "watch": (
                "Pause activity and verify the reading with a validated device."
            ),
            "urgent": (
                "Stop activity and verify with a validated device. If serious "
                "symptoms are present, seek emergency help."
            ),
        }[guard.state]
        if status == "not_scheduled":
            feedback = (
                "Groq reasoning was not scheduled for this tick; the final "
                "state is the deterministic measurement guard."
            )
        elif status == "disabled":
            feedback = (
                "Groq reasoning is disabled because no API credential is "
                "configured; the final state is the deterministic "
                "measurement guard."
            )
        else:
            feedback = (
                "The Groq request failed for this tick; the final state is the "
                "deterministic measurement guard."
            )
        return AgentResponse(
            decision=AgentDecision(
                state=guard.state,
                trend="Deterministic guard-only tick",
                forecast="No model forecast was produced.",
                evidence=guard.reasons or ("measurement guard result",),
                feedback=feedback,
                suggestion=suggestion,
                confidence=0.0,
                memory_update="No LLM memory update was produced for this tick.",
            ),
            usage=AgentUsage(0, 0, 0, None, None),
            request_id=None,
            system_fingerprint=None,
            reasoning_available=False,
            reasoning_status=status,
            error=error,
        )

    def _should_call_model(self, reading: Reading, guard: GuardResult) -> bool:
        if self.client is None:
            return False
        latest = self.db.latest_decision()
        previous_state = None if latest is None else latest.get("final_state")
        state_transition = previous_state != guard.state
        if state_transition and guard.state in {"watch", "urgent"}:
            return True
        last_success = self.db.last_successful_reasoning_ts_us()
        return (
            last_success is None
            or reading.ts_us - last_success >= self.llm_interval_us
        )

    def tick(self, reading: Reading) -> TickResult:
        if self.db.observation_exists(reading):
            raise ValueError(
                f"Reading {reading.ts_us}/{reading.source} was already processed"
            )
        history = self.db.recent_reading_models(limit=60)
        guard = evaluate_guard(
            history,
            reading,
            baseline_bpm=self.db.resting_baseline(),
            episode_active=self.db.active_episode() is not None,
        )
        context = self._build_context(reading, guard)
        trace = {
            "run_id": self.run_id,
            "ts_us": reading.ts_us,
            "source": reading.source,
            "bpm": reading.bpm,
            "model": self.model_name,
            "context_bytes": len(
                json.dumps(context, separators=(",", ":")).encode("utf-8")
            ),
            "working_memory_count": len(context["working_memory"]),
            "episode_count": len(context["episodic_memory"]),
            "guard_state": guard.state,
        }
        started = time.perf_counter()
        if self.client is None:
            response = self._guard_only_response(guard, status="disabled")
        elif not self._should_call_model(reading, guard):
            response = self._guard_only_response(
                guard, status="not_scheduled"
            )
        else:
            try:
                response = self.client.analyze(context)
            except AgentClientError as exc:
                response = self._guard_only_response(
                    guard,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
            except Exception as exc:
                trace.update(
                    {
                        "latency_ms": (
                            time.perf_counter() - started
                        )
                        * 1000.0,
                        "outcome": "error",
                        "status": None,
                        "confidence": None,
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                        "final_state": None,
                    }
                )
                self.db.add_trace(trace)
                raise
        decision = response.decision
        final_state = (
            merge_states(guard.state, decision.state)
            if response.reasoning_available
            else guard.state
        )
        trace.update(
            {
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "outcome": (
                    "success"
                    if response.reasoning_available
                    else response.reasoning_status
                ),
                "status": (
                    decision.state if response.reasoning_available else None
                ),
                "confidence": (
                    decision.confidence
                    if response.reasoning_available
                    else None
                ),
                "error": response.error,
                "final_state": final_state,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "request_id": response.request_id,
            }
        )
        action = self.db.commit_tick(
            reading=reading,
            response=response,
            guard=guard,
            final_state=final_state,
            model=self.model_name,
            created_us=int(time.time() * 1_000_000),
            trace=trace,
        )
        return TickResult(
            reading=reading,
            decision=decision,
            response=response,
            guard=guard,
            final_state=final_state,
            action=action,
        )
