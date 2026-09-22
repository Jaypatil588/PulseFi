"""Groq client used by the heartbeat agent."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class AgentClientError(RuntimeError):
    """The model could not produce a valid agent decision."""


@dataclass(frozen=True)
class AgentDecision:
    state: str
    trend: str
    forecast: str
    evidence: tuple[str, ...]
    feedback: str
    suggestion: str
    confidence: float
    memory_update: str

    @classmethod
    def from_dict(cls, value: dict) -> "AgentDecision":
        required = (
            "state",
            "trend",
            "forecast",
            "evidence",
            "feedback",
            "suggestion",
            "confidence",
            "memory_update",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise AgentClientError(f"Model response missing: {', '.join(missing)}")
        extra = sorted(set(value) - set(required))
        if extra:
            raise AgentClientError(
                f"Model response contains unsupported fields: {', '.join(extra)}"
            )
        state = str(value["state"])
        if state not in {
            "no_person",
            "normal",
            "recovering",
            "watch",
            "urgent",
        }:
            raise AgentClientError(f"Invalid state: {state}")
        confidence = float(value["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise AgentClientError("confidence must be between 0 and 1")
        evidence_value = value["evidence"]
        if not isinstance(evidence_value, list) or not 1 <= len(evidence_value) <= 8:
            raise AgentClientError("evidence must contain 1 to 8 strings")
        evidence = tuple(str(item)[:300] for item in evidence_value)
        decision = cls(
            state=state,
            trend=str(value["trend"])[:80],
            forecast=str(value["forecast"])[:500],
            evidence=evidence,
            feedback=str(value["feedback"])[:500],
            suggestion=str(value["suggestion"])[:500],
            confidence=confidence,
            memory_update=str(value["memory_update"])[:500],
        )
        _validate_safe_language(decision)
        return decision

    @property
    def status(self) -> str:
        return self.state

    @property
    def prediction(self) -> str:
        return self.forecast

    @property
    def memory_summary(self) -> str:
        return self.memory_update


@dataclass(frozen=True)
class AgentUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    queue_time_s: float | None
    total_time_s: float | None


@dataclass(frozen=True)
class AgentResponse:
    decision: AgentDecision
    usage: AgentUsage
    request_id: str | None
    system_fingerprint: str | None
    reasoning_available: bool = True
    reasoning_status: str = "success"
    error: str | None = None


class AgentClient(Protocol):
    def analyze(self, context: dict) -> AgentResponse: ...


UNSAFE_LANGUAGE = (
    re.compile(
        r"\byou (?:have|are having|are experiencing) "
        r"(?:a |an )?(?:heart attack|arrhythmia|disease|condition)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:stop|start|increase|decrease|double|skip) "
        r"(?:taking )?(?:your )?(?:medication|medicine|dose)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:guaranteed|definitely safe)\b", re.IGNORECASE),
)


def _validate_safe_language(decision: AgentDecision) -> None:
    text = " ".join(
        (
            decision.trend,
            decision.forecast,
            *decision.evidence,
            decision.feedback,
            decision.suggestion,
            decision.memory_update,
        )
    )
    for pattern in UNSAFE_LANGUAGE:
        if pattern.search(text):
            raise AgentClientError(
                "Model response failed the deterministic safety-language check"
            )


def _validate_grounded_evidence(
    decision: AgentDecision, context: dict
) -> None:
    evidence = " ".join(decision.evidence).lower()
    current = context.get("current_reading") or {}
    unsupported = {
        "spo2": ("spo2", "oxygen saturation"),
        "blood_pressure": ("blood pressure", "systolic", "diastolic"),
        "ecg": ("ecg", "electrocardiogram"),
    }
    for field, terms in unsupported.items():
        if current.get(field) is None and any(term in evidence for term in terms):
            raise AgentClientError(
                f"Model evidence cites unavailable measurement: {field}"
            )
    reported_symptoms = set(current.get("symptoms") or ())
    for symptom in (
        "chest pain",
        "fainting",
        "shortness of breath",
        "dizziness",
        "palpitations",
    ):
        if symptom in evidence and symptom not in reported_symptoms:
            raise AgentClientError(
                f"Model evidence invents an unreported symptom: {symptom}"
            )


SYSTEM_PROMPT = """You are the reasoning component of PulseFi, a research
heartbeat-monitoring system. A deterministic measurement guard runs separately
and is authoritative. You receive its result, experimental WiFi-LSTM presence
and BPM estimates, recent readings, active/recent episodes, a personal profile,
and procedural memory.

Interpret only the supplied evidence. Do not invent symptoms, measurements,
diagnoses, or medical history. Never diagnose a disease, guarantee safety,
promise an outcome, recommend changing medication, or claim an emergency from
the device reading alone. Clearly state uncertainty. If evidence is concerning,
recommend stopping activity and verifying with a validated device. Emergency
guidance is conditional on serious symptoms or the explicit deterministic
policy included in context.

Return only the requested JSON schema. The application deterministically merges
your state with the measurement guard; you cannot downgrade a guard-raised
state.
"""


DECISION_SCHEMA = {
    "name": "pulsefi_agent_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "state": {
                "type": "string",
                "enum": [
                    "no_person",
                    "normal",
                    "recovering",
                    "watch",
                    "urgent",
                ],
            },
            "trend": {"type": "string"},
            "forecast": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
            },
            "feedback": {"type": "string"},
            "suggestion": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "memory_update": {"type": "string"},
        },
        "required": [
            "state",
            "trend",
            "forecast",
            "evidence",
            "feedback",
            "suggestion",
            "confidence",
            "memory_update",
        ],
    },
}


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.removesuffix("```").strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise AgentClientError("Model did not return valid JSON") from exc
    if not isinstance(value, dict):
        raise AgentClientError("Model response must be a JSON object")
    return value


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_string(value) -> str | None:
    if value is None:
        return None
    return str(value)


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 30.0,
    ):
        if not api_key:
            raise AgentClientError("GROQ_API_KEY is required")
        if not model:
            raise AgentClientError("PULSEFI_AGENT_MODEL is required")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "GroqClient":
        return cls(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            model=os.environ.get(
                "PULSEFI_AGENT_MODEL", "openai/gpt-oss-120b"
            ),
            base_url=os.environ.get(
                "PULSEFI_AGENT_BASE_URL",
                "https://api.groq.com/openai/v1",
            ),
        )

    def analyze(self, context: dict) -> AgentResponse:
        body = {
            "model": self.model,
            "temperature": 0.2,
            "reasoning_effort": "low",
            "max_completion_tokens": 1000,
            "response_format": {
                "type": "json_schema",
                "json_schema": DECISION_SCHEMA,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(context, separators=(",", ":")),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "PulseFi-Agent/0.2",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise AgentClientError(f"Model API returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AgentClientError(f"Model API request failed: {exc}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentClientError("Model API returned an unexpected response") from exc
        decision = AgentDecision.from_dict(_extract_json(content))
        _validate_grounded_evidence(decision, context)
        usage = payload.get("usage") or {}
        return AgentResponse(
            decision=decision,
            usage=AgentUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                queue_time_s=_optional_float(usage.get("queue_time")),
                total_time_s=_optional_float(usage.get("total_time")),
            ),
            request_id=_optional_string(
                (payload.get("x_groq") or {}).get("id") or payload.get("id")
            ),
            system_fingerprint=_optional_string(
                payload.get("system_fingerprint")
            ),
        )
