"""Small OpenAI-compatible client used by the heartbeat agent."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class AgentClientError(RuntimeError):
    """The model could not produce a valid agent decision."""


@dataclass(frozen=True)
class AgentDecision:
    status: str
    trend: str
    prediction: str
    feedback: str
    suggestion: str
    confidence: float
    memory_summary: str

    @classmethod
    def from_dict(cls, value: dict) -> "AgentDecision":
        required = (
            "status",
            "trend",
            "prediction",
            "feedback",
            "suggestion",
            "confidence",
            "memory_summary",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise AgentClientError(f"Model response missing: {', '.join(missing)}")
        status = str(value["status"])
        if status not in {"no_person", "normal", "watch", "urgent"}:
            raise AgentClientError(f"Invalid status: {status}")
        confidence = float(value["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise AgentClientError("confidence must be between 0 and 1")
        return cls(
            status=status,
            trend=str(value["trend"])[:80],
            prediction=str(value["prediction"])[:500],
            feedback=str(value["feedback"])[:500],
            suggestion=str(value["suggestion"])[:500],
            confidence=confidence,
            memory_summary=str(value["memory_summary"])[:500],
        )


class AgentClient(Protocol):
    def analyze(self, context: dict) -> AgentDecision: ...


SYSTEM_PROMPT = """You are the PulseFi heartbeat monitoring agent.
You receive WiFi-LSTM heart-rate estimates, presence confidence, recent
readings, past episodes, and learned user facts.

Analyze the trend and provide cautious wellness feedback. This research
prototype is not a medical device. Never diagnose a disease, promise an
outcome, or claim a medical emergency from these readings. If a sustained
reading looks concerning, advise the user to stop activity, recheck with a
validated device, and seek professional or emergency help only when symptoms
or immediate danger are present.

Return only a JSON object with exactly these fields:
status: one of no_person, normal, watch, urgent
trend: short description
prediction: what the next few readings may do, explicitly uncertain
feedback: concise interpretation
suggestion: one concrete, cautious action
confidence: number from 0 to 1
memory_summary: one sentence worth remembering for later ticks
"""


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


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 30.0,
    ):
        if not api_key:
            raise AgentClientError("OPENAI_API_KEY is required")
        if not model:
            raise AgentClientError("PULSEFI_AGENT_MODEL is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("PULSEFI_AGENT_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get(
                "PULSEFI_AGENT_BASE_URL", "https://api.openai.com/v1"
            ),
        )

    def analyze(self, context: dict) -> AgentDecision:
        body = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
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
        return AgentDecision.from_dict(_extract_json(content))
