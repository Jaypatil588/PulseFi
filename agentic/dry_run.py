"""Real API-backed dry run with clearly labeled synthetic heart-rate input."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from agentic.db import MemoryDB
from agentic.llm import AgentClientError, GroqClient
from agentic.loop import HeartbeatAgent
from agentic.models import Reading
from agentic.worker import ROOT, load_local_env


DEFAULT_DB = ROOT / "runtime" / "pulsefi_dry_run.db"
DEFAULT_TRACE = ROOT / "runtime" / "pulsefi_dry_run_trace.jsonl"


def generate_readings(seconds: int, spikes: int, seed: int) -> list[Reading]:
    if seconds < 2:
        raise ValueError("seconds must be at least 2")
    if spikes < 1 or spikes >= seconds:
        raise ValueError("spikes must be between 1 and seconds - 1")
    rng = random.Random(seed)
    spike_indexes = set(rng.sample(range(1, seconds), spikes))
    start_us = int(time.time() * 1_000_000)
    readings = []
    for index in range(seconds):
        bpm = 150.0 if index in spike_indexes else float(rng.randint(80, 100))
        readings.append(
            Reading(
                ts_us=start_us + index * 1_000_000,
                human=True,
                presence=round(rng.uniform(0.85, 0.99), 3),
                bpm=bpm,
                sensor_bpm=None,
                sensor_valid=False,
                source="synthetic_dry_run",
            )
        )
    return readings


def main() -> int:
    load_local_env()
    parser = argparse.ArgumentParser(
        description="Call the real agent on labeled synthetic BPM readings."
    )
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument("--spikes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--trace-jsonl", default=str(DEFAULT_TRACE))
    args = parser.parse_args()
    if not os.environ.get("GROQ_API_KEY"):
        parser.error("GROQ_API_KEY is missing")

    client = GroqClient.from_env()
    db = MemoryDB(Path(args.db))
    run_id = f"dry_{int(time.time())}_{args.seed}"
    agent = HeartbeatAgent(db, client, client.model, run_id=run_id)
    readings = generate_readings(args.seconds, args.spikes, args.seed)
    reactions: list[tuple[bool, bool]] = []

    print("REAL API DRY RUN — input readings are synthetic")
    print(f"Model: {client.model}")
    print(f"Database: {args.db}")
    try:
        for second, reading in enumerate(readings):
            print(f"\nsecond={second:02d} synthetic_bpm={reading.bpm:.0f}")
            try:
                result = agent.tick(reading)
            except AgentClientError as exc:
                print(f"API failure: {exc}")
                return 1
            decision = result.decision
            expected_spike = reading.bpm >= 150.0
            predicted_concerning = decision.status in {"watch", "urgent"}
            reactions.append((expected_spike, predicted_concerning))
            print(
                f"status={decision.status} confidence={decision.confidence:.2f}"
            )
            print(f"trend: {decision.trend}")
            print(f"prediction: {decision.prediction}")
            print(f"feedback: {decision.feedback}")
            print(f"suggestion: {decision.suggestion}")
        traces = db.traces(run_id)
    finally:
        db.close()

    trace_path = Path(args.trace_jsonl)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")

    tp = sum(expected and predicted for expected, predicted in reactions)
    fp = sum(not expected and predicted for expected, predicted in reactions)
    fn = sum(expected and not predicted for expected, predicted in reactions)
    tn = sum(not expected and not predicted for expected, predicted in reactions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(reactions) if reactions else 0.0
    latency = [float(trace["latency_ms"]) for trace in traces]

    print("\nEVALUATION")
    print(f"run_id={run_id}")
    print(f"tp={tp} fp={fp} fn={fn} tn={tn}")
    print(
        f"precision={precision:.3f} recall={recall:.3f} "
        f"accuracy={accuracy:.3f}"
    )
    if latency:
        print(
            f"latency_ms_mean={sum(latency) / len(latency):.1f} "
            f"latency_ms_max={max(latency):.1f}"
        )
    print(f"trace={trace_path}")
    print("DRY RUN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
