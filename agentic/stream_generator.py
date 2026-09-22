"""Generate a schema-accurate PulseFi prediction stream.

This substitutes only for unavailable ESP32/LSTM runtime output. It never
generates agent decisions, feedback, memories, or alerts.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from agentic.models import LIVE_PREDICTION_HEADER
from agentic.worker import DEFAULT_CSV


HEADER = list(LIVE_PREDICTION_HEADER)


def build_bpm_stream(
    seconds: int,
    spike_count: int,
    spike_length: int,
    seed: int,
) -> list[float]:
    if seconds < 5:
        raise ValueError("seconds must be at least 5")
    if spike_count < 0:
        raise ValueError("spike_count cannot be negative")
    if spike_length < 1:
        raise ValueError("spike_length must be positive")
    rng = random.Random(seed)
    values = [float(rng.randint(80, 100)) for _ in range(seconds)]
    available = list(range(3, max(3, seconds - spike_length)))
    starts: list[int] = []
    while available and len(starts) < spike_count:
        start = rng.choice(available)
        starts.append(start)
        available = [
            value
            for value in available
            if abs(value - start) > spike_length + 2
        ]
    if len(starts) != spike_count:
        raise ValueError("duration is too short for requested separated spikes")
    for start in starts:
        for offset in range(spike_length):
            values[start + offset] = 150.0
    return values


def write_stream(
    output: Path,
    bpms: list[float],
    *,
    interval_s: float,
    seed: int,
    append: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + 1)
    if append and output.exists() and output.stat().st_size > 0:
        with output.open(newline="") as existing:
            existing_header = next(csv.reader(existing), [])
        if existing_header != HEADER:
            raise ValueError(
                f"Cannot append: {output} does not use the live schema"
            )
    mode = "a" if append and output.exists() else "w"
    needs_header = mode == "w" or output.stat().st_size == 0
    print("GENERATED INPUT STREAM — not ESP32 or LSTM output")
    print(f"Schema: {', '.join(HEADER)}")
    print(f"Output: {output}")
    with output.open(mode, newline="") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(HEADER)
            handle.flush()
        for index, bpm in enumerate(bpms):
            timestamp_us = int(time.time() * 1_000_000)
            probability = round(rng.uniform(0.88, 0.98), 6)
            writer.writerow(
                [
                    timestamp_us,
                    1,
                    probability,
                    probability,
                    round(bpm, 2),
                    "",
                    0,
                ]
            )
            handle.flush()
            print(f"second={index:03d} generated_bpm={bpm:.0f}")
            if index + 1 < len(bpms) and interval_s > 0:
                time.sleep(interval_s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate only the live-prediction input stream; all agent output "
            "must come from the real configured Groq model."
        )
    )
    parser.add_argument("--output", default=str(DEFAULT_CSV))
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--spikes", type=int, default=2)
    parser.add_argument("--spike-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    bpms = build_bpm_stream(
        args.seconds,
        args.spikes,
        args.spike_length,
        args.seed,
    )
    write_stream(
        Path(args.output),
        bpms,
        interval_s=max(0.0, args.interval_seconds),
        seed=args.seed,
        append=args.append,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
