"""Deterministic measurement guard and state policy.

The LLM explains and contextualizes readings. It cannot downgrade this guard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from statistics import fmean, pvariance
from typing import Iterable

from agentic.models import Reading


STATES = ("no_person", "normal", "recovering", "watch", "urgent")
SEVERITY = {
    "no_person": -1,
    "normal": 0,
    "recovering": 1,
    "watch": 2,
    "urgent": 3,
}


@dataclass(frozen=True)
class PolicyConfig:
    presence_threshold: float = 0.55
    plausible_bpm_min: float = 30.0
    plausible_bpm_max: float = 220.0
    normal_bpm_low: float = 60.0
    normal_bpm_high: float = 100.0
    watch_bpm_low: float = 50.0
    watch_bpm_high: float = 120.0
    urgent_bpm_low: float = 40.0
    urgent_bpm_high: float = 150.0
    exercise_watch_bpm_high: float = 180.0
    exercise_urgent_bpm_high: float = 200.0
    sleep_normal_bpm_low: float = 50.0
    sleep_watch_bpm_low: float = 45.0
    sleep_urgent_bpm_low: float = 35.0
    abnormal_sustain_count: int = 2
    urgent_sustain_count: int = 2
    recovery_count: int = 5
    sudden_delta_bpm: float = 30.0
    sudden_window_s: float = 5.0
    disagreement_bpm: float = 20.0
    disagreement_count: int = 3
    baseline_deviation_bpm: float = 25.0
    baseline_sustain_count: int = 2


@dataclass(frozen=True)
class GuardFeatures:
    bpm: float | None
    rolling_mean: float | None
    rolling_variance: float | None
    slope_bpm_per_s: float | None
    sudden_delta_bpm: float | None
    baseline_bpm: float | None
    baseline_deviation_bpm: float | None
    consecutive_outside_normal: int
    consecutive_high: int
    consecutive_low: int
    consecutive_urgent_high: int
    consecutive_urgent_low: int
    consecutive_sensor_disagreement: int
    consecutive_recovery: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GuardResult:
    state: str
    reasons: tuple[str, ...]
    measurement_valid: bool
    features: GuardFeatures

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "reasons": list(self.reasons),
            "measurement_valid": self.measurement_valid,
            "features": self.features.to_dict(),
        }


def _reliable(reading: Reading, config: PolicyConfig) -> bool:
    return (
        reading.human
        and isfinite(reading.presence)
        and reading.presence >= config.presence_threshold
        and reading.bpm is not None
        and isfinite(reading.bpm)
        and config.plausible_bpm_min <= reading.bpm <= config.plausible_bpm_max
    )


def _count_suffix(readings: list[Reading], predicate) -> int:
    count = 0
    for reading in reversed(readings):
        if not predicate(reading):
            break
        count += 1
    return count


def _slope(readings: list[Reading]) -> float | None:
    if len(readings) < 2:
        return None
    first = readings[0]
    last = readings[-1]
    elapsed = (last.ts_us - first.ts_us) / 1_000_000.0
    if elapsed <= 0 or first.bpm is None or last.bpm is None:
        return None
    return (last.bpm - first.bpm) / elapsed


def _window(
    history: Iterable[Reading],
    current: Reading,
    config: PolicyConfig,
) -> list[Reading]:
    cutoff = current.ts_us - int(config.sudden_window_s * 1_000_000)
    ordered = sorted(
        (
            reading
            for reading in history
            if cutoff <= reading.ts_us <= current.ts_us
            and _reliable(reading, config)
        ),
        key=lambda reading: reading.ts_us,
    )
    if not ordered or ordered[-1].ts_us != current.ts_us:
        if _reliable(current, config):
            ordered.append(current)
    return ordered


def evaluate_guard(
    history: Iterable[Reading],
    current: Reading,
    *,
    config: PolicyConfig | None = None,
    baseline_bpm: float | None = None,
    episode_active: bool = False,
) -> GuardResult:
    config = config or PolicyConfig()
    reliable_history = sorted(
        (reading for reading in history if _reliable(reading, config)),
        key=lambda reading: reading.ts_us,
    )
    if (
        not reliable_history
        or reliable_history[-1].ts_us != current.ts_us
        or reliable_history[-1].source != current.source
    ) and _reliable(current, config):
        reliable_history.append(current)

    empty = GuardFeatures(
        bpm=current.bpm,
        rolling_mean=None,
        rolling_variance=None,
        slope_bpm_per_s=None,
        sudden_delta_bpm=None,
        baseline_bpm=baseline_bpm,
        baseline_deviation_bpm=None,
        consecutive_outside_normal=0,
        consecutive_high=0,
        consecutive_low=0,
        consecutive_urgent_high=0,
        consecutive_urgent_low=0,
        consecutive_sensor_disagreement=0,
        consecutive_recovery=0,
    )

    if not current.human or current.presence < config.presence_threshold:
        return GuardResult(
            state="no_person",
            reasons=("presence_not_reliable",),
            measurement_valid=False,
            features=empty,
        )
    if (
        current.bpm is None
        or not isfinite(current.bpm)
        or not config.plausible_bpm_min
        <= current.bpm
        <= config.plausible_bpm_max
    ):
        return GuardResult(
            state="watch",
            reasons=("invalid_bpm_measurement",),
            measurement_valid=False,
            features=empty,
        )

    recent = reliable_history[-20:]
    bpms = [float(reading.bpm) for reading in recent if reading.bpm is not None]
    short_window = _window(reliable_history, current, config)
    normal_low = (
        config.sleep_normal_bpm_low
        if current.activity == "sleep"
        else config.normal_bpm_low
    )
    normal_high = (
        config.exercise_watch_bpm_high
        if current.activity == "exercise"
        else config.normal_bpm_high
    )
    watch_low = (
        config.sleep_watch_bpm_low
        if current.activity == "sleep"
        else config.watch_bpm_low
    )
    watch_high = (
        config.exercise_watch_bpm_high
        if current.activity == "exercise"
        else config.watch_bpm_high
    )
    urgent_low_threshold = (
        config.sleep_urgent_bpm_low
        if current.activity == "sleep"
        else config.urgent_bpm_low
    )
    urgent_high_threshold = (
        config.exercise_urgent_bpm_high
        if current.activity == "exercise"
        else config.urgent_bpm_high
    )
    sudden_delta = None
    if len(short_window) >= 2 and short_window[0].bpm is not None:
        sudden_delta = current.bpm - short_window[0].bpm

    outside = _count_suffix(
        reliable_history,
        lambda item: item.bpm is not None
        and not normal_low <= item.bpm <= normal_high,
    )
    high = _count_suffix(
        reliable_history,
        lambda item: item.bpm is not None and item.bpm >= watch_high,
    )
    low = _count_suffix(
        reliable_history,
        lambda item: item.bpm is not None and item.bpm <= watch_low,
    )
    urgent_high = _count_suffix(
        reliable_history,
        lambda item: item.bpm is not None
        and item.bpm >= urgent_high_threshold,
    )
    urgent_low = _count_suffix(
        reliable_history,
        lambda item: item.bpm is not None
        and item.bpm <= urgent_low_threshold,
    )
    disagreement = _count_suffix(
        reliable_history,
        lambda item: item.sensor_valid
        and item.sensor_bpm is not None
        and item.bpm is not None
        and abs(item.bpm - item.sensor_bpm) >= config.disagreement_bpm,
    )
    recovery = _count_suffix(
        reliable_history,
        lambda item: item.bpm is not None
        and normal_low <= item.bpm <= normal_high,
    )
    baseline_deviation = (
        abs(current.bpm - baseline_bpm) if baseline_bpm is not None else None
    )
    baseline_deviation_count = 0
    if baseline_bpm is not None and current.activity == "rest":
        baseline_deviation_count = _count_suffix(
            reliable_history,
            lambda item: item.bpm is not None
            and abs(item.bpm - baseline_bpm) >= config.baseline_deviation_bpm,
        )

    features = GuardFeatures(
        bpm=current.bpm,
        rolling_mean=fmean(bpms) if bpms else None,
        rolling_variance=pvariance(bpms) if len(bpms) > 1 else 0.0,
        slope_bpm_per_s=_slope(short_window),
        sudden_delta_bpm=sudden_delta,
        baseline_bpm=baseline_bpm,
        baseline_deviation_bpm=baseline_deviation,
        consecutive_outside_normal=outside,
        consecutive_high=high,
        consecutive_low=low,
        consecutive_urgent_high=urgent_high,
        consecutive_urgent_low=urgent_low,
        consecutive_sensor_disagreement=disagreement,
        consecutive_recovery=recovery,
    )

    reasons: list[str] = []
    state = "normal"
    serious_symptoms = {
        "chest pain",
        "fainting",
        "shortness of breath",
    }.intersection(current.symptoms)
    if serious_symptoms:
        state = "urgent"
        reasons.append("user_reported_serious_symptom")
    if urgent_high >= config.urgent_sustain_count:
        state = "urgent"
        reasons.append("sustained_urgent_high_bpm")
    if urgent_low >= config.urgent_sustain_count:
        state = "urgent"
        reasons.append("sustained_urgent_low_bpm")
    if state != "urgent":
        if current.symptoms:
            state = "watch"
            reasons.append("user_reported_symptom")
        if high >= config.abnormal_sustain_count:
            state = "watch"
            reasons.append("sustained_high_bpm")
        if low >= config.abnormal_sustain_count:
            state = "watch"
            reasons.append("sustained_low_bpm")
        if outside >= config.abnormal_sustain_count:
            state = "watch"
            reasons.append("outside_resting_band")
        if (
            sudden_delta is not None
            and abs(sudden_delta) >= config.sudden_delta_bpm
            and (
                not episode_active
                or not normal_low <= current.bpm <= normal_high
            )
        ):
            state = "watch"
            reasons.append("sudden_bpm_change")
        if disagreement >= config.disagreement_count:
            state = "watch"
            reasons.append("sensor_disagreement")
        if baseline_deviation_count >= config.baseline_sustain_count:
            state = "watch"
            reasons.append("personal_baseline_deviation")

    if episode_active and state == "normal" and recovery < config.recovery_count:
        state = "recovering"
        reasons.append("recovery_not_yet_confirmed")
    elif episode_active and state == "normal":
        reasons.append("recovery_confirmed")
    elif state == "normal":
        reasons.append("within_expected_band")

    return GuardResult(
        state=state,
        reasons=tuple(dict.fromkeys(reasons)),
        measurement_valid=True,
        features=features,
    )


def merge_states(guard_state: str, model_state: str) -> str:
    if guard_state not in STATES:
        raise ValueError(f"Unknown guard state: {guard_state}")
    if model_state not in STATES:
        raise ValueError(f"Unknown model state: {model_state}")
    if guard_state == "no_person":
        return "no_person"
    if model_state == "no_person":
        return guard_state
    return (
        guard_state
        if SEVERITY[guard_state] >= SEVERITY[model_state]
        else model_state
    )
