"""Readings the agent loop consumes.

A reading is the LSTM's presence flag and heart-rate estimate. Raw CSI
stays in the model file, which the database records as LSTM memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


DISCLAIMER = (
    "Wellness feedback from a research prototype. Not a medical diagnosis."
)
ALLOWED_ACTIVITIES = {"rest", "exercise", "sleep", "unknown"}
ALLOWED_SYMPTOMS = {
    "chest pain",
    "fainting",
    "shortness of breath",
    "dizziness",
    "palpitations",
}
LIVE_PREDICTION_HEADER = (
    "rx_ts_us",
    "detected_human",
    "ml_prob_human_30s",
    "ml_prob_human_30s_smooth",
    "pred_bpm_ml_display",
    "sensor_bpm",
    "sensor_bpm_valid",
)
SUPPORTED_PREDICTION_HEADERS = {
    LIVE_PREDICTION_HEADER,
    (*LIVE_PREDICTION_HEADER, "activity", "symptoms"),
}


@dataclass
class Reading:
    ts_us: int
    human: bool
    presence: float
    bpm: float | None
    sensor_bpm: float | None
    sensor_valid: bool
    source: str = "lstm"
    activity: str = "unknown"
    symptoms: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def _num(row: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        raw = str(row.get(key, "")).strip()
        if raw == "":
            continue
        try:
            value = float(raw)
            return value if isfinite(value) else None
        except ValueError:
            continue
    return None


def reading_from_row(row: dict, source: str = "lstm") -> Reading | None:
    """Accept the live-inference CSV and the older dashboard CSV."""
    if str(row.get("rx_ts_us", "")).strip() == "":
        return None
    try:
        ts_us = int(float(row["rx_ts_us"]))
    except (KeyError, TypeError, ValueError):
        return None
    human_raw = _num(row, ("detected_human", "class_pred"))
    if human_raw is None:
        return None
    presence = _num(row, ("ml_prob_human_30s_smooth", "ml_prob_human_30s", "class_prob_human"))
    if presence is None:
        presence = 1.0 if human_raw >= 1 else 0.0
    bpm = _num(row, ("pred_bpm_ml_display", "pred_bpm"))
    if bpm is not None and bpm <= 0:
        bpm = None
    sensor = _num(row, ("sensor_bpm",))
    if sensor is not None and sensor <= 0:
        sensor = None
    valid_raw = _num(row, ("sensor_bpm_valid",))
    sensor_valid = bool(valid_raw) if valid_raw is not None else sensor is not None
    activity = str(row.get("activity", "unknown") or "unknown").strip().lower()
    if activity not in ALLOWED_ACTIVITIES:
        activity = "unknown"
    symptoms_raw = str(row.get("symptoms", "") or "")
    symptoms = tuple(
        item
        for item in (
            raw_item.strip().lower()
            for raw_item in symptoms_raw.split("|")
        )
        if item in ALLOWED_SYMPTOMS
    )
    return Reading(
        ts_us=ts_us,
        human=bool(human_raw >= 1),
        presence=float(presence),
        bpm=bpm,
        sensor_bpm=sensor,
        sensor_valid=sensor_valid,
        source=source,
        activity=activity,
        symptoms=symptoms,
    )
