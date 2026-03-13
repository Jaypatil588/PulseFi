#!/usr/bin/env python3
"""
Time-stretch augmentation: resample existing human CSI windows to simulate
different heart rate ranges while preserving realistic waveform shape, noise
characteristics, and multi-subcarrier correlations.

Stretching a 100 BPM signal by factor 1.33 shifts the heartbeat periodicity
to 75 BPM.  The LSTM then learns to map both periodicities to the correct
BPM output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.signal import resample as sp_resample

from build_training_csv import (
    load_csi,
    load_bpm,
    load_markers,
    build_label_intervals,
    class_for_window,
    overlap_weighted_bpm,
    label_staleness_max_us,
    process_window,
    compute_periodicity_vector,
)

WINDOW_SIZE = 1600
SC_COUNT = 64

HEADERS = [
    "day_id",
    "session_id",
    "win_start_ts_us",
    "win_end_ts_us",
    "class_label",
    "y_bpm",
    "regression_mask",
    "label_staleness_ms_max",
    "sync_ok",
    "window_packet_count",
    "rssi_mean",
    "csi_len_mean",
    "csi_features_json",
    "periodicity_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate time-stretched synthetic training data."
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Root data directory containing session folders.",
    )
    parser.add_argument(
        "--target-bpms", default="75,80,85",
        help="Comma-separated target BPM centers (default: 75,80,85).",
    )
    parser.add_argument("--window-size", type=int, default=2400)
    parser.add_argument("--stride", type=int, default=300)
    parser.add_argument("--max-staleness-ms", type=float, default=2000.0)
    return parser.parse_args()


def find_human_sessions(data_root: Path) -> List[Path]:
    """Return session directories that contain human_start markers."""
    sessions: List[Path] = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        csi = d / "csi_packets.csv"
        bpm = d / "bpm_stream.csv"
        markers = d / "markers.csv"
        if not (csi.exists() and bpm.exists() and markers.exists()):
            continue
        with open(markers) as f:
            text = f.read()
        if "human_start" in text:
            sessions.append(d)
    return sessions


def collect_source_windows(
    session_dir: Path,
    window_size: int,
    stride: int,
    max_staleness_ms: float,
) -> Tuple[List[Tuple[np.ndarray, float, float, float]], List[float]]:
    """Extract raw amplitude windows and BPM labels from a human session.

    Returns (windows, bpms) where each window entry is
    (raw_amps [W,64], fs, rssi_mean, csi_len_mean).
    """
    ts, _seq, rssi, csi_len, amps = load_csi(str(session_dir / "csi_packets.csv"))
    bpm_ts, bpm_val, bpm_valid = load_bpm(str(session_dir / "bpm_stream.csv"))
    markers = load_markers(str(session_dir / "markers.csv"))

    if len(ts) < window_size:
        return [], []

    intervals = build_label_intervals(markers, int(ts[0]), int(ts[-1]))
    max_stale_us = max_staleness_ms * 1000.0

    windows: List[Tuple[np.ndarray, float, float, float]] = []
    bpms: List[float] = []

    n = len(ts)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        ws, we = int(ts[start]), int(ts[end - 1])

        if class_for_window(ws, we, intervals) != 1:
            continue

        y_bpm, has_overlap = overlap_weighted_bpm(ws, we, bpm_ts, bpm_val, bpm_valid)
        stale = label_staleness_max_us(ts[start:end], bpm_ts, bpm_valid)

        if not (has_overlap and stale <= max_stale_us and np.isfinite(y_bpm)):
            continue

        dt = np.diff(ts[start:end]).astype(np.float64) / 1_000_000.0
        dt = dt[dt > 0]
        fs = float(np.clip(1.0 / np.median(dt), 20.0, 200.0)) if len(dt) > 0 else 80.0

        windows.append((
            amps[start:end, :].copy(),
            fs,
            float(np.mean(rssi[start:end])),
            float(np.mean(csi_len[start:end])),
        ))
        bpms.append(float(y_bpm))

    return windows, bpms


def time_stretch_window(
    raw_amps: np.ndarray,
    stretch_factor: float,
    window_size: int = 1600,
) -> np.ndarray:
    """Resample raw CSI amplitudes to shift heartbeat periodicity.

    stretch_factor > 1 → slower heart rate (lower BPM).
    Returns (window_size, n_sc) array of stretched raw amplitudes.
    """
    n_stretched = max(window_size + 1, int(window_size * stretch_factor))

    stretched = np.zeros((n_stretched, raw_amps.shape[1]), dtype=np.float64)
    for sc in range(raw_amps.shape[1]):
        stretched[:, sc] = sp_resample(raw_amps[:, sc].astype(np.float64), n_stretched)

    start = (n_stretched - window_size) // 2
    return stretched[start : start + window_size].astype(np.float32)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    targets = [int(t.strip()) for t in args.target_bpms.split(",")]

    human_sessions = find_human_sessions(data_root)
    if not human_sessions:
        print("[FAIL] No human sessions found in", data_root)
        sys.exit(1)

    print(f"Found {len(human_sessions)} human session(s):")
    for s in human_sessions:
        print(f"  {s.name}")

    all_windows: List[Tuple[np.ndarray, float, float, float]] = []
    all_bpms: List[float] = []
    for session_dir in human_sessions:
        windows, bpms = collect_source_windows(
            session_dir, args.window_size, args.stride, args.max_staleness_ms,
        )
        all_windows.extend(windows)
        all_bpms.extend(bpms)
        print(f"  {session_dir.name}: {len(windows)} human windows")

    if not all_windows:
        print("[FAIL] No valid human windows extracted.")
        sys.exit(1)

    source_median = float(np.median(all_bpms))
    print(f"\nSource pool: {len(all_windows)} windows, "
          f"median BPM = {source_median:.1f}, "
          f"range = [{min(all_bpms):.1f}, {max(all_bpms):.1f}]")

    for target in targets:
        stretch_factor = source_median / target
        session_id = f"synth_{target}bpm"
        out_dir = data_root / session_id
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "training_main.csv"

        print(f"\n=== Generating {session_id} "
              f"(stretch={stretch_factor:.3f}, target center={target}) ===")

        kept = 0
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

            for i, ((raw_amps, fs, rssi_m, csi_len_m), orig_bpm) in enumerate(
                zip(all_windows, all_bpms)
            ):
                new_bpm = orig_bpm / stretch_factor

                if not (40.0 <= new_bpm <= 200.0):
                    continue

                stretched_raw = time_stretch_window(
                    raw_amps, stretch_factor, args.window_size
                )

                proc = process_window(stretched_raw, fs=fs)
                pvec = compute_periodicity_vector(proc, fs)

                feat_json = json.dumps(
                    np.round(proc, 4).tolist(), separators=(",", ":")
                )
                period_json = json.dumps(
                    np.round(pvec, 5).tolist(), separators=(",", ":")
                )

                fake_ts = 1_000_000_000_000 + target * 1_000_000 + i * 100_000

                writer.writerow([
                    "synth",
                    session_id,
                    fake_ts,
                    fake_ts + int(args.window_size / fs * 1_000_000),
                    1,
                    round(new_bpm, 4),
                    1,
                    0.0,
                    1,
                    args.window_size,
                    round(rssi_m, 3),
                    round(csi_len_m, 3),
                    feat_json,
                    period_json,
                ])
                kept += 1

        new_bpms = [b / stretch_factor for b in all_bpms if 40 <= b / stretch_factor <= 200]
        print(f"  Wrote {kept} windows → {out_path}")
        if new_bpms:
            print(f"  BPM range: [{min(new_bpms):.1f}, {max(new_bpms):.1f}]")


if __name__ == "__main__":
    main()
