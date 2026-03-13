#!/usr/bin/env python3
"""
Build training_main.csv from:
  - csi_packets.csv
  - bpm_stream.csv
  - markers.csv

Implements:
  - packet-gated windows (exactly 1600 packets / ~20s at 80 Hz)
  - Pulse-Fi-style filtering per subcarrier
  - overlap-weighted BPM labels
  - staleness gate for strict time alignment
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, help="Directory with raw CSV files.")
    parser.add_argument("--output", default=None, help="Output path for training_main.csv.")
    parser.add_argument("--window-size", type=int, default=1600)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--max-staleness-ms", type=float, default=2000.0)
    parser.add_argument("--session-id", default="session_001")
    parser.add_argument("--day-id", default="day_1")
    return parser.parse_args()


def load_csi(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ts, seq, rssi, csi_len = [], [], [], []
    amps = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        amp_cols = [f"amp_sc{i:02d}" for i in range(64)]
        for row in reader:
            try:
                ts.append(int(row["rx_ts_us"]))
                seq.append(int(row["seq"]))
                rssi.append(float(row["rssi"]))
                csi_len.append(float(row["csi_len"]))
                amps.append([float(row[c]) for c in amp_cols])
            except (KeyError, ValueError):
                continue
    return (
        np.asarray(ts, dtype=np.int64),
        np.asarray(seq, dtype=np.int64),
        np.asarray(rssi, dtype=np.float32),
        np.asarray(csi_len, dtype=np.float32),
        np.asarray(amps, dtype=np.float32),
    )


def load_bpm(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts, bpm, valid = [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts.append(int(row["rx_ts_us"]))
                bpm.append(float(row["bpm_value"]))
                valid.append(int(row["bpm_valid"]))
            except (KeyError, ValueError):
                continue
    return (
        np.asarray(ts, dtype=np.int64),
        np.asarray(bpm, dtype=np.float32),
        np.asarray(valid, dtype=np.int8),
    )


def load_markers(path: str) -> List[Tuple[int, str]]:
    if not os.path.exists(path):
        return []
    markers = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row["marker_ts_us"])
                event = row["event"].strip()
            except (KeyError, ValueError):
                continue
            markers.append((ts, event))
    markers.sort(key=lambda x: x[0])
    return markers


def build_label_intervals(markers: List[Tuple[int, str]], csi_start_ts: int, csi_end_ts: int) -> List[Tuple[int, int, int]]:
    """
    Returns intervals as (start_ts, end_ts, class_label).
    class_label: 0=stable_nonhuman, 1=human

    Human intervals come from hs/he marker pairs.
    Stable intervals are automatically derived from gaps between human intervals
    (before first human, between human blocks, after last human).
    """
    # Collect human intervals from hs/he pairs.
    human_intervals = []
    open_human = None

    for ts, event in markers:
        if event == "human_start":
            if open_human is None:
                open_human = ts
        elif event == "human_end":
            if open_human is not None and ts > open_human:
                human_intervals.append((open_human, ts, 1))
                open_human = None

    # Close any unclosed human interval at end of data.
    if open_human is not None and csi_end_ts > open_human:
        human_intervals.append((open_human, csi_end_ts, 1))

    human_intervals.sort(key=lambda x: x[0])

    # Derive stable intervals from gaps between human intervals.
    intervals = list(human_intervals)
    cursor = csi_start_ts

    for h_start, h_end, _ in human_intervals:
        if h_start > cursor:
            intervals.append((cursor, h_start, 0))  # stable gap
        cursor = max(cursor, h_end)

    # Stable interval after the last human block.
    if cursor < csi_end_ts:
        intervals.append((cursor, csi_end_ts, 0))

    intervals.sort(key=lambda x: x[0])
    return intervals


def class_for_window(ws: int, we: int, intervals: List[Tuple[int, int, int]]) -> int:
    dur_by_class: Dict[int, int] = {0: 0, 1: 0}
    for start, end, cl in intervals:
        overlap = min(we, end) - max(ws, start)
        if overlap > 0:
            dur_by_class[cl] += int(overlap)
    if dur_by_class[0] == 0 and dur_by_class[1] == 0:
        return -1
    return 1 if dur_by_class[1] >= dur_by_class[0] else 0


def estimate_fs(ts_us: np.ndarray) -> float:
    if len(ts_us) < 2:
        return 80.0
    dt = np.diff(ts_us).astype(np.float64) / 1_000_000.0
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 80.0
    fs = 1.0 / np.median(dt)
    return float(np.clip(fs, 20.0, 200.0))


def process_window(window_arr: np.ndarray, fs: float = 80.0) -> np.ndarray:
    """5-stage CSI processing: DC removal, Butterworth bandpass, SavGol
    smoothing (all via filtfilt zero-phase), per-window amplitude normalization.

    Exact match with live inference pipeline.
    """
    x = window_arr.astype(np.float64).copy()
    nyq = 0.5 * fs
    low = 0.8 / nyq
    high = 2.17 / nyq
    use_band = 0 < low < high < 1
    if use_band:
        b, a = butter(3, [low, high], btype="bandpass")

    n_sc = window_arr.shape[1] if hasattr(window_arr, 'shape') else 64
    out = np.zeros_like(x, dtype=np.float32)
    for sc in range(n_sc):
        sig = x[:, sc]
        sig = sig - np.mean(sig)
        if use_band:
            try:
                sig = filtfilt(b, a, sig)
            except ValueError:
                pass
        if len(sig) >= 15:
            sig = savgol_filter(sig, 15, 3)
        out[:, sc] = sig.astype(np.float32)

    g_std = np.std(out)
    if g_std > 1e-8:
        out /= g_std

    return out


def compute_periodicity_vector(processed: np.ndarray, fs: float) -> np.ndarray:
    """Per-subcarrier autocorrelation peak strength with second-cycle validation.

    Returns a (n_sc,) vector where each element encodes how periodic the
    corresponding subcarrier's signal is in the HR lag range (42-180 BPM).
    Filter ringing from static objects decays after one cycle, so requiring
    a second-cycle peak separates true heartbeats from transient disturbances.
    """
    n_samples, n_sc = processed.shape
    lag_lo = max(1, int(fs * 60.0 / 180.0))
    lag_hi = min(n_samples // 2, int(fs * 60.0 / 42.0))
    periodicity = np.zeros(n_sc, dtype=np.float32)

    if lag_hi <= lag_lo:
        return periodicity

    for sc in range(n_sc):
        sig = processed[:, sc].astype(np.float64)
        energy = np.dot(sig, sig)
        if energy < 1e-12:
            continue

        n_fft = 1
        while n_fft < 2 * n_samples:
            n_fft <<= 1
        sig_f = np.fft.rfft(sig, n=n_fft)
        acf = np.fft.irfft(sig_f * np.conj(sig_f), n=n_fft)[:n_samples] / energy

        acf_hr = acf[lag_lo:lag_hi + 1]
        if len(acf_hr) < 3:
            continue

        peak_idx = int(np.argmax(acf_hr))
        peak_val = float(acf_hr[peak_idx])
        if peak_val < 0.05:
            continue

        abs_lag = lag_lo + peak_idx
        double_lag = 2 * abs_lag
        if double_lag < n_samples:
            second_peak = float(acf[double_lag])
            if second_peak < peak_val * 0.3:
                peak_val *= 0.3
        else:
            peak_val *= 0.5

        periodicity[sc] = peak_val

    return periodicity


def overlap_weighted_bpm(
    ws: int,
    we: int,
    bpm_ts: np.ndarray,
    bpm_val: np.ndarray,
    bpm_valid: np.ndarray,
) -> Tuple[float, bool]:
    if len(bpm_ts) == 0 or ws >= we:
        return math.nan, False

    total_w = 0.0
    weighted_sum = 0.0

    for i in range(len(bpm_ts)):
        if bpm_valid[i] != 1:
            continue
        seg_start = int(bpm_ts[i])
        seg_end = int(bpm_ts[i + 1]) if i + 1 < len(bpm_ts) else we
        if seg_end <= seg_start:
            continue

        overlap_start = max(ws, seg_start)
        overlap_end = min(we, seg_end)
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue

        w = float(overlap)
        total_w += w
        weighted_sum += w * float(bpm_val[i])

    if total_w <= 0:
        return math.nan, False
    return weighted_sum / total_w, True


def label_staleness_max_us(window_ts: np.ndarray, bpm_ts: np.ndarray, bpm_valid: np.ndarray) -> float:
    valid_ts = bpm_ts[bpm_valid == 1]
    if len(valid_ts) == 0:
        return float("inf")
    idx = np.searchsorted(valid_ts, window_ts, side="right") - 1
    covered = idx >= 0
    if not np.any(covered):
        return float("inf")
    stale = np.full(len(window_ts), 0.0)
    stale[covered] = (window_ts[covered] - valid_ts[idx[covered]]).astype(np.float64)
    # Use 95th percentile of covered packets instead of max to ignore
    # pre-BPM edge packets that would otherwise poison the whole window.
    return float(np.percentile(stale[covered], 95))


def main() -> None:
    args = parse_args()
    session_dir = args.session_dir
    output_path = args.output or os.path.join(session_dir, "training_main.csv")

    csi_path = os.path.join(session_dir, "csi_packets.csv")
    bpm_path = os.path.join(session_dir, "bpm_stream.csv")
    markers_path = os.path.join(session_dir, "markers.csv")

    ts, seq, rssi, csi_len, amps = load_csi(csi_path)
    bpm_ts, bpm_val, bpm_valid = load_bpm(bpm_path)
    markers = load_markers(markers_path)

    if len(ts) < args.window_size:
        raise RuntimeError("Not enough CSI packets for one full window.")

    if len(ts) < args.window_size:
        raise RuntimeError("Not enough CSI packets for one full window.")

    # We will estimate fs dynamically per window, matching the live inference.
    # Alternatively, we could define a global fs. Let's stick to global to avoid recomputing if too slow, or per-window if we want an exact match.
    # In live dashboard:
    # fs = float(np.clip(1.0 / np.median(dt), 20.0, 200.0)) if len(dt) > 0 else 80.0


    intervals = build_label_intervals(markers, int(ts[0]), int(ts[-1]))
    if len(intervals) == 0:
        print("Warning: no stable/human intervals from markers. Windows may be dropped.")

    max_staleness_us = args.max_staleness_ms * 1000.0

    headers = [
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

    kept = 0
    dropped_no_class = 0
    dropped_sync = 0

    with open(output_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(headers)

        n = len(ts)
        w = args.window_size
        s = args.stride
        for start in range(0, n - w + 1, s):
            end = start + w
            ws = int(ts[start])
            we = int(ts[end - 1])

            class_label = class_for_window(ws, we, intervals)
            if class_label < 0:
                dropped_no_class += 1
                continue

            y_bpm, has_overlap = overlap_weighted_bpm(ws, we, bpm_ts, bpm_val, bpm_valid)
            stale_us = label_staleness_max_us(ts[start:end], bpm_ts, bpm_valid)

            # Rule:
            # - Human window (class=1) must have valid BPM overlap and freshness.
            # - Stable window (class=0) is kept even without BPM; regression is masked out.
            if class_label == 1:
                sync_ok = int(has_overlap and stale_us <= max_staleness_us and np.isfinite(y_bpm))
                if sync_ok == 0:
                    dropped_sync += 1
                    continue
                regression_mask = 1
            else:
                sync_ok = 1
                y_bpm = math.nan
                regression_mask = 0
                
            # Exact pipeline parity: Process per window instead of global
            window_ts = ts[start:end]
            dt = np.diff(window_ts).astype(np.float64) / 1_000_000.0
            dt = dt[dt > 0]
            if len(dt) > 0:
                fs = float(np.clip(1.0 / np.median(dt), 20.0, 200.0))
            else:
                fs = 80.0
                
            window_amps = amps[start:end, :]
            proc_window = process_window(window_amps, fs=fs)
            
            features = np.round(proc_window, 4).tolist()
            features_json = json.dumps(features, separators=(",", ":"))

            period_vec = compute_periodicity_vector(proc_window, fs)
            period_json = json.dumps(
                np.round(period_vec, 5).tolist(), separators=(",", ":")
            )

            writer.writerow(
                [
                    args.day_id,
                    args.session_id,
                    ws,
                    we,
                    class_label,
                    round(float(y_bpm), 4),
                    regression_mask,
                    round(float(stale_us / 1000.0), 3),
                    sync_ok,
                    w,
                    round(float(np.mean(rssi[start:end])), 3),
                    round(float(np.mean(csi_len[start:end])), 3),
                    features_json,
                    period_json,
                ]
            )
            kept += 1

    print(f"Saved: {output_path}")
    print(
        f"windows_kept={kept} dropped_no_class={dropped_no_class} "
        f"dropped_sync={dropped_sync}"
    )


if __name__ == "__main__":
    main()
