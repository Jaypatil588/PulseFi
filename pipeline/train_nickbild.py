#!/usr/bin/env python3
"""
Train nickbild/csi_hr-style LSTM on PulseFi data.

Uses the exact architecture from https://github.com/nickbild/csi_hr:
  Input (100, 64) - 100 packets, 64 subcarriers
  LSTM 64 -> Dropout 0.2 -> LSTM 32 -> Dropout 0.2 -> Dense 16 -> Dense 1
  MSE loss, predicts average BPM over window

Adapted for PulseFi: 64 subcarriers (vs nickbild's 192), same processing pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from tensorflow import keras


WINDOW_SIZE = 100
SC_COUNT = 64


def process_window(amps: np.ndarray, fs: float) -> np.ndarray:
    """Pulse-Fi processing: DC removal, bandpass 0.8-2.17 Hz, Savitzky-Golay.
    amps: (n_samples, 64) - process each subcarrier along time axis."""
    x = amps.astype(np.float64).copy()
    nyq = 0.5 * fs
    low, high = 0.8 / nyq, 2.17 / nyq
    if not (0 < low < high < 1):
        return x.astype(np.float32)
    b, a = butter(3, [low, high], btype="bandpass")
    out = np.zeros_like(x, dtype=np.float32)
    for sc in range(SC_COUNT):
        sig = x[:, sc] - np.mean(x[:, sc])
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


def load_session(session_dir: Path, stride: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load CSI and BPM from a session. Returns (processed_windows, bpm_labels, fs) or None."""
    csi_path = session_dir / "csi_packets.csv"
    bpm_path = session_dir / "bpm_stream.csv"
    markers_path = session_dir / "markers.csv"
    if not csi_path.exists() or not bpm_path.exists():
        return None

    # Load CSI
    ts_list, amps_list = [], []
    amp_cols = [f"amp_sc{i:02d}" for i in range(SC_COUNT)]
    with open(csi_path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ts_list.append(int(row["rx_ts_us"]))
                amps_list.append([float(row[c]) for c in amp_cols])
            except (KeyError, ValueError):
                continue
    if len(ts_list) < WINDOW_SIZE:
        return None

    ts = np.array(ts_list, dtype=np.int64)
    amps = np.array(amps_list, dtype=np.float32)  # (n_packets, 64)

    # Estimate fs
    dt = np.diff(ts).astype(np.float64) / 1e6
    dt = dt[dt > 0]
    fs = float(np.clip(1.0 / np.median(dt), 15.0, 120.0)) if len(dt) > 0 else 20.0

    # Load markers for human intervals
    human_start, human_end = None, None
    if markers_path.exists():
        with open(markers_path) as f:
            for row in csv.DictReader(f):
                ev = row.get("event", "").strip()
                t = int(row.get("marker_ts_us", 0))
                if ev == "human_start":
                    human_start = t
                elif ev == "human_end":
                    human_end = t

    # Load BPM
    bpm_ts, bpm_val, bpm_valid = [], [], []
    with open(bpm_path) as f:
        for row in csv.DictReader(f):
            try:
                bpm_ts.append(int(row["rx_ts_us"]))
                bpm_val.append(float(row["bpm_value"]))
                bpm_valid.append(int(row["bpm_valid"]))
            except (KeyError, ValueError):
                continue
    bpm_ts = np.array(bpm_ts)
    bpm_val = np.array(bpm_val)
    bpm_valid = np.array(bpm_valid)

    # Build sliding windows
    X_windows = []
    y_bpm = []

    for i in range(0, len(amps) - WINDOW_SIZE, stride):
        win_ts_start = ts[i]
        win_ts_end = ts[i + WINDOW_SIZE - 1]

        # Only use windows in human interval
        if human_start is not None and human_end is not None:
            if win_ts_end < human_start or win_ts_start > human_end:
                continue

        # Get BPM for this window: interpolate from bpm stream
        win_center = (win_ts_start + win_ts_end) / 2
        valid_mask = bpm_valid == 1
        valid_bpm = bpm_val[valid_mask]
        valid_ts = bpm_ts[valid_mask]
        if len(valid_bpm) < 3:
            continue
        # Average BPM over window (nearest valid readings)
        idx = np.searchsorted(valid_ts, win_center)
        if idx <= 0:
            avg_bpm = valid_bpm[0]
        elif idx >= len(valid_bpm):
            avg_bpm = valid_bpm[-1]
        else:
            avg_bpm = (valid_bpm[idx - 1] + valid_bpm[idx]) / 2.0
        if not (35 < avg_bpm < 220):
            continue

        # Process window: (WINDOW_SIZE, 64)
        win_amps = amps[i : i + WINDOW_SIZE]
        processed = process_window(win_amps, fs)
        X_windows.append(processed)
        y_bpm.append(avg_bpm)

    if len(X_windows) == 0:
        return None
    return (
        np.array(X_windows, dtype=np.float32),
        np.array(y_bpm, dtype=np.float32),
        fs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data", help="Root dir with session folders")
    parser.add_argument("--outdir", default="models/nickbild", help="Output dir for model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=10, help="Window stride (1=all, 10=faster)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Find human sessions (real + synth)
    all_X, all_y = [], []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        # Real sessions: need csi_packets + markers with human_start
        if not d.name.startswith("synth"):
            markers = d / "markers.csv"
            if not markers.exists():
                continue
            with open(markers) as f:
                if "human_start" not in f.read():
                    continue
            result = load_session(d, stride=args.stride)
            if result is None:
                continue
            X, y, _ = result
            all_X.append(X)
            all_y.append(y)
            print(f"  {d.name}: {len(X)} windows")
            continue
        # Synth: load from training_main.csv (1600-packet windows), take first 100
        csv_path = d / "training_main.csv"
        if not csv_path.exists():
            continue
        X_s, y_s = [], []
        with open(csv_path) as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    feat = np.array(json.loads(row["csi_features_json"]), dtype=np.float32)
                    if feat.shape[0] < WINDOW_SIZE:
                        continue
                    bpm = float(row["y_bpm"])
                    if not (35 < bpm < 220):
                        continue
                    X_s.append(feat[:WINDOW_SIZE])
                    y_s.append(bpm)
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
        if X_s:
            all_X.append(np.array(X_s))
            all_y.append(np.array(y_s))
            print(f"  {d.name}: {len(X_s)} windows (from preprocessed)")

    if not all_X:
        raise SystemExit("No human session data found.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\nTotal: {len(X)} windows, BPM range [{y.min():.0f}, {y.max():.0f}]")

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Nickbild architecture (adapted for 64 subcarriers)
    main_input = keras.Input(shape=(WINDOW_SIZE, SC_COUNT), name="main_input")
    layers = keras.layers.LSTM(64, return_sequences=True, name="lstm_1")(main_input)
    layers = keras.layers.Dropout(0.2, name="dropout_1")(layers)
    layers = keras.layers.LSTM(32, name="lstm_2")(layers)
    layers = keras.layers.Dropout(0.2, name="dropout_2")(layers)
    layers = keras.layers.Dense(16, activation="relu", name="dense_1")(layers)
    hr_output = keras.layers.Dense(1, name="hr_output")(layers)
    model = keras.Model(inputs=main_input, outputs=hr_output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    model.summary()

    # Train
    model.fit(
        X, y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.val_split,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
            ),
        ],
    )

    model_path = outdir / "csi_hr.keras"
    model.save(model_path)
    print(f"\nSaved model → {model_path}")


if __name__ == "__main__":
    main()
