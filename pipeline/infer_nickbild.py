#!/usr/bin/env python3
"""
Live inference using nickbild-style model.

Reads CSI_PKT from serial, buffers 100 packets, processes with Pulse-Fi pipeline,
predicts BPM. Compatible with PulseFi receiver format.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque

import numpy as np
import serial
from scipy.signal import butter, filtfilt, savgol_filter
from tensorflow import keras

WINDOW_SIZE = 100
SC_COUNT = 64


def process_window(amps: np.ndarray, fs: float) -> np.ndarray:
    """Pulse-Fi: DC removal, bandpass 0.8-2.17 Hz, Savitzky-Golay."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", required=True, help="Serial port")
    parser.add_argument("-m", "--model", default="models/nickbild/csi_hr.keras")
    parser.add_argument("-b", "--baud", type=int, default=921600)
    args = parser.parse_args()

    model = keras.models.load_model(args.model, safe_mode=False)
    ser = serial.Serial(port=args.port, baudrate=args.baud, timeout=1)
    buf: deque[tuple[int, np.ndarray]] = deque(maxlen=WINDOW_SIZE)

    print("Waiting for 100 CSI packets...", file=sys.stderr)
    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        parts = line.split(",")
        if parts[0] != "CSI_PKT" or len(parts) < 5 + SC_COUNT:
            continue
        try:
            ts = int(parts[1])
            amps = np.array([float(parts[5 + i]) for i in range(SC_COUNT)], dtype=np.float32)
        except (ValueError, IndexError):
            continue
        buf.append((ts, amps))
        if len(buf) < WINDOW_SIZE:
            continue
        ts_arr = np.array([x[0] for x in buf])
        amp_arr = np.array([x[1] for x in buf])
        dt = np.diff(ts_arr).astype(np.float64) / 1e6
        dt = dt[dt > 0]
        fs = float(np.clip(1.0 / np.median(dt), 15.0, 120.0)) if len(dt) > 0 else 20.0
        processed = process_window(amp_arr, fs)
        x_in = processed.reshape(1, WINDOW_SIZE, SC_COUNT)
        pred = float(model.predict(x_in, verbose=0)[0, 0])
        print(f"{pred:.1f}")


if __name__ == "__main__":
    main()
