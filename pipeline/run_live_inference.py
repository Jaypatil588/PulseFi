#!/usr/bin/env python3
"""
Run live two-stage inference from RX serial stream.

Input lines expected from RX sketch:
  CSI_PKT,rx_ts_us,seq,rssi,csi_len,amp_sc00..amp_sc63
  BPM,rx_ts_us,bpm_value,bpm_valid,sensor_age_ms
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import serial
from scipy.signal import butter, filtfilt, savgol_filter
from tensorflow import keras

WINDOW_SIZE = 1600
SC_COUNT = 64



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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="RX serial port.")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--classifier-model", required=True, help="30s ML classifier model (.keras).")
    parser.add_argument("--regressor-model", required=True)
    parser.add_argument("--out-csv", required=True, help="Runtime predictions output CSV.")
    parser.add_argument("--infer-stride", type=int, default=10, help="30s ML infer every N CSI packets.")
    parser.add_argument("--human-threshold", type=float, default=0.5, help="ML classifier threshold (if enabled).")
    parser.add_argument("--log-period-s", type=float, default=0.5, help="CSV logging period.")
    return parser.parse_args()


def process_window(window_arr: np.ndarray, fs: float = 80.0) -> np.ndarray:
    """5-stage CSI processing: DC removal, Butterworth bandpass, SavGol
    smoothing (all via filtfilt zero-phase), per-window amplitude normalization."""
    x = window_arr.astype(np.float64).copy()
    nyq = 0.5 * fs
    low = 0.8 / nyq
    high = 2.17 / nyq
    use_band = 0 < low < high < 1
    if use_band:
        b, a = butter(3, [low, high], btype="bandpass")

    out = np.zeros_like(x, dtype=np.float32)
    for sc in range(SC_COUNT):
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


@dataclass
class LiveState:
    lock: threading.Lock

    # Updated by serial reader
    latest_csi_ts_us: int = 0
    csi_counter: int = 0
    sensor_bpm: Optional[float] = None
    sensor_bpm_valid: int = 0

    # Updated by 30s ML thread
    ml_ts_us: int = 0
    ml_prob_human_30s: Optional[float] = None
    ml_prob_human_30s_smooth: Optional[float] = None
    ml_detected: int = 0
    pred_bpm_ml: float = 0.0


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    clf = keras.models.load_model(args.classifier_model, safe_mode=False)
    reg = keras.models.load_model(args.regressor_model, safe_mode=False)

    ser = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=1,
    )

    # Keep more than 30s so the 5s detector can work even while ML is running.
    csi_buf: Deque[Tuple[int, np.ndarray]] = deque(maxlen=WINDOW_SIZE * 3)
    buf_lock = threading.Lock()
    state = LiveState(lock=threading.Lock())
    data_event = threading.Event()
    stop_event = threading.Event()

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rx_ts_us",
                "detected_human",
                "ml_prob_human_30s",
                "ml_prob_human_30s_smooth",
                "pred_bpm_ml_display",
                "sensor_bpm",
                "sensor_bpm_valid",
            ]
        )

        def ml_worker() -> None:
            last_seen = 0
            bpm_buf: List[float] = []
            smooth_bpm = 0.0

            while not stop_event.is_set():
                data_event.wait(timeout=0.25)
                data_event.clear()

                with state.lock:
                    cur = state.csi_counter
                if cur - last_seen < int(args.infer_stride):
                    continue
                last_seen = cur

                with buf_lock:
                    if len(csi_buf) < WINDOW_SIZE:
                        continue
                    tail = list(csi_buf)[-WINDOW_SIZE:]
                ts_window = np.asarray([x[0] for x in tail], dtype=np.int64)
                amp_window = np.asarray([x[1] for x in tail], dtype=np.float32)

                dt = np.diff(ts_window).astype(np.float64) / 1_000_000.0
                dt = dt[dt > 0]
                fs = float(np.clip(1.0 / np.median(dt), 20.0, 200.0)) if len(dt) > 0 else 80.0

                feat = process_window(amp_window, fs=fs)
                x = feat.reshape(1, WINDOW_SIZE, SC_COUNT)

                pred_bpm_ml = float(reg.predict(x, verbose=0)[0, 0]) * 100.0

                # Smooth BPM (median buffer rejects outliers, then EMA).
                EMA_BPM = 0.15
                BPM_BUF_SIZE = 15
                if 30 < pred_bpm_ml < 200:
                    bpm_buf.append(pred_bpm_ml)
                    if len(bpm_buf) > BPM_BUF_SIZE:
                        bpm_buf = bpm_buf[-BPM_BUF_SIZE:]
                    med_bpm = float(np.median(bpm_buf))
                    if smooth_bpm < 30:
                        smooth_bpm = med_bpm
                    else:
                        smooth_bpm += EMA_BPM * (med_bpm - smooth_bpm)

                with state.lock:
                    state.ml_ts_us = int(ts_window[-1])
                    state.pred_bpm_ml = float(smooth_bpm)

        def ml_classifier_worker() -> None:
            if clf is None:
                return
            last_seen = 0
            EMA_CONF = 0.3
            while not stop_event.is_set():
                data_event.wait(timeout=0.25)
                data_event.clear()

                with state.lock:
                    cur = state.csi_counter
                if cur - last_seen < int(args.infer_stride):
                    continue
                last_seen = cur

                with buf_lock:
                    if len(csi_buf) < WINDOW_SIZE:
                        continue
                    tail = list(csi_buf)[-WINDOW_SIZE:]
                ts_window = np.asarray([x[0] for x in tail], dtype=np.int64)
                amp_window = np.asarray([x[1] for x in tail], dtype=np.float32)

                dt = np.diff(ts_window).astype(np.float64) / 1_000_000.0
                dt = dt[dt > 0]
                fs = float(np.clip(1.0 / np.median(dt), 20.0, 200.0)) if len(dt) > 0 else 80.0

                feat = process_window(amp_window, fs=fs)
                x = feat.reshape(1, WINDOW_SIZE, SC_COUNT)
                pvec = compute_periodicity_vector(feat, fs)
                p_in = pvec.reshape(1, SC_COUNT)
                prob_human = float(clf.predict([x, p_in], verbose=0)[0, 0])

                with state.lock:
                    state.ml_ts_us = int(ts_window[-1])
                    state.ml_prob_human_30s = prob_human
                    if state.ml_prob_human_30s_smooth is None:
                        smooth = prob_human
                    else:
                        smooth = state.ml_prob_human_30s_smooth + EMA_CONF * (
                            prob_human - state.ml_prob_human_30s_smooth
                        )
                    state.ml_prob_human_30s_smooth = float(smooth)
                    if state.ml_detected:
                        detected = smooth >= float(args.human_threshold) * 0.5
                    else:
                        detected = smooth >= float(args.human_threshold)
                    state.ml_detected = int(detected)

        def writer_worker() -> None:
            last_written_ts = 0
            while not stop_event.is_set():
                time.sleep(max(0.05, float(args.log_period_s)))
                with state.lock:
                    rx_ts_us = int(state.ml_ts_us or state.latest_csi_ts_us)
                    if rx_ts_us <= 0 or rx_ts_us == last_written_ts:
                        continue
                    last_written_ts = rx_ts_us
                    ml_prob = state.ml_prob_human_30s
                    ml_prob_smooth = state.ml_prob_human_30s_smooth
                    detected_human = int(state.ml_detected)
                    pred_bpm_ml = float(state.pred_bpm_ml) if detected_human else 0.0
                    sensor_bpm = state.sensor_bpm
                    sensor_valid = int(state.sensor_bpm_valid)

                writer.writerow(
                    [
                        rx_ts_us,
                        detected_human,
                        "" if ml_prob is None else round(float(ml_prob), 6),
                        "" if ml_prob_smooth is None else round(float(ml_prob_smooth), 6),
                        round(pred_bpm_ml, 2),
                        sensor_bpm if sensor_bpm is not None else "",
                        sensor_valid,
                    ]
                )
                f.flush()

                print(
                    f"ts={rx_ts_us} present={bool(detected_human)} "
                    f"bpm_ml={pred_bpm_ml:.1f} sensor_bpm={sensor_bpm}"
                )

        t_ml_reg = threading.Thread(target=ml_worker, name="ml-regressor-30s", daemon=True)
        t_ml_clf = threading.Thread(target=ml_classifier_worker, name="ml-classifier-30s", daemon=True)
        t_writer = threading.Thread(target=writer_worker, name="writer", daemon=True)
        t_ml_reg.start()
        t_ml_clf.start()
        t_writer.start()

        print(f"Live inference started on {args.port}. Ctrl+C to stop.")
        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                parts = line.split(",")
                tag = parts[0]

                if tag == "BPM" and len(parts) >= 5:
                    try:
                        latest_sensor_bpm = float(parts[2])
                        latest_sensor_valid = int(parts[3])
                    except ValueError:
                        continue
                    with state.lock:
                        state.sensor_bpm = latest_sensor_bpm
                        state.sensor_bpm_valid = latest_sensor_valid
                    continue

                if tag != "CSI_PKT":
                    continue
                if len(parts) < 5 + SC_COUNT:
                    continue

                try:
                    ts_us = int(parts[1])
                    amps = np.asarray([float(v) for v in parts[5 : 5 + SC_COUNT]], dtype=np.float32)
                except ValueError:
                    continue

                with buf_lock:
                    csi_buf.append((ts_us, amps))
                with state.lock:
                    state.latest_csi_ts_us = int(ts_us)
                    state.csi_counter += 1
                data_event.set()

        except KeyboardInterrupt:
            print("Stopping live inference...")
        finally:
            stop_event.set()
            data_event.set()
            ser.close()
            t_presence.join(timeout=2.0)
            t_ml_reg.join(timeout=2.0)
            if clf is not None:
                t_ml_clf.join(timeout=2.0)
            t_writer.join(timeout=2.0)


if __name__ == "__main__":
    main()
