#!/usr/bin/env python3
"""
PulseFi Live Dashboard — Real-time CSI-based heart rate monitoring.

Combines serial inference with a Streamlit dashboard showing:
- Live human/stable detection with confidence
- Real-time predicted BPM + sensor BPM comparison
- Rolling BPM chart
- CSI signal heatmap

Usage:
  python3.11 -m streamlit run pipeline/dashboard.py
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import butter, filtfilt, savgol_filter

WINDOW_SIZE_TWO_STAGE = 1600
WINDOW_SIZE_NICKBILD = 100
SC_COUNT = 64


# ─── Signal Processing ──────────────────────────────────────────────────────

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


# ─── Shared State ────────────────────────────────────────────────────────────

@dataclass
class LiveState:
    """Thread-safe shared state between serial reader and dashboard."""
    # Serial connection
    connected: bool = False
    rx_status: str = "Disconnected"
    model_type: str = ""  # "nickbild" or "two_stage" when running
    csi_count: int = 0
    bpm_count: int = 0

    # Latest inference results
    human_prob: float = 0.0
    pred_class: int = 0
    pred_bpm: float = 0.0
    sensor_bpm: float = 0.0
    sensor_valid: int = 0
    last_infer_time: float = 0.0
    # Raw ML outputs (before smoothing)
    spectral_bpm: float = 0.0
    spectral_conf: float = 0.0
    detected_human: bool = False

    # Smoothed outputs (EMA + median)
    smooth_conf: float = 0.0
    smooth_bpm: float = 0.0
    bpm_buf: List[float] = field(default_factory=list)

    # Rolling history for charts (max 120 points ≈ ~2 min at stride 10)
    bpm_history: List[Tuple[float, float, float]] = field(default_factory=list)  # (time, pred, sensor)
    prob_history: List[Tuple[float, float]] = field(default_factory=list)  # (time, prob)
    max_history: int = 120


def serial_reader(
    state: LiveState,
    port: str,
    baud: int,
    use_nickbild: bool,
    clf,
    reg,
    nickbild_model,
    stride: int,
    threshold: float,
) -> None:
    """Background thread: reads serial, runs inference, updates state."""
    import serial as ser_mod

    window_size = WINDOW_SIZE_NICKBILD if use_nickbild else WINDOW_SIZE_TWO_STAGE
    csi_buf: Deque[Tuple[int, np.ndarray]] = deque(maxlen=window_size)
    csi_counter = 0
    start_time = time.time()

    try:
        serial_conn = ser_mod.Serial(
            port=port, baudrate=baud, bytesize=8,
            parity="N", stopbits=1, timeout=1,
        )
        state.connected = True
        state.rx_status = "Connected"
    except Exception as e:
        state.rx_status = f"Error: {e}"
        return

    try:
        while state.connected:
            raw = serial_conn.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            parts = line.split(",")
            tag = parts[0]

            if tag == "BPM" and len(parts) >= 5:
                try:
                    state.sensor_bpm = float(parts[2])
                    state.sensor_valid = int(parts[3])
                    state.bpm_count += 1
                except ValueError:
                    pass
                continue

            if tag != "CSI_PKT" or len(parts) < 5 + SC_COUNT:
                continue

            try:
                ts_us = int(parts[1])
                amps = np.asarray(
                    [float(v) for v in parts[5:5 + SC_COUNT]], dtype=np.float32
                )
            except ValueError:
                continue

            csi_buf.append((ts_us, amps))
            csi_counter += 1
            state.csi_count = csi_counter

            if len(csi_buf) < window_size or csi_counter % stride != 0:
                continue

            # Run inference
            try:
                ts_window = np.asarray([x[0] for x in csi_buf], dtype=np.int64)
                amp_window = np.asarray([x[1] for x in csi_buf], dtype=np.float32)
                dt = np.diff(ts_window).astype(np.float64) / 1_000_000.0
                dt = dt[dt > 0]
                fs = float(np.clip(1.0 / np.median(dt), 20.0, 200.0)) if len(dt) > 0 else 80.0

                feat = process_window(amp_window, fs=fs)
                x_in = feat.reshape(1, window_size, SC_COUNT)

                if use_nickbild:
                    pred_bpm_ml = float(nickbild_model.predict(x_in, verbose=0)[0, 0])
                    prob = 1.0
                    detected = True
                else:
                    prob = 0.0
                    if clf is not None:
                        pvec = compute_periodicity_vector(feat, fs)
                        p_in = pvec.reshape(1, SC_COUNT)
                        prob = float(clf.predict([x_in, p_in], verbose=0)[0, 0])
                    pred_bpm_ml = float(reg.predict(x_in, verbose=0)[0, 0]) * 100.0
                    if state.detected_human:
                        detected = state.smooth_conf >= threshold * 0.5
                    else:
                        detected = state.smooth_conf >= threshold
            except Exception as e:
                state.rx_status = f"Inference error: {e}"
                continue

            # ── Smooth confidence (EMA) + hysteresis (two-stage only) ──
            EMA_CONF = 0.3
            EMA_BPM = 0.15
            BPM_BUF_SIZE = 15

            if not use_nickbild and clf is not None:
                if state.smooth_conf < 1e-6:
                    state.smooth_conf = prob
                else:
                    state.smooth_conf += EMA_CONF * (prob - state.smooth_conf)
                state.pred_class = 1 if detected else 0
                state.human_prob = float(state.smooth_conf)
            else:
                state.smooth_conf = 1.0
                state.pred_class = 1
                state.human_prob = 1.0

            # ── Smooth BPM (median buffer rejects outliers, then EMA) ──
            if 30 < pred_bpm_ml < 200:
                state.bpm_buf.append(pred_bpm_ml)
                if len(state.bpm_buf) > BPM_BUF_SIZE:
                    state.bpm_buf = state.bpm_buf[-BPM_BUF_SIZE:]
                med_bpm = float(np.median(state.bpm_buf))
                if state.smooth_bpm < 30:
                    state.smooth_bpm = med_bpm
                else:
                    state.smooth_bpm += EMA_BPM * (med_bpm - state.smooth_bpm)

            display_bpm = state.smooth_bpm if (detected or use_nickbild) and state.smooth_bpm > 30 else 0.0

            now = time.time() - start_time

            state.pred_bpm = display_bpm
            state.spectral_bpm = pred_bpm_ml
            state.spectral_conf = prob
            state.detected_human = detected or use_nickbild
            state.last_infer_time = now

            # Update history
            sensor_val = state.sensor_bpm if state.sensor_valid else 0.0
            state.bpm_history.append((now, display_bpm if (detected or use_nickbild) else float("nan"), sensor_val))
            state.prob_history.append((now, state.smooth_conf))
            if len(state.bpm_history) > state.max_history:
                state.bpm_history = state.bpm_history[-state.max_history:]
            if len(state.prob_history) > state.max_history:
                state.prob_history = state.prob_history[-state.max_history:]

    except Exception as e:
        state.rx_status = f"Error: {e}"
    finally:
        serial_conn.close()
        state.connected = False
        state.rx_status = "Disconnected"


# ─── Dashboard UI ────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="PulseFi Live",
        page_icon="💓",
        layout="wide",
    )

    # Custom CSS — Donezo-style
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #121212;
        --surface: #1e1e1e;
        --surface-variant: #2d2d2d;
        --ink: #e1e1e1;
        --muted: #9e9e9e;
        --line: #424242;
        --green: #4caf50;
        --green-dark: #43a047;
        --green-darker: #2e7d32;
        --green-light: rgba(76, 175, 80, 0.15);
        --shadow: 0 1px 3px rgba(0,0,0,0.3);
        --shadow-lg: 0 4px 12px rgba(0,0,0,0.4);
    }

    html, body, [class*="css"] {
        font-family: "Inter", -apple-system, sans-serif;
        color: var(--ink);
    }

    .stApp { background: var(--bg); }
    [data-testid="stAppViewContainer"] { background: var(--bg); }
    [data-testid="block-container"] { background: var(--bg); }

    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 800;
        font-size: 20px;
        color: var(--green);
        margin-bottom: 24px;
    }
    .brand-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: var(--green);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        font-size: 18px;
        font-weight: 800;
    }

    .sidebar-section {
        margin-top: 20px;
        font-size: 11px;
        letter-spacing: 0.08em;
        color: var(--muted);
        text-transform: uppercase;
        font-weight: 600;
    }
    .sidebar-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        border-radius: 12px;
        font-weight: 600;
        color: var(--ink);
        margin: 2px 0;
    }
    .sidebar-item.active {
        background: var(--green-light);
        color: var(--green-dark);
    }

    .sidebar-cta {
        margin-top: 32px;
        padding: 20px;
        border-radius: 16px;
        background: var(--surface-variant);
        border: 1px solid var(--line);
        color: var(--ink);
        font-size: 14px;
        font-weight: 600;
    }
    .sidebar-cta-title { font-size: 15px; margin-bottom: 6px; }
    .sidebar-cta-sub { font-size: 12px; opacity: 0.9; margin-bottom: 12px; }

    .stat-card {
        background: var(--surface);
        border-radius: 16px;
        padding: 20px;
        box-shadow: var(--shadow);
        border: 1px solid var(--line);
        position: relative;
    }
    .stat-card.primary {
        background: var(--surface-variant);
        border: 1px solid var(--green-dark);
        color: var(--green);
    }
    .stat-card .arrow-up {
        position: absolute;
        top: 16px;
        right: 16px;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: rgba(255,255,255,0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }
    .stat-card.primary .arrow-up { background: var(--green-light); color: var(--green); }
    .stat-value { font-size: 32px; font-weight: 800; line-height: 1.1; }
    .stat-sub { font-size: 12px; color: var(--muted); margin-top: 6px; }
    .stat-card.primary .stat-sub { color: var(--muted); }

    .content-card {
        background: var(--surface);
        border-radius: 16px;
        padding: 24px;
        box-shadow: var(--shadow);
        border: 1px solid var(--line);
    }

    .header-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 24px;
    }
    .header-title { font-size: 24px; font-weight: 800; color: var(--ink) !important; }
    .header-sub { color: var(--muted); font-size: 13px; margin-top: 4px; }

    .pill-btn {
        border-radius: 999px;
        padding: 10px 18px;
        background: var(--green);
        color: #fff;
        font-weight: 700;
        font-size: 13px;
        border: none;
    }
    .pill-btn.outline {
        background: #fff;
        color: var(--green);
        border: 2px solid var(--green);
    }

    .big-bpm { font-size: 80px; font-weight: 800; text-align: center; line-height: 1; margin: 0; }
    .big-bpm-xl { font-size: 100px; font-weight: 800; text-align: center; line-height: 1; margin: 0; }
    .bpm-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); text-align: center; margin-top: 8px; }
    .status-badge {
        padding: 8px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .human { background: var(--green-light); color: var(--green); }
    .stable { background: var(--surface-variant); color: var(--muted); }
    .disconnected { background: #fee2e2; color: #991b1b; }

    [data-testid="stMetric"] {
        background: var(--surface-variant);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px 14px;
    }

    /* Streamlit dark theme overrides */
    .stMarkdown, .stMarkdown p { color: var(--ink) !important; }
    label { color: var(--muted) !important; }
    .stTextInput input, .stNumberInput input { background: var(--surface-variant) !important; color: var(--ink) !important; border-color: var(--line) !important; }
    [data-testid="stAlert"] { background: var(--surface-variant) !important; border: 1px solid var(--line) !important; color: var(--ink) !important; }

    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar — Donezo-style
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-icon">P</span> PulseFi</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">MENU</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-item active">📊 Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-item">📁 Sessions</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-item">📈 Analytics</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">CONFIG</div>', unsafe_allow_html=True)
        use_nickbild = st.toggle("Use Nickbild model (100-pkt)", value=False, help="Nickbild: simpler LSTM, 100 packets. Two-stage: classifier + regressor, 1600 packets.")
        port = st.text_input("Serial Port", value="/dev/cu.usbserial-0001")
        baud = st.number_input("Baud Rate", value=921600, step=1)
        clf_path = "models/phase1/stage_a_classifier.keras"
        reg_path = "models/phase1/stage_b_regressor.keras"
        nickbild_path = "models/nickbild/csi_hr.keras"
        if use_nickbild:
            nickbild_path = st.text_input("Nickbild Model", value=nickbild_path)
            stride = st.slider("Inference Stride", 1, 20, 5, help="Nickbild: every N packets (100-pkt window)")
            threshold = 0.5
        else:
            clf_path = st.text_input("Classifier Model", value=clf_path)
            reg_path = st.text_input("Regressor Model", value=reg_path)
            stride = st.slider("Inference Stride", 1, 50, 10)
            threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.05)

        start_btn = st.button("Start Session", use_container_width=True, type="primary")
        stop_btn = st.button("Stop", use_container_width=True)

        st.markdown("""
        <div class="sidebar-cta">
            <div class="sidebar-cta-title">CSI Monitor</div>
            <div class="sidebar-cta-sub">Real-time heart rate from Wi-Fi signals</div>
        </div>
        """, unsafe_allow_html=True)

    # Initialize session state
    if "state" not in st.session_state:
        st.session_state.state = LiveState()
    if "thread" not in st.session_state:
        st.session_state.thread = None

    state: LiveState = st.session_state.state

    # Start/Stop logic
    if start_btn and not state.connected:
        from tensorflow import keras
        if use_nickbild:
            nickbild_model = keras.models.load_model(nickbild_path, safe_mode=False)
            clf, reg = None, None
            state.model_type = "nickbild"
        else:
            clf = keras.models.load_model(clf_path, safe_mode=False)
            reg = keras.models.load_model(reg_path, safe_mode=False)
            nickbild_model = None
            state.model_type = "two_stage"
        state.connected = True
        state.rx_status = "Connecting..."
        t = threading.Thread(
            target=serial_reader,
            args=(
                state,
                port,
                baud,
                use_nickbild,
                clf,
                reg,
                nickbild_model,
                stride,
                threshold,
            ),
            daemon=True,
        )
        t.start()
        st.session_state.thread = t

    if stop_btn:
        state.connected = False
        state.model_type = ""

    # ── Main Display ──
    sensor_ok = bool(state.sensor_valid and state.sensor_bpm > 0)

    # Header
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown('<div class="header-title">Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-sub">Monitor live CSI heart rate signals in real time.</div>', unsafe_allow_html=True)

    # Summary cards — Donezo-style (one primary green)
    model_label = state.model_type or ("nickbild" if use_nickbild else "two_stage")
    model_label = model_label.replace("_", " ").title() if model_label else ""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        status = "Human" if state.detected_human and state.connected else "Stable"
        sub = f"{state.rx_status} · {model_label}" if state.connected else state.rx_status
        st.markdown(
            f'<div class="stat-card primary"><span class="arrow-up">▲</span>'
            f'<div class="stat-value">{status}</div>'
            f'<div class="stat-sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><span class="arrow-up">▲</span>'
            f'<div class="stat-value">{state.smooth_conf:.0%}</div>'
            f'<div class="stat-sub">Confidence</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card"><span class="arrow-up">▲</span>'
            f'<div class="stat-value">{state.csi_count:,}</div>'
            f'<div class="stat-sub">CSI Packets</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        sensor_disp = f"{state.sensor_bpm:.0f}" if sensor_ok else "-"
        st.markdown(
            f'<div class="stat-card"><span class="arrow-up">▲</span>'
            f'<div class="stat-value">{sensor_disp}</div>'
            f'<div class="stat-sub">Sensor BPM</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Main content row
    row_left, row_right = st.columns([1.5, 1])
    with row_left:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        if state.detected_human and state.smooth_bpm > 30:
            bpm_display = f"{state.smooth_bpm:.0f}"
            bpm_color = "var(--green-dark)"
        else:
            bpm_display = "—"
            bpm_color = "var(--muted)"
        pred_class = "big-bpm" if sensor_ok else "big-bpm-xl"
        st.markdown(
            f'<p class="{pred_class}" style="color:{bpm_color}">{bpm_display}</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<p class="bpm-label">Predicted BPM (CSI)</p>', unsafe_allow_html=True)
        sensor_display = f"{state.sensor_bpm:.0f}" if sensor_ok else "-"
        st.markdown(
            f'<p class="big-bpm" style="font-size:42px;color:var(--green-dark)">{sensor_display}</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<p class="bpm-label">Sensor BPM (MAX30102)</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row_right:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;color:var(--muted);margin-bottom:12px;font-weight:600">DETECTION</div>', unsafe_allow_html=True)
        if state.detected_human and state.connected:
            st.markdown('<p class="status-badge human">HUMAN DETECTED</p>', unsafe_allow_html=True)
        elif state.connected:
            st.markdown('<p class="status-badge stable">STABLE</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-badge disconnected">NOT CONNECTED</p>', unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.metric("Confidence", f"{state.smooth_conf:.1%}")
        st.metric("Sensor Valid", "Yes" if state.sensor_valid else "No")
        st.metric("BPM Readings", f"{state.bpm_count:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px;font-weight:700;margin-bottom:12px">BPM Over Time</div>', unsafe_allow_html=True)
        if state.bpm_history:
            times = [h[0] for h in state.bpm_history]
            preds = [h[1] for h in state.bpm_history]
            sensors = [h[2] for h in state.bpm_history]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times, y=preds, mode="lines+markers",
                name="Predicted (CSI)",
                line=dict(color="#16a34a", width=2.5),
                marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=times, y=sensors, mode="lines+markers",
                name="Sensor (MAX30102)",
                line=dict(color="#22c55e", width=2.5),
                marker=dict(size=5),
            ))
            fig.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="BPM",
                yaxis=dict(range=[40, 180], gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#e1e1e1")),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#e1e1e1")),
                height=320,
                margin=dict(l=20, r=20, t=10, b=40),
                legend=dict(orientation="h", y=-0.15, font=dict(color="#e1e1e1")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#e1e1e1"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for data...")
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px;font-weight:700;margin-bottom:12px">Detection Confidence</div>', unsafe_allow_html=True)
        if state.prob_history:
            times = [h[0] for h in state.prob_history]
            probs = [h[1] for h in state.prob_history]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=times, y=probs, mode="lines",
                fill="tozeroy",
                line=dict(color="#16a34a", width=2.5),
                fillcolor="rgba(34, 197, 94, 0.2)",
            ))
            fig2.add_hline(y=threshold, line_dash="dash",
                          line_color="#9e9e9e", annotation_text="Threshold")
            fig2.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Confidence",
                yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#e1e1e1")),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#e1e1e1")),
                height=320,
                margin=dict(l=20, r=20, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#e1e1e1"),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Waiting for data...")
        st.markdown('</div>', unsafe_allow_html=True)

    # Auto-refresh
    if state.connected:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
