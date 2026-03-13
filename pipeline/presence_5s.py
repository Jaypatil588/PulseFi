#!/usr/bin/env python3
"""
5-second presence features for fast "human present" detection.

These are intentionally lightweight and do not depend on the 30-second ML model.
The functions here are used both by offline threshold sweeps and live inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


SC_COUNT = 64


def estimate_fs_hz(ts_us: np.ndarray, default_fs_hz: float = 80.0) -> float:
    """Estimate sample rate from timestamp deltas (us)."""
    if ts_us.size < 3:
        return float(default_fs_hz)
    dt_s = np.diff(ts_us.astype(np.float64)) / 1_000_000.0
    dt_s = dt_s[dt_s > 0]
    if dt_s.size < 3:
        return float(default_fs_hz)
    fs = 1.0 / float(np.median(dt_s))
    return float(np.clip(fs, 20.0, 200.0))


def window_by_duration(
    ts_us: np.ndarray,
    amps: np.ndarray,
    *,
    duration_s: float,
    anchor_ts_us: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a window of samples covering the last `duration_s` seconds.

    If `anchor_ts_us` is set, the returned window starts at the first sample
    with timestamp >= anchor (and extends for duration_s). If insufficient
    samples are available, returns the largest possible suffix/prefix window.
    """
    if ts_us.size == 0:
        return ts_us, amps

    if anchor_ts_us is None:
        end_ts = int(ts_us[-1])
        start_ts = int(end_ts - duration_s * 1_000_000.0)
        idx0 = int(np.searchsorted(ts_us, start_ts, side="left"))
        return ts_us[idx0:], amps[idx0:]

    idx0 = int(np.searchsorted(ts_us, int(anchor_ts_us), side="left"))
    if idx0 >= ts_us.size:
        return ts_us[-1:], amps[-1:]
    end_ts = int(ts_us[idx0] + duration_s * 1_000_000.0)
    idx1 = int(np.searchsorted(ts_us, end_ts, side="right"))
    return ts_us[idx0:idx1], amps[idx0:idx1]


def _demean_cols(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=0, keepdims=True)


def raw_amplitude_variance(amps: np.ndarray) -> float:
    """Variance over all samples/subcarriers, on raw (unnormalized) amplitude."""
    if amps.size == 0:
        return 0.0
    return float(np.var(amps.astype(np.float64)))


def subcarrier_crosscorr_mean_abs(amps: np.ndarray) -> float:
    """Mean absolute off-diagonal correlation between subcarriers."""
    if amps.shape[0] < 5:
        return 0.0
    x = _demean_cols(amps.astype(np.float64))
    std = x.std(axis=0, ddof=1)
    std[std < 1e-9] = 1.0
    z = x / std
    c = (z.T @ z) / max(1, (z.shape[0] - 1))
    # Exclude diagonal efficiently by subtracting identity then averaging abs.
    abs_off = np.abs(c - np.eye(c.shape[0], dtype=np.float64))
    denom = abs_off.size - c.shape[0]
    return float(abs_off.sum() / max(1, denom))


def _rfft_power(x: np.ndarray, n_fft: int) -> np.ndarray:
    xf = np.fft.rfft(x, n=n_fft)
    return (np.abs(xf) ** 2).astype(np.float64)


def respiratory_power_ratio(
    amps: np.ndarray,
    fs_hz: float,
    *,
    band_hz: Tuple[float, float] = (0.1, 0.5),
    n_fft: int = 1024,
    statistic: str = "mean",
) -> float:
    """Respiratory band power ratio over total (excluding DC), aggregated across subcarriers.

    `statistic` controls how to pool per-subcarrier ratios: mean|p90|max.
    """
    if amps.shape[0] < 16:
        return 0.0
    x = _demean_cols(amps.astype(np.float64))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs_hz))
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    ratios = np.zeros(x.shape[1], dtype=np.float64)
    for sc in range(x.shape[1]):
        p = _rfft_power(x[:, sc], n_fft)
        dc = p[0]
        total = float(max(1e-9, p.sum() - dc))
        ratios[sc] = float(p[band].sum() / total)

    statistic = statistic.lower().strip()
    if statistic == "p90":
        return float(np.quantile(ratios, 0.9))
    if statistic == "max":
        return float(np.max(ratios))
    return float(np.mean(ratios))


def resp_peak_to_highband_mean(
    amps: np.ndarray,
    fs_hz: float,
    *,
    resp_band_hz: Tuple[float, float] = (0.1, 0.5),
    hi_band_hz: Tuple[float, float] = (0.8, 5.0),
    n_fft: int = 1024,
    statistic: str = "median",
) -> float:
    """Peak PSD in resp band divided by mean PSD in a higher band, pooled across subcarriers."""
    if amps.shape[0] < 16:
        return 0.0
    x = _demean_cols(amps.astype(np.float64))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs_hz))
    resp = (freqs >= resp_band_hz[0]) & (freqs <= resp_band_hz[1])
    hi = (freqs >= hi_band_hz[0]) & (freqs <= hi_band_hz[1])
    scores = np.zeros(x.shape[1], dtype=np.float64)
    for sc in range(x.shape[1]):
        p = _rfft_power(x[:, sc], n_fft)
        resp_peak = float(np.max(p[resp])) if np.any(resp) else 0.0
        hi_mean = float(np.mean(p[hi])) if np.any(hi) else 0.0
        scores[sc] = resp_peak / float(hi_mean + 1e-9)

    statistic = statistic.lower().strip()
    if statistic == "p90":
        return float(np.quantile(scores, 0.9))
    if statistic == "max":
        return float(np.max(scores))
    return float(np.median(scores))


def bandpass_filtered_std(
    amps: np.ndarray,
    fs_hz: float,
    *,
    band_hz: Tuple[float, float] = (0.8, 2.17),
) -> float:
    """Global std after DC removal + bandpass + SavGol smoothing, before normalization."""
    if amps.shape[0] < 16:
        return 0.0
    x = amps.astype(np.float64).copy()
    nyq = 0.5 * float(fs_hz)
    low = band_hz[0] / nyq
    high = band_hz[1] / nyq
    if not (0 < low < high < 1):
        return 0.0

    b, a = butter(3, [low, high], btype="bandpass")
    out = np.zeros_like(x, dtype=np.float32)
    for sc in range(x.shape[1]):
        sig = x[:, sc]
        sig = sig - np.mean(sig)
        try:
            sig = filtfilt(b, a, sig)
        except ValueError:
            pass
        if sig.size >= 15:
            sig = savgol_filter(sig, 15, 3)
        out[:, sc] = sig.astype(np.float32)

    return float(np.std(out.astype(np.float64)))


@dataclass(frozen=True)
class PresenceFeatures:
    raw_var: float
    xcorr_mean_abs: float
    resp_ratio_mean: float
    resp_ratio_p90: float
    resp_ratio_max: float
    resp_peak_hi_median: float
    resp_peak_hi_p90: float
    bandpass_std: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "raw_var": self.raw_var,
            "xcorr_mean_abs": self.xcorr_mean_abs,
            "resp_ratio_mean": self.resp_ratio_mean,
            "resp_ratio_p90": self.resp_ratio_p90,
            "resp_ratio_max": self.resp_ratio_max,
            "resp_peak_hi_median": self.resp_peak_hi_median,
            "resp_peak_hi_p90": self.resp_peak_hi_p90,
            "bandpass_std": self.bandpass_std,
        }


def compute_presence_features(ts_us: np.ndarray, amps: np.ndarray) -> PresenceFeatures:
    ts_us = np.asarray(ts_us, dtype=np.int64)
    amps = np.asarray(amps, dtype=np.float32)
    fs_hz = estimate_fs_hz(ts_us)
    return PresenceFeatures(
        raw_var=raw_amplitude_variance(amps),
        xcorr_mean_abs=subcarrier_crosscorr_mean_abs(amps),
        resp_ratio_mean=respiratory_power_ratio(amps, fs_hz, statistic="mean"),
        resp_ratio_p90=respiratory_power_ratio(amps, fs_hz, statistic="p90"),
        resp_ratio_max=respiratory_power_ratio(amps, fs_hz, statistic="max"),
        resp_peak_hi_median=resp_peak_to_highband_mean(amps, fs_hz, statistic="median"),
        resp_peak_hi_p90=resp_peak_to_highband_mean(amps, fs_hz, statistic="p90"),
        bandpass_std=bandpass_filtered_std(amps, fs_hz),
    )


def get_feature_value(features: PresenceFeatures, key: str) -> float:
    if not hasattr(features, key):
        raise KeyError(f"Unknown feature '{key}'. Available: {', '.join(features.as_dict().keys())}")
    return float(getattr(features, key))

