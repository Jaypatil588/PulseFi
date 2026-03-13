#!/usr/bin/env python3
"""
Offline sweep to find a 5-second presence threshold that separates human vs stable sessions.

Uses raw session files:
  - csi_packets.csv
  - markers.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from presence_5s import compute_presence_features, get_feature_value, window_by_duration


SC_COUNT = 64


@dataclass(frozen=True)
class SessionScore:
    session: str
    label: int  # 1=human, 0=stable
    score: float
    details: Dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data", help="Root directory containing session folders.")
    p.add_argument("--duration-s", type=float, default=5.0, help="Window duration in seconds.")
    p.add_argument(
        "--feature",
        default="resp_peak_hi_p90",
        help="Feature key to threshold (see output for available keys).",
    )
    p.add_argument(
        "--use-marker-start",
        action="store_true",
        help="For human sessions, anchor the 5s window at the human_start marker timestamp.",
    )
    p.add_argument(
        "--out-json",
        default="runtime/presence_threshold_sweep.json",
        help="Where to write results JSON.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, skip sweep and just evaluate this threshold.",
    )
    return p.parse_args()


def _read_markers(markers_csv: Path) -> List[Dict[str, str]]:
    if not markers_csv.exists():
        return []
    out: List[Dict[str, str]] = []
    with markers_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out


def _infer_label_from_markers(markers: List[Dict[str, str]]) -> Optional[int]:
    events = [str(m.get("event", "")).strip() for m in markers]
    # Accept human_start-only sessions (capture may have been stopped early).
    has_human_start = "human_start" in events
    is_stable = len([e for e in events if e]) == 0
    if has_human_start:
        return 1
    if is_stable:
        return 0
    return None


def _human_start_ts_us(markers: List[Dict[str, str]]) -> Optional[int]:
    for m in markers:
        if str(m.get("event", "")).strip() == "human_start":
            try:
                return int(float(m.get("marker_ts_us", "0")))
            except Exception:
                return None
    return None


def load_session_scores(
    data_root: Path, duration_s: float, feature_key: str, use_marker_start: bool
) -> Tuple[List[SessionScore], List[str]]:
    scores: List[SessionScore] = []
    skipped: List[str] = []

    for session_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        csi_csv = session_dir / "csi_packets.csv"
        markers_csv = session_dir / "markers.csv"
        if not csi_csv.exists():
            continue

        markers = _read_markers(markers_csv)
        label = _infer_label_from_markers(markers)
        if label is None:
            # Fallback heuristics based on folder name when markers are incomplete.
            n = session_dir.name.lower()
            if "human" in n:
                label = 1
            elif "stable" in n:
                label = 0
        if label is None:
            skipped.append(f"{session_dir.name}: invalid markers")
            continue

        try:
            df = pd.read_csv(csi_csv)
        except Exception as e:
            skipped.append(f"{session_dir.name}: failed read csi ({e})")
            continue

        if df.shape[0] < 50 or df.shape[1] < 4 + SC_COUNT:
            skipped.append(f"{session_dir.name}: too small")
            continue

        ts_us = df.iloc[:, 0].to_numpy(np.int64)
        amps = df.iloc[:, 4 : 4 + SC_COUNT].to_numpy(np.float32)

        anchor: Optional[int] = None
        if use_marker_start and label == 1:
            anchor = _human_start_ts_us(markers)
        w_ts, w_amps = window_by_duration(ts_us, amps, duration_s=duration_s, anchor_ts_us=anchor)
        if w_ts.size < 50:
            skipped.append(f"{session_dir.name}: insufficient window")
            continue

        feats = compute_presence_features(w_ts, w_amps)
        try:
            score = get_feature_value(feats, feature_key)
        except KeyError as e:
            raise SystemExit(str(e))

        scores.append(
            SessionScore(
                session=session_dir.name,
                label=int(label),
                score=float(score),
                details=feats.as_dict(),
            )
        )

    return scores, skipped


def _confusion(scores: List[SessionScore], threshold: float, *, higher_is_human: bool = True) -> Dict[str, int]:
    tp = tn = fp = fn = 0
    for s in scores:
        pred = 1 if (s.score >= threshold) == higher_is_human else 0
        if s.label == 1 and pred == 1:
            tp += 1
        elif s.label == 0 and pred == 0:
            tn += 1
        elif s.label == 0 and pred == 1:
            fp += 1
        elif s.label == 1 and pred == 0:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _f1(conf: Dict[str, int]) -> float:
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom > 0 else 0.0


def _balanced_acc(conf: Dict[str, int]) -> float:
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(0.5 * (tpr + tnr))


def sweep_threshold(scores: List[SessionScore]) -> Dict[str, object]:
    vals = np.asarray([s.score for s in scores], dtype=np.float64)
    if vals.size == 0:
        return {"best_threshold": None, "best_f1": 0.0, "best_balanced_acc": 0.0}

    # Candidate thresholds: midpoints between sorted unique values.
    uniq = np.unique(np.sort(vals))
    if uniq.size == 1:
        thr = float(uniq[0])
        conf = _confusion(scores, thr)
        return {
            "best_threshold": thr,
            "best_f1": _f1(conf),
            "best_balanced_acc": _balanced_acc(conf),
            "best_confusion": conf,
        }

    mids = (uniq[:-1] + uniq[1:]) / 2.0
    best = {
        "best_threshold": float(mids[0]),
        "best_f1": -1.0,
        "best_balanced_acc": -1.0,
        "best_confusion": {},
    }
    for thr in mids:
        conf = _confusion(scores, float(thr))
        f1 = _f1(conf)
        bal = _balanced_acc(conf)
        # Primary objective: F1. Tie-breaker: balanced accuracy.
        if (f1 > best["best_f1"]) or (abs(f1 - best["best_f1"]) < 1e-12 and bal > best["best_balanced_acc"]):
            best = {
                "best_threshold": float(thr),
                "best_f1": float(f1),
                "best_balanced_acc": float(bal),
                "best_confusion": conf,
            }
    return best


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    os.makedirs(Path(args.out_json).parent, exist_ok=True)

    scores, skipped = load_session_scores(
        data_root, duration_s=float(args.duration_s), feature_key=str(args.feature), use_marker_start=bool(args.use_marker_start)
    )
    n_h = sum(1 for s in scores if s.label == 1)
    n_s = sum(1 for s in scores if s.label == 0)

    if args.threshold is None:
        best = sweep_threshold(scores)
        thr = best.get("best_threshold")
    else:
        best = {}
        thr = float(args.threshold)

    conf = _confusion(scores, float(thr)) if thr is not None else {}
    out = {
        "feature": args.feature,
        "duration_s": float(args.duration_s),
        "use_marker_start": bool(args.use_marker_start),
        "threshold": float(thr) if thr is not None else None,
        "counts": {"total": len(scores), "human": n_h, "stable": n_s, "skipped": len(skipped)},
        "confusion": conf,
        "f1": _f1(conf) if conf else 0.0,
        "balanced_acc": _balanced_acc(conf) if conf else 0.0,
        "best_sweep": best,
        "sessions": [
            {"session": s.session, "label": s.label, "score": s.score, "features": s.details} for s in scores
        ],
        "skipped": skipped[:200],
    }

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    # Console summary (quick scan).
    print(f"sessions={len(scores)} (human={n_h}, stable={n_s}), skipped={len(skipped)}")
    print(f"feature={args.feature} duration_s={float(args.duration_s):.2f} threshold={thr}")
    if conf:
        print(f"confusion tp={conf['tp']} tn={conf['tn']} fp={conf['fp']} fn={conf['fn']}")
        print(f"f1={out['f1']:.3f} balanced_acc={out['balanced_acc']:.3f}")


if __name__ == "__main__":
    main()
