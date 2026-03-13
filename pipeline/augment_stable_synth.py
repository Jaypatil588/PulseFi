#!/usr/bin/env python3
"""
Generate synthetic stable windows to balance class counts.

Reads a training_main.csv, augments stable windows with mild noise/scale/drift,
and writes a new CSV with extra stable rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

csv.field_size_limit(sys.maxsize)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--target-ratio", type=float, default=1.0, help="Target stable:human ratio.")
    p.add_argument("--noise-std", type=float, default=0.02, help="Gaussian noise std (post-processed).")
    p.add_argument("--scale-jitter", type=float, default=0.02, help="Per-subcarrier scale jitter std.")
    p.add_argument("--drift-std", type=float, default=0.01, help="Linear drift std across time.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def synthesize_window(
    x: np.ndarray,
    rng: np.random.RandomState,
    noise_std: float,
    scale_jitter: float,
    drift_std: float,
) -> np.ndarray:
    n_t, n_sc = x.shape
    out = x.astype(np.float32).copy()

    if scale_jitter > 0:
        scale = rng.normal(1.0, scale_jitter, size=(1, n_sc)).astype(np.float32)
        out *= scale

    if drift_std > 0:
        drift = rng.normal(0.0, drift_std, size=(1, n_sc)).astype(np.float32)
        ramp = np.linspace(-0.5, 0.5, n_t, dtype=np.float32).reshape(-1, 1)
        out += ramp * drift

    if noise_std > 0:
        out += rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    return out


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    with open(args.input_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise SystemExit("No rows found in input CSV.")

    counts = Counter(int(r["class_label"]) for r in rows)
    n_h = counts.get(1, 0)
    n_s = counts.get(0, 0)
    target_s = int(round(n_h * args.target_ratio))

    if n_s >= target_s:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Stable already >= target. copied={len(rows)}")
        return

    stable_rows = [r for r in rows if int(r["class_label"]) == 0]
    if not stable_rows:
        raise SystemExit("No stable rows found to augment.")

    needed = target_s - n_s
    synth_rows: List[Dict[str, str]] = []

    for i in range(needed):
        src = stable_rows[i % len(stable_rows)]
        x = np.asarray(json.loads(src["csi_features_json"]), dtype=np.float32)
        x2 = synthesize_window(x, rng, args.noise_std, args.scale_jitter, args.drift_std)

        new_row = dict(src)
        new_row["csi_features_json"] = json.dumps(x2.tolist())
        new_row["periodicity_json"] = json.dumps([0.0] * x2.shape[1])
        new_row["session_id"] = f"{src['session_id']}_synth"
        new_row["day_id"] = "synth"
        try:
            ts = int(src["win_start_ts_us"])
        except Exception:
            ts = 0
        new_row["win_start_ts_us"] = str(ts + i + 1)
        new_row["win_end_ts_us"] = str(ts + i + 2)
        synth_rows.append(new_row)

    out_rows = rows + synth_rows
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        f"rows_in={len(rows)} rows_synth={len(synth_rows)} rows_out={len(out_rows)} "
        f"class_counts={counts} target_stable={target_s}"
    )


if __name__ == "__main__":
    main()
