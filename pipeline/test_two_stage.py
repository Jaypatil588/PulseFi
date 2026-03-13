#!/usr/bin/env python3
"""
Evaluate trained two-stage models on a held-out training_main.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, Tuple

csv.field_size_limit(sys.maxsize)

import numpy as np
from tensorflow import keras

WINDOW_SIZE = 1600

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--classifier-model", required=True)
    parser.add_argument("--regressor-model", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y_cls, y_bpm = [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                feat = np.asarray(json.loads(row["csi_features_json"]), dtype=np.float32)
                if feat.shape != (WINDOW_SIZE, 64):
                    continue
                X.append(feat)
                y_cls.append(int(row["class_label"]))
                y_bpm.append(float(row["y_bpm"]))
            except Exception:
                continue
    return np.asarray(X), np.asarray(y_cls), np.asarray(y_bpm)


def binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    def f1_for(label: int) -> float:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    f1_0 = f1_for(0)
    f1_1 = f1_for(1)
    return {
        "f1_stable_nonhuman": float(f1_0),
        "f1_human": float(f1_1),
        "macro_f1": float((f1_0 + f1_1) / 2.0),
    }


def main() -> None:
    args = parse_args()

    X, y_cls, y_bpm = load_data(args.test_csv)
    if len(X) == 0:
        raise RuntimeError("No valid rows in test CSV.")

    clf = keras.models.load_model(args.classifier_model, safe_mode=False)
    reg = keras.models.load_model(args.regressor_model, safe_mode=False)

    cls_prob = clf.predict(X, verbose=0).reshape(-1)
    cls_pred = (cls_prob >= 0.5).astype(np.int32)
    f1 = binary_macro_f1(y_cls, cls_pred)

    human_idx = np.where(y_cls == 1)[0]
    mae_human = None
    if len(human_idx) > 0:
        pred_bpm = reg.predict(X[human_idx], verbose=0).reshape(-1)
        pred_bpm = pred_bpm * 100.0
        mae_human = float(np.mean(np.abs(pred_bpm - y_bpm[human_idx])))

    out = {
        "rows_test": int(len(X)),
        "rows_human_test": int(len(human_idx)),
        **f1,
        "bpm_mae_human": mae_human,
        "f1_target_met_0_90": bool(f1["macro_f1"] >= 0.90),
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
