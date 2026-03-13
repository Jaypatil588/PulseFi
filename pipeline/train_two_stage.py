#!/usr/bin/env python3
"""
Two-stage training:
  Stage A: binary classifier (stable_nonhuman vs human)
  Stage B: BPM regressor (human windows only)

Training uses session-based splitting to prevent data leakage between
overlapping windows.  Splits are 64 / 16 / 20 % (train / val / test)
by session, so no window from a given recording session appears in more
than one partition.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

csv.field_size_limit(sys.maxsize)

import numpy as np
import tensorflow as tf
from tensorflow import keras

WINDOW_SIZE = 1600


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_training(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_cls, y_bpm, reg_mask, ts, sessions, periodicity).

    ``periodicity`` is shape (N, 64) — per-subcarrier autocorrelation peak
    strength with second-cycle validation.  If the column is missing from an
    older CSV the vector is filled with zeros so training still works.
    """
    X: List[np.ndarray] = []
    y_cls: List[int] = []
    y_bpm: List[float] = []
    reg_mask: List[int] = []
    ts: List[int] = []
    sessions: List[str] = []
    periodicity: List[np.ndarray] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = np.asarray(
                    json.loads(row["csi_features_json"]), dtype=np.float32
                )
                if features.shape != (WINDOW_SIZE, 64):
                    continue

                if "periodicity_json" in row and row["periodicity_json"]:
                    pvec = np.asarray(
                        json.loads(row["periodicity_json"]), dtype=np.float32
                    )
                    if pvec.shape != (64,):
                        pvec = np.zeros(64, dtype=np.float32)
                else:
                    pvec = np.zeros(64, dtype=np.float32)

                X.append(features)
                periodicity.append(pvec)
                y_cls.append(int(row["class_label"]))
                y_bpm.append(float(row["y_bpm"]) / 100.0)
                reg_mask.append(int(row["regression_mask"]))
                ts.append(int(row["win_start_ts_us"]))
                sessions.append(row["session_id"])
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_cls, dtype=np.int32),
        np.asarray(y_bpm, dtype=np.float32),
        np.asarray(reg_mask, dtype=np.int32),
        np.asarray(ts, dtype=np.int64),
        np.asarray(sessions),
        np.asarray(periodicity, dtype=np.float32),
    )


# ── Session-based splitting ──────────────────────────────────────────────────

def split_by_session(
    sessions: np.ndarray,
    y_cls: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices by session ID → ~64 % train, ~16 % val, ~20 % test.

    Entire sessions stay in one partition so overlapping windows never
    leak across splits.  Human and non-human sessions are split
    independently so every partition contains both classes.
    """
    unique = np.unique(sessions)
    rng = np.random.RandomState(seed)

    synth_sess = [s for s in unique if "_synth" in str(s)]
    base_unique = np.array([s for s in unique if s not in synth_sess])

    human_sess = []
    nonhuman_sess = []
    for s in base_unique:
        mask = sessions == s
        if np.any(y_cls[mask] == 1):
            human_sess.append(s)
        else:
            nonhuman_sess.append(s)

    human_sess = np.array(human_sess)
    nonhuman_sess = np.array(nonhuman_sess)
    rng.shuffle(human_sess)
    rng.shuffle(nonhuman_sess)

    def _ratio_split(arr: np.ndarray) -> Tuple[list, list, list]:
        n = len(arr)
        if n < 3:
            n_tr = max(1, n - 1)
            return list(arr[:n_tr]), list(arr[n_tr:]), []
        n_tr = max(1, round(n * 0.64))
        n_va = max(1, round(n * 0.16))
        return list(arr[:n_tr]), list(arr[n_tr:n_tr + n_va]), list(arr[n_tr + n_va:])

    h_tr, h_va, h_te = _ratio_split(human_sess)
    n_tr, n_va, n_te = _ratio_split(nonhuman_sess)

    train_sess = set(h_tr + n_tr + synth_sess)
    val_sess = set(h_va + n_va)
    test_sess = set(h_te + n_te)

    train_idx = np.where(np.isin(sessions, list(train_sess)))[0]
    val_idx = np.where(np.isin(sessions, list(val_sess)))[0]
    test_idx = np.where(np.isin(sessions, list(test_sess)))[0]

    for name, idx, sess in [
        ("train", train_idx, sorted(train_sess)),
        ("val", val_idx, sorted(val_sess)),
        ("test", test_idx, sorted(test_sess)),
    ]:
        cls_counts = (
            {int(c): int(np.sum(y_cls[idx] == c)) for c in np.unique(y_cls[idx])}
            if len(idx) > 0
            else {}
        )
        print(f"  {name}: {len(idx)} windows  sessions={sess}  classes={cls_counts}")

    return train_idx, val_idx, test_idx


# ── Model builders ───────────────────────────────────────────────────────────

def build_classifier() -> keras.Model:
    """Dual-input classifier: CSI window (1600x64) + periodicity vector (64,).

    The periodicity vector gives the LSTM explicit information about whether
    each subcarrier exhibits sustained rhythmic patterns (heartbeat) vs
    decaying oscillation (object / filter ringing).
    """
    csi_in = keras.Input(shape=(WINDOW_SIZE, 64), name="csi_input")
    period_in = keras.Input(shape=(64,), name="periodicity_input")

    x = keras.layers.LSTM(128, return_sequences=True)(csi_in)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dropout(0.3)(x)

    p = keras.layers.Dense(16, activation="relu")(period_in)

    merged = keras.layers.Concatenate()([x, p])
    merged = keras.layers.Dense(32, activation="relu")(merged)
    merged = keras.layers.Dropout(0.2)(merged)
    out = keras.layers.Dense(1, activation="sigmoid", name="class_out")(merged)

    model = keras.Model(inputs=[csi_in, period_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_regressor() -> keras.Model:
    x_in = keras.Input(shape=(WINDOW_SIZE, 64), name="csi_input")
    x = keras.layers.LSTM(128, return_sequences=True)(x_in)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    out = keras.layers.Dense(1, name="bpm_out")(x)
    model = keras.Model(inputs=x_in, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# ── Metrics ──────────────────────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.epochs > 10:
        args.epochs = 10
    os.makedirs(args.outdir, exist_ok=True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("Loading training data …")
    X, y_cls, y_bpm, reg_mask, ts, sessions, P = load_training(args.training_csv)
    if len(X) == 0:
        raise RuntimeError("No training rows found.")

    print(f"Loaded {len(X)} windows from {len(np.unique(sessions))} sessions.\n")

    train_idx, val_idx, test_idx = split_by_session(sessions, y_cls, seed=args.seed)
    if len(val_idx) == 0:
        raise RuntimeError("Validation split is empty. Collect more sessions.")

    X_train, P_train, y_cls_train = X[train_idx], P[train_idx], y_cls[train_idx]
    X_val, P_val, y_cls_val = X[val_idx], P[val_idx], y_cls[val_idx]

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True,
        ),
    ]

    # Compute class weights for Stage A based on train split
    n_train = len(y_cls_train)
    n_0 = np.sum(y_cls_train == 0)
    n_1 = np.sum(y_cls_train == 1)

    weight_0 = (n_train / (2.0 * n_0)) if n_0 > 0 else 1.0
    weight_1 = (n_train / (2.0 * n_1)) if n_1 > 0 else 1.0
    class_weight_dict = {0: weight_0, 1: weight_1}
    print(f"\nComputed class weights for Stage A: {class_weight_dict}")

    # ── Stage A: classification (dual-input: CSI + periodicity) ──
    print("\n=== Stage A: Classifier (dual-input) ===")
    stage_a = build_classifier()
    stage_a.fit(
        [X_train, P_train], y_cls_train,
        validation_data=([X_val, P_val], y_cls_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        verbose=2,
        callbacks=callbacks,
    )

    cls_prob_val = stage_a.predict([X_val, P_val], verbose=0).reshape(-1)
    cls_pred_val = (cls_prob_val >= 0.5).astype(np.int32)
    f1_val = binary_macro_f1(y_cls_val, cls_pred_val)

    f1_test: Dict[str, float] = {}
    if len(test_idx) > 0:
        cls_prob_test = stage_a.predict(
            [X[test_idx], P[test_idx]], verbose=0
        ).reshape(-1)
        cls_pred_test = (cls_prob_test >= 0.5).astype(np.int32)
        f1_test = binary_macro_f1(y_cls[test_idx], cls_pred_test)

    stage_a.save(os.path.join(args.outdir, "stage_a_classifier.keras"))

    # ── Stage B: BPM regression (human windows only) ──
    human_train = train_idx[(y_cls[train_idx] == 1) & (reg_mask[train_idx] == 1)]
    human_val = val_idx[(y_cls[val_idx] == 1) & (reg_mask[val_idx] == 1)]
    human_test = (
        test_idx[(y_cls[test_idx] == 1) & (reg_mask[test_idx] == 1)]
        if len(test_idx) > 0
        else np.array([], dtype=int)
    )

    reg_mae_val = None
    reg_mae_test = None
    stage_b_path = os.path.join(args.outdir, "stage_b_regressor.keras")

    if len(human_train) > 0 and len(human_val) > 0:
        X_train_h, y_train_h = X[human_train], y_bpm[human_train]
        X_val_h, y_val_h = X[human_val], y_bpm[human_val]

        print(f"\n=== Stage B: Regressor ({len(human_train)} train, "
              f"{len(human_val)} val, {len(human_test)} test) ===")
        stage_b = build_regressor()
        stage_b.fit(
            X_train_h, y_train_h,
            validation_data=(X_val_h, y_val_h),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
            callbacks=callbacks,
        )
        eval_vals = stage_b.evaluate(X_val_h, y_val_h, verbose=0)
        reg_mae_val = float(eval_vals[1]) * 100.0

        if len(human_test) > 0:
            test_vals = stage_b.evaluate(
                X[human_test], y_bpm[human_test], verbose=0
            )
            reg_mae_test = float(test_vals[1]) * 100.0

        stage_b.save(stage_b_path)
    else:
        print("\n[WARN] Not enough human windows for regressor training.")
        stage_b = None

    # ── Save metrics ──
    metrics = {
        "rows_total": int(len(X)),
        "rows_train": int(len(train_idx)),
        "rows_val": int(len(val_idx)),
        "rows_test": int(len(test_idx)),
        "rows_human_train": int(len(human_train)),
        "rows_human_val": int(len(human_val)),
        "rows_human_test": int(len(human_test)),
        "sessions_train": sorted(set(sessions[train_idx].tolist())),
        "sessions_val": sorted(set(sessions[val_idx].tolist())),
        "sessions_test": (
            sorted(set(sessions[test_idx].tolist())) if len(test_idx) > 0 else []
        ),
        "val_f1_stable_nonhuman": f1_val.get("f1_stable_nonhuman"),
        "val_f1_human": f1_val.get("f1_human"),
        "val_macro_f1": f1_val.get("macro_f1"),
        "val_bpm_mae": reg_mae_val,
        "test_f1_stable_nonhuman": f1_test.get("f1_stable_nonhuman"),
        "test_f1_human": f1_test.get("f1_human"),
        "test_macro_f1": f1_test.get("macro_f1"),
        "test_bpm_mae": reg_mae_test,
        "f1_target_met_0_90": bool(f1_val.get("macro_f1", 0) >= 0.90),
    }
    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + json.dumps(metrics, indent=2))
    print(f"\nSaved classifier → {os.path.join(args.outdir, 'stage_a_classifier.keras')}")
    if stage_b is not None:
        print(f"Saved regressor  → {stage_b_path}")
    print(f"Saved metrics    → {metrics_path}")


if __name__ == "__main__":
    main()
