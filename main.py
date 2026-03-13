#!/usr/bin/env python3
"""
PulseFi phase-1 orchestrator.

Provides:
1) Interactive pipeline menu
2) Step-by-step execution wrappers for existing scripts
3) Validation checks after each step
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT / "pipeline"
UI_DIR = ROOT / "ui"
DOCS_DIR = ROOT / "docs"
FIRMWARE_DIR = ROOT / "firmware"
DATA_ROOT = ROOT / "data"
MODELS_ROOT = ROOT / "models" / "phase1"
RUNTIME_ROOT = ROOT / "runtime"
PYTHON_311 = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"

RAW_REQUIRED = ["csi_packets.csv", "bpm_stream.csv", "markers.csv"]
TRAIN_REQUIRED_COLS = [
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


@dataclass
class CheckResult:
    ok: bool
    message: str
    details: Optional[Dict[str, object]] = None


def run_cmd(cmd: List[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}\n")
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        # Keep main orchestrator alive when user interrupts a long-running child step.
        print("[INFO] Step interrupted by user.")


def prompt(text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{text}{suffix}: ").strip()
    if not raw and default is not None:
        return default
    return raw


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv_header(path: Path) -> Tuple[List[str], int]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], 0
    return rows[0], max(len(rows) - 1, 0)


def validate_raw_session(session_dir: Path) -> CheckResult:
    if not session_dir.exists():
        return CheckResult(False, f"Session directory not found: {session_dir}")

    missing = [name for name in RAW_REQUIRED if not (session_dir / name).exists()]
    if missing:
        return CheckResult(False, f"Missing raw files: {', '.join(missing)}")

    csi_path = session_dir / "csi_packets.csv"
    bpm_path = session_dir / "bpm_stream.csv"
    markers_path = session_dir / "markers.csv"

    csi_header, csi_rows = _read_csv_header(csi_path)
    bpm_header, bpm_rows = _read_csv_header(bpm_path)
    marker_header, marker_rows = _read_csv_header(markers_path)

    csi_expected = ["rx_ts_us", "seq", "rssi", "csi_len"] + [f"amp_sc{i:02d}" for i in range(64)]
    bpm_expected = ["rx_ts_us", "bpm_value", "bpm_valid", "sensor_age_ms"]
    marker_expected = ["marker_ts_us", "event", "note"]

    if csi_header != csi_expected:
        return CheckResult(False, "csi_packets.csv schema mismatch.")
    if bpm_header != bpm_expected:
        return CheckResult(False, "bpm_stream.csv schema mismatch.")
    if marker_header != marker_expected:
        return CheckResult(False, "markers.csv schema mismatch.")

    if csi_rows < 200:
        return CheckResult(False, f"Too few CSI rows ({csi_rows}).")
    if bpm_rows < 20:
        return CheckResult(False, f"Too few BPM rows ({bpm_rows}).")

    events = []
    with markers_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row.get("event", "").strip())

    # Human sessions must have human_start/end markers.
    # Stable sessions (no markers) are also valid.
    has_human = ("human_start" in events) and ("human_end" in events)
    is_stable = len(events) == 0

    if not has_human and not is_stable:
        return CheckResult(
            False,
            "Markers invalid (need human_start+human_end pair, or empty for stable).",
            details={"events_seen": sorted(set(events))},
        )

    return CheckResult(
        True,
        "Raw session validation passed.",
        details={"csi_rows": csi_rows, "bpm_rows": bpm_rows, "marker_rows": marker_rows},
    )


def validate_training_csv(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(False, f"training_main.csv not found: {path}")

    header, rows = _read_csv_header(path)
    if header != TRAIN_REQUIRED_COLS:
        return CheckResult(False, "training_main.csv schema mismatch.")
    if rows < 1:
        return CheckResult(False, f"Too few training rows ({rows}).")

    bad_class = 0
    bad_mask = 0
    bad_features = 0
    max_staleness = 0.0

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                cl = int(row["class_label"])
                mask = int(row["regression_mask"])
                st = float(row["label_staleness_ms_max"])
                sync_ok = int(row["sync_ok"])
                feat = json.loads(row["csi_features_json"])
            except Exception:
                return CheckResult(False, f"Invalid row parse at index {i}.")

            if cl not in (0, 1):
                bad_class += 1
            if mask not in (0, 1):
                bad_mask += 1
            if (cl == 1 and mask != 1) or (cl == 0 and mask != 0):
                bad_mask += 1
            if sync_ok != 1:
                return CheckResult(False, "Found sync_ok=0 row in training_main.csv.")
            max_staleness = max(max_staleness, st)
            if not (isinstance(feat, list) and len(feat) == 1600 and isinstance(feat[0], list) and len(feat[0]) == 64):
                bad_features += 1
                if bad_features > 3:
                    break

    if bad_class > 0:
        return CheckResult(False, f"Invalid class labels found: {bad_class}")
    if bad_mask > 0:
        return CheckResult(False, f"Invalid regression masks found: {bad_mask}")
    if bad_features > 0:
        return CheckResult(False, f"Invalid csi_features_json rows found: {bad_features}")

    return CheckResult(
        True,
        "training_main.csv validation passed.",
        details={"rows": rows, "max_staleness_ms": round(max_staleness, 3)},
    )


def validate_models(outdir: Path) -> CheckResult:
    clf = outdir / "stage_a_classifier.keras"
    reg = outdir / "stage_b_regressor.keras"
    metrics = outdir / "metrics.json"
    missing = [str(p.name) for p in (clf, reg, metrics) if not p.exists()]
    if missing:
        return CheckResult(False, f"Missing model artifacts: {', '.join(missing)}")

    with metrics.open("r") as f:
        m = json.load(f)
    if "val_macro_f1" not in m and "macro_f1" not in m:
        return CheckResult(False, "metrics.json missing val_macro_f1")
    return CheckResult(True, "Model artifacts validation passed.", details=m)


def print_check(result: CheckResult) -> None:
    tag = "PASS" if result.ok else "FAIL"
    print(f"[{tag}] {result.message}")
    if result.details:
        for k, v in result.details.items():
            print(f"  - {k}: {v}")


def step_capture() -> None:
    port = '/dev/cu.usbserial-0001'
    baud = '921600'

    day_id = prompt("Day ID", "day1")
    session_id = prompt("Session ID", "session1")

    # Recording type menu
    print("\n  Select recording type:")
    print("    1. Human (sit between ESPs, finger on sensor)")
    print("    2. Stable (empty room, step away)")
    rec_choice = prompt("Choice", "1")

    if rec_choice == "2":
        record_class = "stable"
        activity = "none"
        dir_suffix = "stable"
    else:
        record_class = "human"
        # Activity sub-menu
        print("\n  Select activity:")
        print("    1. Rest")
        print("    2. Exercise")
        act_choice = prompt("Choice", "1")
        if act_choice == "2":
            activity = "exercise"
        else:
            activity = "rest"
        dir_suffix = f"human_{activity}"

    outdir = DATA_ROOT / f"{day_id}_{session_id}_{dir_suffix}"
    ensure_dir(outdir)

    run_cmd(
        [
            PYTHON_311,
            str(PIPELINE_DIR / "logger.py"),
            "--port",
            port,
            "--baud",
            baud,
            "--outdir",
            str(outdir),
            "--session-id",
            session_id,
            "--day-id",
            day_id,
            "--record-class",
            record_class,
            "--activity",
            activity,
        ]
    )
    print_check(validate_raw_session(outdir))


def step_validate_raw() -> None:
    session_dir = Path(prompt("Session dir", str(DATA_ROOT / "day1_session1")))
    print_check(validate_raw_session(session_dir))


def step_build_session() -> None:
    session_dir = Path(prompt("Session dir", str(DATA_ROOT / "day1_session1")))
    session_id = prompt("Session ID", session_dir.name)
    day_id = prompt("Day ID", "day1")
    window_size = prompt("Window size", "1600")
    stride = prompt("Stride", "300")
    staleness = prompt("Max staleness ms", "250")

    raw_result = validate_raw_session(session_dir)
    print_check(raw_result)
    if not raw_result.ok:
        return

    run_cmd(
        [
            PYTHON_311,
            str(PIPELINE_DIR / "build_training_csv.py"),
            "--session-dir",
            str(session_dir),
            "--session-id",
            session_id,
            "--day-id",
            day_id,
            "--window-size",
            window_size,
            "--stride",
            stride,
            "--max-staleness-ms",
            "2000",
        ]
    )
    print_check(validate_training_csv(session_dir / "training_main.csv"))


def step_build_all() -> None:
    data_root = Path(prompt("Data root", str(DATA_ROOT)))
    day_id = prompt("Fallback Day ID for build-all", "dayX")
    if not data_root.exists():
        print(f"[FAIL] data root not found: {data_root}")
        return

    any_built = False
    for session_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if not (session_dir / "csi_packets.csv").exists():
            continue
        print(f"\n=== Session: {session_dir.name} ===")
        raw_result = validate_raw_session(session_dir)
        print_check(raw_result)
        if not raw_result.ok:
            continue

        run_cmd(
            [
                PYTHON_311,
                str(PIPELINE_DIR / "build_training_csv.py"),
                "--session-dir",
                str(session_dir),
                "--session-id",
                session_dir.name,
                "--day-id",
                day_id,
                "--window-size",
                "1600",
            "--stride",
            "300",
                "--max-staleness-ms",
                "2000",
            ]
        )
        tr = validate_training_csv(session_dir / "training_main.csv")
        print_check(tr)
        any_built = any_built or tr.ok

    if not any_built:
        print("[WARN] No sessions were built successfully.")


def step_merge() -> None:
    input_glob = prompt("Input glob", str(DATA_ROOT / "*" / "training_main.csv"))
    output = Path(prompt("Merged output path", str(DATA_ROOT / "all_training_main.csv")))
    ensure_dir(output.parent)

    run_cmd(
        [
            PYTHON_311,
            str(PIPELINE_DIR / "merge_training_csv.py"),
            "--input-glob",
            input_glob,
            "--output",
            str(output),
        ]
    )
    print_check(validate_training_csv(output))


def step_train_nickbild() -> None:
    """Train nickbild/csi_hr-style LSTM (100 packets, 64 subcarriers)."""
    data_root = Path(prompt("Data root", str(DATA_ROOT)))
    outdir = Path(prompt("Model output dir", str(MODELS_ROOT / "nickbild")))
    run_cmd(
        [
            PYTHON_311,
            str(PIPELINE_DIR / "train_nickbild.py"),
            "--data-root", str(data_root),
            "--outdir", str(outdir),
            "--epochs", "100",
            "--batch-size", "32",
        ]
    )
    print(f"Model saved to {outdir / 'csi_hr.keras'}")


def step_train() -> None:
    training_csv = Path(prompt("Training CSV", str(DATA_ROOT / "all_training_main.csv")))
    outdir = Path(prompt("Model output dir", str(MODELS_ROOT)))
    epochs = prompt("Epochs", "80")
    batch_size = prompt("Batch size", "32")

    tr = validate_training_csv(training_csv)
    print_check(tr)
    if not tr.ok:
        return

    ensure_dir(outdir)
    run_cmd(
        [
            PYTHON_311,
            str(PIPELINE_DIR / "train_two_stage.py"),
            "--training-csv",
            str(training_csv),
            "--outdir",
            str(outdir),
            "--epochs",
            epochs,
            "--batch-size",
            batch_size,
        ]
    )
    print_check(validate_models(outdir))


def step_test() -> None:
    test_csv = Path(prompt("Test CSV", str(DATA_ROOT / "day2_session_test" / "training_main.csv")))
    clf = Path(prompt("Classifier model", str(MODELS_ROOT / "stage_a_classifier.keras")))
    reg = Path(prompt("Regressor model", str(MODELS_ROOT / "stage_b_regressor.keras")))
    out_json = Path(prompt("Test metrics output", str(MODELS_ROOT / "test_metrics.json")))
    ensure_dir(out_json.parent)

    run_cmd(
        [
            PYTHON_311,
            str(PIPELINE_DIR / "test_two_stage.py"),
            "--test-csv",
            str(test_csv),
            "--classifier-model",
            str(clf),
            "--regressor-model",
            str(reg),
            "--output-json",
            str(out_json),
        ]
    )
    with out_json.open("r") as f:
        metrics = json.load(f)
    print("[PASS] test completed.")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")


def step_live() -> None:
    port = prompt("RX serial port", "/dev/cu.usbserial-0001")
    baud = prompt("Baud", "921600")
    clf = prompt("Classifier model", str(MODELS_ROOT / "stage_a_classifier.keras"))
    reg = prompt("Regressor model", str(MODELS_ROOT / "stage_b_regressor.keras"))
    out_csv = Path(prompt("Live output CSV", str(RUNTIME_ROOT / "live_predictions.csv")))
    stride = prompt("Inference stride (packets)", "10")
    ensure_dir(out_csv.parent)

    run_cmd(
        [
             PYTHON_311,
            str(PIPELINE_DIR / "run_live_inference.py"),
            "--port",
            port,
            "--baud",
            baud,
            "--classifier-model",
            clf,
            "--regressor-model",
            reg,
            "--out-csv",
            str(out_csv),
            "--infer-stride",
            stride,
        ]
    )


def step_dashboard() -> None:
    run_cmd([PYTHON_311, "-m", "streamlit", "run", str(PIPELINE_DIR / "dashboard.py")])


def step_augment() -> None:
    data_root = Path(prompt("Data root", str(DATA_ROOT)))
    target_bpms = prompt("Target BPM centers (comma-separated)", "75,80,85")

    run_cmd(
        [
            PYTHON_311,
            str(PIPELINE_DIR / "augment_time_stretch.py"),
            "--data-root",
            str(data_root),
            "--target-bpms",
            target_bpms,
            "--window-size",
            "1600",
            "--stride",
            "300",
            "--max-staleness-ms",
            "2000",
        ]
    )
    for t in target_bpms.split(","):
        synth_dir = data_root / f"synth_{t.strip()}bpm"
        csv_path = synth_dir / "training_main.csv"
        if csv_path.exists():
            print_check(validate_training_csv(csv_path))


def step_full_pipeline() -> None:
    print("Running full pipeline: build-all -> augment -> merge -> train")
    step_build_all()
    step_augment()
    step_merge()
    step_train()


MENU = {
    "1": ("Capture raw session", step_capture),
    "2": ("Validate raw session", step_validate_raw),
    "3": ("Build training CSV for one session", step_build_session),
    "4": ("Build training CSV for all sessions", step_build_all),
    "5": ("Generate time-stretch augmented data", step_augment),
    "6": ("Merge training CSVs", step_merge),
    "7": ("Train two-stage models", step_train),
    "8": ("Test models", step_test),
    "9": ("Train nickbild-style model (100-pkt LSTM)", step_train_nickbild),
    "10": ("Run live inference", step_live),
    "11": ("Run dashboard UI", step_dashboard),
    "12": ("Run full offline pipeline", step_full_pipeline),
}


def print_menu() -> None:
    print("\nPulseFi Main Pipeline")
    print("-" * 32)
    for k, (title, _) in MENU.items():
        print(f"{k}. {title}")
    print("0. Exit")


def main() -> None:
    ensure_dir(DATA_ROOT)
    ensure_dir(MODELS_ROOT)
    ensure_dir(RUNTIME_ROOT)
    ensure_dir(DOCS_DIR)
    ensure_dir(FIRMWARE_DIR)

    while True:
        print_menu()
        choice = input("\nChoose step: ").strip()
        if choice == "0":
            print("Exiting.")
            break
        if choice not in MENU:
            print("Invalid choice.")
            continue
        _, fn = MENU[choice]
        try:
            fn()
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] Command failed with exit code {e.returncode}.")
        except Exception as e:
            print(f"[FAIL] {e}")


if __name__ == "__main__":
    main()
