#!/usr/bin/env python3
"""Convenience CLI to run phase-1 steps."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def cmd_build_all(args: argparse.Namespace) -> None:
    data_root = args.data_root
    for name in sorted(os.listdir(data_root)):
        session_dir = os.path.join(data_root, name)
        if not os.path.isdir(session_dir):
            continue
        if not os.path.exists(os.path.join(session_dir, "csi_packets.csv")):
            continue
        run(
            [
                sys.executable,
                str(PIPELINE_DIR / "build_training_csv.py"),
                "--session-dir",
                session_dir,
                "--session-id",
                name,
                "--day-id",
                args.day_id,
                "--window-size",
                "1600",
                "--stride",
                "200",
                "--max-staleness-ms",
                "250",
            ]
        )


def cmd_merge(args: argparse.Namespace) -> None:
    run(
        [
            sys.executable,
            str(PIPELINE_DIR / "merge_training_csv.py"),
            "--input-glob",
            args.input_glob,
            "--output",
            args.output,
        ]
    )


def cmd_train(args: argparse.Namespace) -> None:
    run(
        [
            sys.executable,
            str(PIPELINE_DIR / "train_two_stage.py"),
            "--training-csv",
            args.training_csv,
            "--outdir",
            args.outdir,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ]
    )


def cmd_test(args: argparse.Namespace) -> None:
    run(
        [
            sys.executable,
            str(PIPELINE_DIR / "test_two_stage.py"),
            "--test-csv",
            args.test_csv,
            "--classifier-model",
            args.classifier_model,
            "--regressor-model",
            args.regressor_model,
            "--output-json",
            args.output_json,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-all")
    p_build.add_argument("--data-root", required=True)
    p_build.add_argument("--day-id", default="dayX")
    p_build.set_defaults(func=cmd_build_all)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--input-glob", required=True)
    p_merge.add_argument("--output", required=True)
    p_merge.set_defaults(func=cmd_merge)

    p_train = sub.add_parser("train")
    p_train.add_argument("--training-csv", required=True)
    p_train.add_argument("--outdir", required=True)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.set_defaults(func=cmd_train)

    p_test = sub.add_parser("test")
    p_test.add_argument("--test-csv", required=True)
    p_test.add_argument("--classifier-model", required=True)
    p_test.add_argument("--regressor-model", required=True)
    p_test.add_argument("--output-json", required=True)
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
