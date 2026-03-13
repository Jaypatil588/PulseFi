#!/usr/bin/env python3
"""Merge many training_main.csv files into one dataset."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import List

csv.field_size_limit(sys.maxsize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob for training_main.csv files, e.g. '/path/data/*/training_main.csv'",
    )
    parser.add_argument("--output", required=True, help="Merged CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files: List[str] = sorted(glob.glob(args.input_glob))
    if not files:
        raise RuntimeError(f"No files matched: {args.input_glob}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total = 0
    written = 0
    header = None

    with open(args.output, "w", newline="") as out_f:
        writer = None
        for i, path in enumerate(files):
            with open(path, "r", newline="") as in_f:
                reader = csv.reader(in_f)
                rows = list(reader)
                if not rows:
                    continue
                h = rows[0]
                if header is None:
                    header = h
                    writer = csv.writer(out_f)
                    writer.writerow(header)
                elif h != header:
                    raise RuntimeError(f"Header mismatch in {path}")

                for row in rows[1:]:
                    total += 1
                    if len(row) != len(header):
                        continue
                    writer.writerow(row)
                    written += 1

    print(f"merged_files={len(files)} rows_total={total} rows_written={written}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
