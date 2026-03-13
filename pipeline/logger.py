#!/usr/bin/env python3
"""
Serial logger for phase-1 PulseFi dataset collection.

Expected RX line formats:
  CSI_PKT,rx_ts_us,seq,rssi,csi_len,amp_sc00,...,amp_sc63
  BPM,rx_ts_us,bpm_value,bpm_valid,sensor_age_ms
  STAT,...

Usage:
  python logger.py --port /dev/cu.usbserial-0001 --outdir data/day1_s1_human_rest \
      --session-id s1 --day-id day1 --record-class human --activity rest

Recording stops when you press Enter or Ctrl+C.
Markers are auto-written based on --record-class.
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from dataclasses import dataclass

import serial


AMP_COUNT = 64


def now_us() -> int:
    return int(time.time() * 1_000_000)


@dataclass
class SharedState:
    latest_rx_ts_us: int = 0
    stop: bool = False


def wait_for_stop(state: SharedState) -> None:
    """Block until Enter is pressed, then set stop flag."""
    try:
        input()  # blocks until Enter
    except EOFError:
        pass
    state.stop = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port of RX ESP32.")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud.")
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for csi_packets.csv, bpm_stream.csv, markers.csv.",
    )
    parser.add_argument("--session-id", default="session_001")
    parser.add_argument("--day-id", default="day_1")
    parser.add_argument(
        "--record-class",
        choices=["human", "stable"],
        required=True,
        help="Recording class: human or stable.",
    )
    parser.add_argument(
        "--activity",
        choices=["rest", "exercise", "none"],
        default="none",
        help="Activity type (only for human class).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=180,
        help="Recording duration in seconds (default: 180 = 3 min). Press Enter to stop early.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    csi_path = os.path.join(args.outdir, "csi_packets.csv")
    bpm_path = os.path.join(args.outdir, "bpm_stream.csv")
    markers_path = os.path.join(args.outdir, "markers.csv")
    raw_path = os.path.join(args.outdir, "serial_raw.log")

    record_class = args.record_class
    activity = args.activity
    duration_s = args.duration

    # Display banner
    if record_class == "human":
        label = f"HUMAN {activity.upper()}"
    else:
        label = "STABLE (empty room)"

    state = SharedState()
    file_lock = threading.Lock()

    csi_header = ["rx_ts_us", "seq", "rssi", "csi_len"]
    csi_header += [f"amp_sc{i:02d}" for i in range(AMP_COUNT)]

    bpm_header = ["rx_ts_us", "bpm_value", "bpm_valid", "sensor_age_ms"]
    markers_header = ["marker_ts_us", "event", "note"]

    with open(csi_path, "w", newline="") as csi_f, \
         open(bpm_path, "w", newline="") as bpm_f, \
         open(markers_path, "w", newline="") as m_f, \
         open(raw_path, "w") as raw_f:

        csi_writer = csv.writer(csi_f)
        bpm_writer = csv.writer(bpm_f)
        markers_writer = csv.writer(m_f)

        csi_writer.writerow(csi_header)
        bpm_writer.writerow(bpm_header)
        markers_writer.writerow(markers_header)
        csi_f.flush()
        bpm_f.flush()
        m_f.flush()
        raw_f.flush()

        # Marker flags: start marker is deferred until first CSI packet arrives
        # so the timestamp is in the same domain as CSI (ESP32 esp_timer_get_time).
        start_marker_written = False

        # Start stop-listener thread (waits for Enter)
        stop_thread = threading.Thread(
            target=wait_for_stop,
            args=(state,),
            daemon=True,
        )
        stop_thread.start()

        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=1,
        )

        print("")
        print(f"  ╔══════════════════════════════════════════╗")
        print(f"  ║  Recording: {label:<29s}║")
        print(f"  ║  Duration:  {duration_s // 60} min ({duration_s}s)               ║")
        print(f"  ║  Press ENTER to stop early               ║")
        print(f"  ╚══════════════════════════════════════════╝")
        print("")
        print(f"Port: {args.port} @ {args.baud}")
        print(f"Output: {args.outdir}")
        print("")

        csi_rows = 0
        bpm_rows = 0
        bad_rows = 0
        latest_bpm = -1
        latest_bpm_valid = 0
        csi_rows_last = 0
        start_wall = time.time()
        last_rx_wall = time.time()
        last_status_wall = 0.0
        last_flush_wall = time.time()

        try:
            while not state.stop:
                raw = ser.readline()
                if not raw:
                    # Periodic status even if no serial bytes are arriving.
                    now = time.time()
                    if now - last_status_wall >= 1.0:
                        csi_hz = csi_rows - csi_rows_last
                        csi_rows_last = csi_rows
                        last_status_wall = now
                        elapsed = int(now - start_wall)
                        remaining = max(0, duration_s - elapsed)
                        no_data_for = now - last_rx_wall
                        rx_state = "RECEIVING" if no_data_for < 2.0 else "NO_DATA"
                        bpm_str = f"{latest_bpm}" if latest_bpm_valid else "N/A"
                        print(
                            f"[STATUS] {elapsed}s/{duration_s}s  csi={csi_rows} ({csi_hz} Hz)  "
                            f"bpm={bpm_str}  sensor={'OK' if latest_bpm_valid else 'NO_BEAT'}  "
                            f"rx={rx_state}"
                        )
                        if elapsed >= duration_s:
                            print(f"\n  ✓ {duration_s}s recording complete. Stopping.")
                            state.stop = True
                            break
                        with file_lock:
                            csi_f.flush()
                            bpm_f.flush()
                            m_f.flush()
                            raw_f.flush()
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                last_rx_wall = time.time()
                raw_f.write(line + "\n")

                parts = line.split(",")
                tag = parts[0]

                if tag == "CSI_PKT":
                    if len(parts) < 5 + AMP_COUNT:
                        bad_rows += 1
                        continue
                    try:
                        ts_us = int(parts[1])
                        seq = int(parts[2])
                        rssi = int(parts[3])
                        csi_len = int(parts[4])
                        amps = [float(x) for x in parts[5:5 + AMP_COUNT]]
                    except ValueError:
                        bad_rows += 1
                        continue

                    state.latest_rx_ts_us = ts_us
                    # Write deferred start marker on first CSI packet.
                    if not start_marker_written and record_class == "human":
                        markers_writer.writerow([ts_us, "human_start", activity])
                        m_f.flush()
                        start_marker_written = True
                    with file_lock:
                        csi_writer.writerow([ts_us, seq, rssi, csi_len, *amps])
                    csi_rows += 1

                elif tag == "BPM":
                    if len(parts) < 5:
                        bad_rows += 1
                        continue
                    try:
                        ts_us = int(parts[1])
                        bpm = float(parts[2])
                        valid = int(parts[3])
                        age_ms = int(parts[4])
                    except ValueError:
                        bad_rows += 1
                        continue

                    state.latest_rx_ts_us = ts_us
                    latest_bpm = int(bpm)
                    latest_bpm_valid = valid
                    with file_lock:
                        bpm_writer.writerow([ts_us, bpm, valid, age_ms])
                    bpm_rows += 1

                # Ignore RX_PKT/STAT and any unknown lines for CSV writing.

                now = time.time()
                if now - last_status_wall >= 1.0:
                    csi_hz = csi_rows - csi_rows_last
                    csi_rows_last = csi_rows
                    last_status_wall = now
                    elapsed = int(now - start_wall)
                    remaining = max(0, duration_s - elapsed)
                    no_data_for = now - last_rx_wall
                    rx_state = "RECEIVING" if no_data_for < 2.0 else "NO_DATA"
                    bpm_str = f"{latest_bpm}" if latest_bpm_valid else "N/A"
                    print(
                        f"[STATUS] {elapsed}s/{duration_s}s  csi={csi_rows} ({csi_hz} Hz)  "
                        f"bpm={bpm_str}  sensor={'OK' if latest_bpm_valid else 'NO_BEAT'}  "
                        f"rx={rx_state}"
                    )
                    if elapsed >= duration_s:
                        print(f"\n  ✓ {duration_s}s recording complete. Stopping.")
                        state.stop = True
                        break

                # Periodic durability flush independent of status print timing.
                if now - last_flush_wall >= 1.0:
                    last_flush_wall = now
                    with file_lock:
                        csi_f.flush()
                        bpm_f.flush()
                        m_f.flush()
                        raw_f.flush()

        except KeyboardInterrupt:
            print("Stopping capture...")
        finally:
            state.stop = True

            # Auto-write end marker using last ESP32 CSI timestamp.
            end_ts = state.latest_rx_ts_us
            if record_class == "human" and end_ts > 0:
                markers_writer.writerow([end_ts, "human_end", activity])

            stop_thread.join(timeout=1.0)
            with file_lock:
                csi_f.flush()
                bpm_f.flush()
                m_f.flush()
                raw_f.flush()
            ser.close()

        print(
            f"Done. csi_rows={csi_rows} bpm_rows={bpm_rows} bad_rows={bad_rows} "
            f"session_id={args.session_id} day_id={args.day_id} "
            f"class={record_class} activity={activity}"
        )


if __name__ == "__main__":
    main()
