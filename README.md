# PulseFi

PulseFi is a hardware-first heartbeat and human-presence detection system that estimates vitals from Wi-Fi Channel State Information (CSI).

## What It Does

- Streams CSI amplitude features from an ESP32 receiver over serial.
- Uses a MAX30105 heart-rate sensor as BPM ground truth during data collection.
- Runs a two-stage ML pipeline:
  - Stage A: human presence classifier
  - Stage B: BPM regressor
- Supports live inference and runtime logging from incoming CSI packets.

## Repository Structure

- `firmware/transmitter/Transmitter.ino`: ESP32 transmitter firmware.
- `firmware/receiver/RecieverESP32.ino`: ESP32 receiver firmware with CSI callback + BPM sensor integration.
- `pipeline/`: data prep, augmentation, training, and inference scripts.
- `models/`: trained model artifacts.
- `ui/ui_dashboard.py`: dashboard for runtime visualization.
- `main.py`: orchestration entrypoint for phase-1 pipeline steps and validation.

## Core Runtime Flow

1. Collect synchronized CSI + BPM + marker streams.
2. Build training CSV windows with feature extraction.
3. Train two-stage models (presence + BPM regression).
4. Run live inference from serial CSI stream and display predictions.

## Quick Start

```bash
python3 main.py
```

Then use the interactive menu to run pipeline steps and validation.

## Notes

- The receiver expects Wi-Fi CSI and can run with or without HR sensor availability.
- Live inference expects CSI lines in this format:
  - `CSI_PKT,rx_ts_us,seq,rssi,csi_len,amp_sc00..amp_sc63`
  - `BPM,rx_ts_us,bpm_value,bpm_valid,sensor_age_ms`
