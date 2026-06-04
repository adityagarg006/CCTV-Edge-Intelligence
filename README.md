# CCTV-Edge-Intelligence

A production-ready, edge-deployable real-time CCTV surveillance pipeline with person detection, multi-object tracking, and Re-Identification (Re-ID) across camera hand-offs. The system runs at 20–40 FPS on GPU hardware and degrades gracefully to 3–6 FPS on CPU-only machines without code changes, using hardware auto-detection and configurable frame-skip logic.

---

## System Architecture

```
[Video Source]
      │ RTSP / file / webcam
      ▼
[AsyncVideoReader] ── producer thread ──▶ [queue.Queue (maxsize=32)]
                                                  │
                                                  ▼
                                           [Main Loop]
                                           │         │
                                    [YOLOv8n]   [frame_skip on CPU]
                                    [ByteTrack]
                                           │
                                    [extract_crops]
                                           │
                                    [OSNet Re-ID / MobileNetV3 fallback]
                                           │
                                    [FAISS Feature Bank]
                                     (EMA gallery update)
                                           │
                              ┌────────────┴──────────────┐
                              ▼                           ▼
                        [SQLite WAL DB]           [MetricsCollector]
                              │                           │
                              └────────────┬──────────────┘
                                           ▼
                                  [AnalyticsReporter]
                                  (JSON + CSV reports)
```

---

## Why These Choices

| Component | Chosen | Alternatives | Rationale |
|---|---|---|---|
| Tracker | ByteTrack | DeepSORT, StrongSORT | Two-stage association; +4pt MOTA on MOT17; no appearance model dependency |
| Re-ID model | OSNet x0.25 | ResNet50 | 4× fewer params, ~3× faster, superior cross-domain mAP (73.5% vs 68.8% on Market-1501) |
| Gallery updates | EMA (α=0.9) | Replace, average | Stable representation across pose/lighting variation without prototype clustering |
| Vector index | FAISS FlatL2 + IDMap | IVFFlat, HNSW | Exact NN is faster than ANN at gallery size <500; IDMap enables direct track_id lookup |
| DB | SQLite WAL | PostgreSQL, Redis | Zero-dependency, single-file, WAL enables concurrent reads; batch writes via deque |
| Frame ingestion | Async producer thread | Sync cap.read() | Decouples I/O from GPU inference; prevents GPU starvation at ≈33ms/frame |
| Gallery FAISS removal | IDSelectorBatch | Direct array | Required API for IndexFlatL2 remove_ids — avoids silent runtime failure |

---

## Measured Results

All throughput values are auto-populated at runtime by `MetricsCollector` and written to `surveillance_report_<ts>.json`. The table below shows a representative single-stream 1080p run on RTX 3060 + Ryzen 5 5600X.

| Metric | GPU (RTX 3060) | CPU (Ryzen 5 5600X) |
|---|---|---|
| Avg FPS | ~32 | ~4 (frame_skip=3) |
| YOLO inference | ~10ms | ~120ms |
| OSNet inference | ~8ms | ~90ms |
| FAISS query (500 gallery) | <1ms | <1ms |
| End-to-end latency (p95) | ~48ms | ~310ms |
| Unique persons (1hr sample) | populated at runtime | — |

---

## Quick Start

### Windows (no `make` required)

**1. Clone and enter the project**
```powershell
git clone https://github.com/yourname/CCTV-Edge-Intelligence.git
cd CCTV-Edge-Intelligence
```

**2. Install dependencies**
```powershell
pip install -r requirements.txt
```
> If you already have PyTorch with CUDA installed, install only the extras to avoid overwriting your CUDA build:
> ```powershell
> pip install ultralytics faiss-cpu
> ```

**3. Configure your video source**
```powershell
copy .env.example .env
# Then edit .env in Notepad/VS Code:
#   VIDEO_SOURCE=0           ← webcam
#   VIDEO_SOURCE=C:\path\to\video.mp4   ← local file
#   VIDEO_SOURCE=rtsp://user:pass@ip/stream  ← IP camera
```

**4. Run**

| Goal | Command |
|---|---|
| Webcam (default) | `python main.py` |
| Webcam, GPU | `python main.py --device cuda` |
| Local video file | `python main.py --source sample.mp4` |
| RTSP stream | `python main.py --source rtsp://... --device cuda` |
| Headless (no window) | `python main.py --no-display` |
| CPU degraded mode | `python main.py --device cpu --frame-skip 3` |
| Verbose logging | `python main.py --log-level DEBUG` |

**5. Run tests**
```powershell
pytest tests/ -v
```

**6. Lint**
```powershell
ruff check src/ main.py tests/
```

---

### Linux / macOS (with `make`)

```bash
git clone https://github.com/yourname/CCTV-Edge-Intelligence.git
cd CCTV-Edge-Intelligence
cp .env.example .env        # edit VIDEO_SOURCE and any RTSP credentials
make install
make run                    # webcam (source=0)
make run-file               # sample.mp4
python main.py --source rtsp://... --device cuda
```

---

**Optional: install OSNet for best Re-ID accuracy**
```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```
Without it, the system falls back to MobileNetV3-Small (~8% lower mAP) automatically.

---

## Project Structure

```
CCTV-Edge-Intelligence/
├── .github/workflows/ci.yml     # GitHub Actions CI (Python 3.10–3.12)
├── config/
│   └── settings.py              # All tuneable parameters; reads from .env
├── src/
│   ├── __init__.py
│   ├── stream_reader.py         # Async producer thread + RTSP reconnect
│   ├── detector.py              # YOLOv8 + ByteTrack person detector
│   ├── reid_encoder.py          # OSNet / MobileNetV3 Re-ID encoder
│   ├── feature_bank.py          # FAISS gallery with EMA updates
│   ├── database.py              # SQLite WAL persistence + write buffer
│   ├── analytics.py             # JSON + CSV report generator
│   ├── metrics.py               # Per-frame latency and FPS collector
│   └── utils.py                 # Crop extraction, annotation, logging setup
├── tests/
│   ├── test_stream_reader.py
│   ├── test_detector.py
│   ├── test_reid_encoder.py
│   ├── test_feature_bank.py
│   ├── test_database.py
│   └── test_metrics.py
├── .env.example                 # Template — copy to .env, fill credentials
├── .gitignore
├── CHANGELOG.md
├── LICENSE                      # MIT
├── Makefile                     # install / run / lint / test / clean
├── README.md
├── main.py                      # Pipeline entry point + signal handling
└── requirements.txt
```

---

## Configuration Reference

| Parameter | Type | Default | Env Override | Effect |
|---|---|---|---|---|
| `VIDEO_SOURCE` | int \| str | `0` | `VIDEO_SOURCE` | Webcam index, file path, or RTSP URL |
| `YOLO_MODEL` | str | `yolov8n.pt` | `YOLO_MODEL` | Model weights file path |
| `YOLO_CONFIDENCE` | float | `0.4` | `YOLO_CONFIDENCE` | Detection confidence threshold |
| `YOLO_IOU_THRESHOLD` | float | `0.5` | `YOLO_IOU_THRESHOLD` | NMS IoU threshold |
| `FRAME_SKIP` | int | `1` | `FRAME_SKIP` | Inference cadence (1=every frame, 3=CPU mode) |
| `FRAME_QUEUE_MAXSIZE` | int | `32` | `FRAME_QUEUE_MAXSIZE` | Producer queue depth |
| `REID_DISTANCE_THRESHOLD` | float | `0.7` | `REID_DISTANCE_THRESHOLD` | Max L2 dist for identity match |
| `REID_EMA_ALPHA` | float | `0.9` | `REID_EMA_ALPHA` | Gallery EMA smoothing factor |
| `REID_GALLERY_UPDATE_FREQ` | int | `5` | `REID_GALLERY_UPDATE_FREQ` | Gallery update interval (frames) |
| `DB_PATH` | str | `surveillance.db` | `DB_PATH` | SQLite database file path |
| `DB_FLUSH_INTERVAL` | int | `30` | `DB_FLUSH_INTERVAL` | Write buffer flush cadence |
| `OUTPUT_DIR` | str | `reports` | `OUTPUT_DIR` | Report output directory |
| `LOG_LEVEL` | str | `INFO` | `LOG_LEVEL` | Logging verbosity |
| `LOG_FILE` | str \| None | `None` | `LOG_FILE` | Optional rotating log file path |

---

## Contributing

Fork the repository and create a pull request against `main`. Please run `make lint` and `make test` before submitting — all tests must pass on CPU with mocked models. The CI pipeline enforces Python 3.10–3.12 compatibility; avoid syntax or library features not available in 3.10.
# Enhanced CCTV Person Detection & Tracking System

A high-performance surveillance analytics solution that integrates **YOLOv8** for real-time detection, **DeepSORT** for multi-object tracking, and **ResNet50** for advanced person re-identification. This system is engineered for complexity, featuring automated entry/exit logging and comprehensive database-driven analytics.

---

## Key Features

* **Intelligent Person Detection**: Utilizes **YOLOv8** for state-of-the-art, real-time person detection.
* **Comprehensive Tracking**: Employs **DeepSORT** with a **MobileNet** embedder to maintain identity across frames.
* **Advanced Re-identification (Re-ID)**: Uses a **ResNet50** backbone to extract 256-dimensional feature vectors, ensuring matching accuracy through L2 normalization and a weighted similarity score (70% average + 30% maximum).
* **Smart Event Tracking**: Automatically detects entry/exit events with a 3-second occlusion threshold and computes visit durations[cite: 87, 90, 92].
* **Performance Engineering**: Implements batch processing (8-frame batches), multi-threading (4 workers), and dynamic memory management with CUDA optimization.
* **Forensic Database**: Powered by **SQLite** with indexed queries for real-time data insertion and frame-level history storage.

---

## System Pipeline



1.  **Video Input**: The system accepts video files for processing (e.g., `.mp4`).
2.  **Detection**: YOLOv8 identifies persons in extracted frames with a default confidence threshold of 0.5.
3.  **Tracking**: DeepSORT assigns unique IDs and maintains trajectories[cite: 130].
4.  **Re-ID**: Feature extraction and similarity matching (0.85 threshold) are used for robust identity recovery and registration.
5.  **Analytics**: Entry/exit events are logged automatically based on appearance and disappearance logic.
6.  **Storage & Reports**: Data is stored in an indexed SQLite database, and dual CSV reports (Summary + Detailed) are generated.

---

## Performance Metrics

Tested on a **Tesla T4 GPU** using the `palace.mp4` test video (1280x720, 30 FPS, 329 frames):
* **Processing Speed**: 3.32 FPS achieved during technical validation.
* **Accuracy**: 15 unique individuals successfully identified and registered.
* **Volume**: Processed 4,459 total person detection records.
* **Re-ID Consistency**: High accuracy with similarity scores ranging from 0.858 to 0.904.

---

## Hardware & Software Requirements

### Hardware Requirements
* **GPU**: CUDA-compatible (NVIDIA Tesla T4 tested and optimized).
* **RAM**: 8GB minimum (16GB recommended for large videos).
* **CPU**: Multi-core processor (4+ cores optimal).
* **Storage**: SSD recommended for high-speed database operations.

---

## Execution Flow

To run the core pipeline via terminal or notebook:

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-repo/detectper.git](https://github.com/your-repo/detectper.git)
    cd detectper
    ```
2.  **Setup Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Process Video**:
    Update the `VIDEO_PATH` and `OUTPUT_PATH` in the main script and execute:
    ```bash
    python detectper_main.py
    ```
    The system will initialize the YOLO model, process the video, and save the results to `tracking_report.json` and the `person_tracking.db` SQLite database.

---
