# Prompt: CCTV-Edge-Intelligence — Production Code Generation v2

You are a Principal AI Systems Engineer and expert Python developer. Generate a complete, production-ready, modular codebase for a high-performance CCTV surveillance and person Re-Identification pipeline. **Write every file completely — no placeholders, no `# TODO` comments, no abbreviated logic.** Every architectural decision must be justified in inline comments where relevant.

---

## Project Name
`CCTV-Edge-Intelligence`

---

## Real-Time Definition (implement to this spec)
**Real-time** in this codebase means: the system processes and displays detections with ≤150ms end-to-end latency per frame, measured from frame capture to annotated display, on a GPU-equipped machine (RTX 3060 class or better). On CPU-only hardware, the system degrades gracefully to ~3–6 FPS using frame-skip logic — it remains functional but is no longer real-time. The architecture must support both modes without code changes; hardware tier is auto-detected.

Target benchmarks (GPU):
- YOLOv8n inference: 8–15ms
- OSNet Re-ID inference: 5–12ms
- FAISS query: <1ms
- End-to-end per-frame: 25–50ms → ~20–40 FPS

---

## Full File Tree to Generate

```
CCTV-Edge-Intelligence/
├── .github/
│   └── workflows/
│       └── ci.yml
├── config/
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── stream_reader.py
│   ├── detector.py
│   ├── reid_encoder.py
│   ├── feature_bank.py
│   ├── database.py
│   ├── analytics.py
│   ├── metrics.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_stream_reader.py
│   ├── test_detector.py
│   ├── test_reid_encoder.py
│   ├── test_feature_bank.py
│   ├── test_database.py
│   └── test_metrics.py
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── Makefile
├── README.md
├── main.py
└── requirements.txt
```

---

## Architectural Decisions (Implement These Precisely)

### 1. Async Frame Ingestion — `src/stream_reader.py`

Implement an `AsyncVideoReader` class using `threading.Thread`.

**Producer thread**: calls `cv2.VideoCapture.read()` in a loop and pushes decoded frames into a `queue.Queue` with configurable `maxsize` (default 32, from `settings.FRAME_QUEUE_MAXSIZE`).

**Drop strategy**: On `queue.Full`, drop the *oldest* frame (not the newest) — implement explicitly with `try: q.get_nowait()` before `q.put_nowait(frame)`. Comment why: dropping oldest prevents the consumer from working on stale frames while the queue is perpetually full; dropping newest would cause the producer to silently discard live data while the consumer drains a lag buffer.

**RTSP reconnect**: When `cap.read()` returns `False` (network drop, end-of-stream on a file, or device disconnect), distinguish between:
- File source (`isinstance(source, str)` and not starting with `rtsp://`): log info and stop cleanly.
- RTSP source: enter an exponential backoff retry loop (1s, 2s, 4s, 8s, max 30s) calling `cap.open(source)` until success or `_stop_event` is set. Log each retry attempt at WARNING level.
- Integer webcam: log error and stop.

Comment in code why synchronous `cap.read()` in the main loop starves the GPU inference pipeline: the main thread blocks on I/O (≈33ms at 30fps) during which the GPU sits idle; the async producer keeps the queue pre-filled so the GPU pipeline always has a frame ready on demand.

**Interface**:
```python
class AsyncVideoReader:
    def __init__(self, source: int | str, maxsize: int = 32) -> None: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...  # non-blocking, timeout=0.05
    def stop(self) -> None: ...
    def qsize(self) -> int: ...
    def __enter__(self) -> "AsyncVideoReader": ...
    def __exit__(self, *args: object) -> None: ...
```

### 2. YOLOv8 + ByteTrack Detector — `src/detector.py`

Implement a thread-safe `PersonDetector` class wrapping `ultralytics.YOLO`.

**Model loading**: Load model path from `settings.YOLO_MODEL` (default `yolov8n.pt`). Accept a `device: str` constructor argument (`"cuda"`, `"cpu"`, or `"auto"`). When `"auto"`, use `torch.cuda.is_available()` to select.

**Inference**: Use `model.track(frame, tracker="bytetrack.yaml", persist=True, classes=[0], conf=settings.YOLO_CONFIDENCE, iou=settings.YOLO_IOU_THRESHOLD, device=self._device, verbose=False)`. Wrap in `threading.Lock` for future multi-stream use.

**ByteTrack justification** (in comments):
- ByteTrack's two-stage association preserves low-confidence detection states across partial occlusions, preventing identity fragmentation (a person half-occluded by a column retains their track_id).
- It avoids DeepSORT's cascade matching, which computes Mahalanobis distance using appearance features on every frame — O(N²) in detection count.
- ByteTrack's MOTA on MOT17 is ~77.8 vs DeepSORT's ~74.5, at comparable inference cost.
- ByteTrack has no external appearance model dependency, keeping the detector self-contained.

**None-safety**: `model.track()` returns `None` when no objects are detected. Handle with early return of empty list.

**Dataclass**:
```python
@dataclasses.dataclass(frozen=True)
class Detection:
    track_id: int
    bbox_xyxy: np.ndarray   # shape (4,), dtype float32
    confidence: float
    class_id: int
```

**Frame-skip logic**: Accept an optional `frame_skip: int = 1` parameter. When `frame_skip > 1`, run inference only on every Nth frame and return the last result otherwise. This is the CPU graceful-degradation mechanism.

### 3. OSNet Re-ID Encoder — `src/reid_encoder.py`

Implement a `ReIDEncoder` class with primary and fallback paths.

**Primary path** (when `torchreid` is importable):
```python
import torchreid
model = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    pretrained=True
)
model.eval()
```
Strip the classifier head by replacing `model.classifier` with `nn.Identity()`. Move to `device`.

**Fallback path** (on `ImportError`):
```python
import torchvision
model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
model.classifier = nn.Identity()  # 576-dim output
model.eval()
```
Log at WARNING: `"torchreid not found; falling back to MobileNetV3-Small (576-dim). Expect ~8% lower Re-ID accuracy. Install: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"`

**OSNet justification** (in comments): OSNet's omni-scale feature learning with unified aggregation gates achieves superior cross-domain Re-ID accuracy (mAP 73.5% on Market-1501) vs ResNet50 (mAP 68.8%), with 4× fewer parameters (~2.2M vs 23.5M) and ~3× faster inference on the same hardware.

**EMA embedding update**: `ReIDEncoder` does not manage the gallery, but it must expose the embedding dimensionality as `self.embed_dim: int` so `FeatureBank` can initialize the FAISS index correctly.

**Methods**:
```python
def encode(self, crop: np.ndarray) -> np.ndarray:
    """Preprocess BGR crop → L2-normalised float32 embedding."""

def encode_batch(self, crops: list[np.ndarray]) -> np.ndarray:
    """Batched encode; primary call path. Returns shape (N, embed_dim)."""
```

**Preprocessing**: Resize to 256×128, convert BGR→RGB, normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`, stack into batch tensor. Apply `torch.no_grad()`. L2-normalize output with `F.normalize(features, p=2, dim=1)`. Return as `np.ndarray` on CPU.

Handle empty `crops` list in `encode_batch` by returning `np.zeros((0, self.embed_dim), dtype=np.float32)`.

### 4. FAISS Feature Bank — `src/feature_bank.py`

Implement a `FeatureBank` class.

**Index construction**:
```python
flat = faiss.IndexFlatL2(embed_dim)
self._index = faiss.IndexIDMap(flat)
```

**Justification** (in comments):
- For a gallery of 10–500 identities, `IndexFlatL2` performs exact nearest-neighbor search. At this scale it is definitively faster than approximate indexes (IVFFlat, HNSW) because ANN indexes incur training and clustering overhead that only pays off beyond ~100K vectors.
- `IndexIDMap` assigns arbitrary integer track_ids as vector IDs, enabling direct lookup without a secondary mapping table.
- The FAISS vector search step completes in <1ms even on CPU. End-to-end Re-ID latency is dominated by OSNet inference (5–15ms on GPU) — documenting this distinction prevents overclaiming "sub-millisecond Re-ID."

**EMA gallery update** in `update()`:
```python
def update(self, reid_id: int, embedding: np.ndarray, alpha: float = 0.9) -> None:
```
If `reid_id` already exists in the gallery, retrieve the stored embedding, compute EMA: `new_emb = alpha * old_emb + (1 - alpha) * embedding`, re-normalize, then replace (remove + add). If new, just add. Comment why EMA: raw embeddings vary across poses and lighting; EMA smooths the gallery representation toward a stable centroid without requiring an explicit prototype computation step.

**ID removal**: Use `faiss.IDSelectorBatch` correctly:
```python
ids_to_remove = np.array([reid_id], dtype=np.int64)
selector = faiss.IDSelectorBatch(ids_to_remove)
self._index.remove_ids(selector)
```
Comment that `remove_ids` on `IndexIDMap(IndexFlatL2)` is supported but requires `IDSelectorBatch`, not a plain numpy array — a common implementation mistake.

**Methods**:
```python
def update(self, reid_id: int, embedding: np.ndarray, alpha: float = 0.9) -> None: ...
def query(self, embedding: np.ndarray, k: int = 5) -> list[tuple[int, float]]: ...
def size(self) -> int: ...
def clear(self) -> None: ...
```

`query()`: filter results where distance > `settings.REID_DISTANCE_THRESHOLD`. Return only passing results.

Thread-safe with `threading.RLock`.

### 5. Shared Utilities — `src/utils.py`

Implement standalone functions (no classes):

```python
def extract_crops(
    frame: np.ndarray,
    detections: list[Detection],
    min_crop_area: int = 500
) -> tuple[list[Detection], list[np.ndarray]]:
    """
    Extract and validate bounding box crops from frame.
    Filters detections whose crop area < min_crop_area (avoids encoding noise).
    Returns parallel lists: (valid_detections, crops).
    """

def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    reid_assignments: dict[int, int],  # track_id -> reid_id
    fps: float,
    queue_depth: int,
    queue_max: int,
) -> np.ndarray:
    """
    Draw bounding boxes, track_id/reid_id labels, and HUD overlay.
    Modifies frame in-place and returns it.
    Uses cv2.rectangle, cv2.putText with antialiased font.
    HUD: 'FPS: 28.4 | Tracked: 3 | Gallery: 7 | Queue: 12/32' in top-left corner
    with a semi-transparent background rect for readability.
    """

def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure root logger with format '%(asctime)s | %(levelname)s | %(name)s | %(message)s'.
    If log_file is provided, attach a RotatingFileHandler (maxBytes=10MB, backupCount=5).
    """

def auto_select_device() -> str:
    """Return 'cuda' if torch.cuda.is_available(), else 'cpu'."""
```

### 6. SQLite Persistence — `src/database.py`

Implement `TrackingDatabase`.

**Schema**:
```sql
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,
    frame_id    INTEGER NOT NULL,
    track_id    INTEGER NOT NULL,
    reid_id     INTEGER,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    confidence  REAL
);
CREATE TABLE IF NOT EXISTS identities (
    reid_id          INTEGER PRIMARY KEY,
    first_seen       REAL NOT NULL,
    last_seen        REAL NOT NULL,
    total_detections INTEGER DEFAULT 1,
    best_confidence  REAL
);
```

**Indexes**: Create `idx_detections_track_id ON detections(track_id)` and `idx_detections_frame_id ON detections(frame_id)`. Comment: these enable O(log N) lookup during analytics aggregation vs O(N) full-table scan.

**PRAGMAs**:
```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
```
Comment `synchronous=NORMAL`: avoids fsync on every write (acceptable for surveillance data; a crash loses at most the last few buffered frames), yielding 3–5× write throughput improvement.

**Write buffer**: internal `collections.deque` flushed every `settings.DB_FLUSH_INTERVAL` frames or on shutdown. Comment: eliminates per-frame transaction overhead.

**Methods**:
```python
def log_detection(self, frame_id: int, timestamp: float, detection: Detection, reid_id: int | None) -> None: ...
def upsert_identity(self, reid_id: int, timestamp: float, confidence: float) -> None: ...
def flush(self) -> None: ...
def close(self) -> None: ...
def __enter__(self) -> "TrackingDatabase": ...
def __exit__(self, *args: object) -> None: ...
```

### 7. Analytics Reporter — `src/analytics.py`

Implement `AnalyticsReporter`.

Constructor: `__init__(self, db: TrackingDatabase, metrics: MetricsCollector, source: str, model_name: str, index_type: str)`.

**`generate_report(output_dir: str) -> dict[str, str]`**: returns mapping of report type → file path. Creates `output_dir` if it doesn't exist. Writes three files with `<timestamp>` = `datetime.now().strftime("%Y%m%d_%H%M%S")`:

1. **`surveillance_report_<timestamp>.json`**:
```json
{
  "system_metadata": {
    "model_name": "yolov8n.pt",
    "reid_model": "osnet_x0_25",
    "index_type": "IndexFlatL2+IndexIDMap",
    "source": "rtsp://...",
    "generated_at": "<ISO8601>",
    "runtime_seconds": 3600.0
  },
  "throughput": {
    "total_frames": 108000,
    "avg_fps": 30.0,
    "min_fps": 18.2,
    "max_fps": 34.1,
    "p95_frame_latency_ms": 48.3
  },
  "tracking_summary": {
    "unique_track_ids": 142,
    "unique_reid_ids": 38,
    "total_detections": 91200,
    "avg_detections_per_frame": 2.6
  },
  "identity_profiles": [...]
}
```
All throughput values populated from `MetricsCollector.summary()` — never hardcoded.

2. **`tracking_summary_<timestamp>.csv`**: reid_id, first_seen_iso, first_seen_ts, last_seen_iso, last_seen_ts, duration_seconds, total_detections, best_confidence. ISO strings via `datetime.fromtimestamp(ts).isoformat()`.

3. **`frame_density_<timestamp>.csv`**: frame_id, timestamp_iso, timestamp_ts, detection_count. Populated from a GROUP BY query: `SELECT frame_id, timestamp, COUNT(*) FROM detections GROUP BY frame_id ORDER BY frame_id`.

### 8. Metrics Collector — `src/metrics.py`

Implement `MetricsCollector`.

```python
class MetricsCollector:
    def __init__(self) -> None:
        self._frame_latencies: deque[float] = deque(maxlen=500)
        self._fps_samples: deque[float] = deque(maxlen=100)
        self._queue_depth_samples: deque[int] = deque(maxlen=100)
        self._reid_hit_count: int = 0
        self._reid_miss_count: int = 0
        self._total_frames: int = 0
        self._start_time: float = time.monotonic()
        self._lock: threading.Lock = threading.Lock()
```

```python
def record_frame(self, latency_ms: float, fps: float, queue_depth: int) -> None: ...
def record_reid(self, hit: bool) -> None: ...
def summary(self) -> dict[str, float]: ...
    # Keys: avg_fps, min_fps, max_fps, p95_latency_ms, reid_hit_rate,
    #       total_frames_processed, runtime_seconds
```

P95 latency: `float(np.percentile(list(self._frame_latencies), 95))` — handle empty deque by returning 0.0.

Thread-safe with `threading.Lock`.

### 9. Configuration — `config/settings.py`

```python
"""
CCTV-Edge-Intelligence configuration.

Performance vs accuracy tradeoffs are documented per-parameter.
Secrets (RTSP passwords) go in .env, not here — see .env.example.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# --- Video source ---
# int = webcam index; str = file path or RTSP URL.
# RTSP URLs with credentials: rtsp://user:pass@host/stream — use .env for credentials.
VIDEO_SOURCE: int | str = int(os.getenv("VIDEO_SOURCE", "0"))

# --- Detection ---
YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
# Confidence threshold [0.0–1.0]. Lower = more detections, more false positives.
YOLO_CONFIDENCE: float = 0.4
# IoU threshold for NMS [0.0–1.0]. Higher = more overlapping boxes kept.
YOLO_IOU_THRESHOLD: float = 0.5
# Frame skip for CPU-mode graceful degradation. 1 = every frame. 3 = ~3× faster, ~3× lower FPS.
FRAME_SKIP: int = 1

# --- Async capture ---
# Queue depth. Higher = more memory, larger latency buffer. Lower = more frame drops under load.
FRAME_QUEUE_MAXSIZE: int = 32

# --- Re-ID ---
# L2 distance threshold. Lower = stricter identity matching (fewer false re-IDs).
# Typical range: 0.5 (strict) – 0.9 (lenient).
REID_DISTANCE_THRESHOLD: float = 0.7
# EMA alpha for gallery updates [0–1]. Higher = gallery changes slowly (stable but slow to adapt).
REID_EMA_ALPHA: float = 0.9
# Update gallery every N frames per track. Lower = fresher embeddings, higher CPU.
REID_GALLERY_UPDATE_FREQ: int = 5

# --- Database ---
DB_PATH: str = os.getenv("DB_PATH", "surveillance.db")
# Frames between write buffer flushes. Higher = fewer transactions, higher crash data-loss window.
DB_FLUSH_INTERVAL: int = 30

# --- Output ---
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "reports")
DISPLAY_OUTPUT: bool = True
LOG_FILE: str | None = os.getenv("LOG_FILE", None)  # None = console only
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
```

### 10. Main Orchestration — `main.py`

**argparse flags**:
- `--source`: video source (default: `settings.VIDEO_SOURCE`)
- `--no-display`: suppress cv2 window
- `--output-dir`: override report output directory
- `--device`: `cuda`, `cpu`, or `auto` (default: `auto`)
- `--frame-skip`: override `settings.FRAME_SKIP`
- `--log-level`: `DEBUG`, `INFO`, `WARNING` (default: from settings)

**Signal handling**: `signal.signal(SIGINT, handler)` and `signal.signal(SIGTERM, handler)` set a `threading.Event` `shutdown_event`. On trigger: (1) stop `AsyncVideoReader`, (2) call `db.flush()`, (3) call `reporter.generate_report()`, (4) print final stats to console. Handler must be idempotent (guard against double-signal).

**Main loop logic per frame**:
1. Call `reader.read()` → if `False`, check if source was a file (expected EOF) or RTSP (unexpected drop — log WARNING, continue).
2. `t_start = time.perf_counter()`
3. `PersonDetector.detect(frame)` → `detections: list[Detection]`
4. `utils.extract_crops(frame, detections)` → `(valid_detections, crops)`
5. `reid_encoder.encode_batch(crops)` → `embeddings` shape `(N, D)`
6. For each `(detection, embedding)`:
   - If `frame_id % settings.REID_GALLERY_UPDATE_FREQ == 0`: `feature_bank.update(detection.track_id, embedding)`
   - `results = feature_bank.query(embedding, k=1)`
   - If results and distance < threshold: `reid_id = results[0][0]`; `metrics.record_reid(hit=True)`
   - Else: assign new `reid_id = next_reid_counter`; `feature_bank.update(reid_id, embedding)` ; `metrics.record_reid(hit=False)`
   - `db.log_detection(frame_id, timestamp, detection, reid_id)`
   - `db.upsert_identity(reid_id, timestamp, detection.confidence)`
7. `latency_ms = (time.perf_counter() - t_start) * 1000`
8. `metrics.record_frame(latency_ms, fps, reader.qsize())`
9. If display: `utils.annotate_frame(...)` → `cv2.imshow(...)`; check `cv2.waitKey(1) & 0xFF == ord('q')` → set `shutdown_event`.
10. Every 30 frames: `print(f"Frame {frame_id:>6} | FPS: {fps:5.1f} | Tracked: {len(detections)} | Gallery: {feature_bank.size()} | Queue: {reader.qsize()}/{settings.FRAME_QUEUE_MAXSIZE}")`

Use context managers: `with TrackingDatabase(...) as db`, `with AsyncVideoReader(...) as reader`.

**FPS calculation**: rolling average over last 30 frame timestamps using a `deque(maxlen=30)`.

### 11. Tests — `tests/`

Use `pytest`. Each test file must be independently runnable without GPU or real video — use mocks and synthetic data.

**`tests/test_stream_reader.py`**:
- Test that `read()` returns `(False, None)` when queue is empty and timeout elapses.
- Test that producer drops oldest frame on `queue.Full` (mock `cv2.VideoCapture`).
- Test context manager `__enter__`/`__exit__` calls `stop()`.

**`tests/test_detector.py`**:
- Test that `detect()` returns `[]` when `model.track()` returns `None` (mock ultralytics YOLO).
- Test that returned objects are `Detection` dataclass instances.
- Test `frame_skip=3` returns cached result on frames 2 and 3.

**`tests/test_reid_encoder.py`**:
- Test `encode_batch([])` returns shape `(0, embed_dim)`.
- Test output is L2-normalized (`np.linalg.norm(vec) ≈ 1.0`) using a synthetic 256×128 BGR array.
- Test fallback path activates when torchreid is absent (mock `importlib`).

**`tests/test_feature_bank.py`**:
- Test `size()` returns 0 on init.
- Test `update()` then `size()` returns 1.
- Test `update()` on same ID does not increase size (replacement).
- Test `query()` returns empty list when distance exceeds threshold.
- Test `clear()` resets size to 0.
- Test thread safety: 10 concurrent threads each calling `update()` with unique IDs — assert `size() == 10` after join.

**`tests/test_database.py`**:
- Use `tmp_path` fixture for DB path.
- Test `log_detection()` + `flush()` → row appears in DB.
- Test `upsert_identity()` updates `last_seen` and increments `total_detections` on second call.
- Test context manager closes connection on exit.

**`tests/test_metrics.py`**:
- Test `summary()` returns 0.0 for all fields on fresh instance.
- Test `record_frame()` updates `avg_fps` correctly.
- Test `record_reid(hit=True)` increments hit count.
- Test P95 latency with a known distribution.

### 12. CI Pipeline — `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Lint
        run: ruff check src/ main.py tests/
      - name: Test
        run: pytest tests/ --cov=src --cov-report=xml -v
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
```

Note: CI skips GPU tests. All tests must pass on CPU with mocked models.

### 13. `.gitignore`

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/
.eggs/

# Environments
.env
.venv/
venv/
env/

# Model weights (large files — download separately)
*.pt
*.pth
*.onnx

# Runtime outputs
*.db
*.db-wal
*.db-shm
reports/
logs/

# OS
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/
*.swp
```

### 14. `.env.example`

```bash
# Copy to .env and fill in. Never commit .env.
# RTSP source with credentials (keeps passwords out of settings.py)
VIDEO_SOURCE=rtsp://admin:YOURPASSWORD@192.168.1.64/stream1

# Optional overrides
YOLO_MODEL=yolov8n.pt
DB_PATH=surveillance.db
OUTPUT_DIR=reports
LOG_LEVEL=INFO
LOG_FILE=logs/cctv.log
```

### 15. `CHANGELOG.md`

```markdown
# Changelog

All notable changes will be documented here following [Keep a Changelog](https://keepachangelog.com/) format.

## [Unreleased]

### Added
- Initial release of CCTV-Edge-Intelligence pipeline
- YOLOv8n + ByteTrack person detection and tracking
- OSNet x0.25 Re-ID with MobileNetV3 fallback
- FAISS IndexFlatL2 + IndexIDMap feature bank with EMA updates
- SQLite WAL persistence with batched writes
- Async frame producer with RTSP reconnect logic
- Metrics collector and analytics reporter
- pytest test suite for all modules
- GitHub Actions CI (Python 3.10–3.12)
```

### 16. `LICENSE`

Use MIT License. Full text with copyright year and placeholder name:
```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 17. `README.md`

Write a polished, technical README with these sections:

**1. Project Overview** — 3-sentence elevator pitch. Must mention: real-time, edge-deployable, GPU+CPU support, Re-ID across camera handoff.

**2. System Architecture** — ASCII flowchart:
```
[Video Source]
      │ RTSP/file/webcam
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

**3. Why These Choices** — markdown table:

| Component | Chosen | Alternatives | Rationale |
|---|---|---|---|
| Tracker | ByteTrack | DeepSORT, StrongSORT | Two-stage association; +4pt MOTA on MOT17; no appearance model dependency |
| Re-ID model | OSNet x0.25 | ResNet50 | 4× fewer params, ~3× faster, superior cross-domain mAP |
| Gallery updates | EMA (α=0.9) | Replace, average | Stable representation across pose/lighting variation without prototype clustering |
| Vector index | FAISS FlatL2 + IDMap | IVFFlat, HNSW | Exact NN is faster than ANN at gallery size <500; IDMap enables direct track_id lookup |
| DB | SQLite WAL | PostgreSQL, Redis | Zero-dependency, single-file, WAL enables concurrent reads; batch writes via deque |
| Frame ingestion | Async producer thread | Sync cap.read() | Decouples I/O from GPU inference; prevents GPU starvation |
| Gallery FAISS removal | IDSelectorBatch | Direct array | Required API for IndexFlatL2 remove_ids — avoids silent failure |

**4. Measured Results** — note these are auto-populated at runtime by `MetricsCollector` and `AnalyticsReporter`. Sample output shown is from a representative single-stream 1080p run on RTX 3060 + Ryzen 5 5600X:

| Metric | GPU (RTX 3060) | CPU (Ryzen 5 5600X) |
|---|---|---|
| Avg FPS | ~32 | ~4 (frame_skip=3) |
| YOLO inference | ~10ms | ~120ms |
| OSNet inference | ~8ms | ~90ms |
| FAISS query (500 gallery) | <1ms | <1ms |
| End-to-end latency (p95) | ~48ms | ~310ms |
| Unique persons (1hr sample) | populated at runtime | — |

**5. Quick Start**:
```bash
git clone https://github.com/yourname/CCTV-Edge-Intelligence.git
cd CCTV-Edge-Intelligence
cp .env.example .env        # edit VIDEO_SOURCE and any RTSP credentials
make install
make run                    # webcam
make run-file               # sample.mp4
python main.py --source rtsp://... --device cuda
```

**6. Project Structure** — annotated tree with one-line role per file.

**7. Configuration Reference** — table: parameter, type, default, env override, effect.

**8. Contributing** — one paragraph pointing to standard fork+PR workflow and asking contributors to run `make lint` and `make test` before submitting.

### 18. `requirements.txt`

```
# torchreid (optional, recommended): pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
# faiss-gpu can replace faiss-cpu for CUDA-accelerated vector search.

ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
faiss-cpu>=1.7.4
numpy>=1.24.0
python-dotenv>=1.0.0

# Dev / test (not required for runtime)
pytest>=7.4.0
pytest-cov>=4.1.0
ruff>=0.1.0
```

### 19. `Makefile`

```makefile
SOURCE ?= 0

.PHONY: install run run-file lint test clean

install:
	pip install -r requirements.txt

run:
	python main.py --source $(SOURCE)

run-file:
	python main.py --source sample.mp4 --output-dir reports

run-rtsp:
	python main.py --source $${VIDEO_SOURCE} --device cuda --output-dir reports

lint:
	ruff check src/ main.py tests/

test:
	pytest tests/ --cov=src --cov-report=term-missing -v

clean:
	rm -f *.db *.db-wal *.db-shm
	rm -rf reports/ logs/ __pycache__ src/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
```

---

## Code Quality Standards

- All classes and public methods: Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections.
- `from __future__ import annotations` at top of every file. Full type annotations on all signatures.
- No bare `except:` — always catch specific exceptions (`cv2.error`, `OSError`, `RuntimeError`, etc.).
- Use `logging` module throughout. Format: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`. The `print()` on the live stats line in `main.py` is the only acceptable exception.
- No global mutable state outside `config/settings.py`.
- Logging rotation: when `LOG_FILE` is set, use `logging.handlers.RotatingFileHandler(maxBytes=10*1024*1024, backupCount=5)`.
- All modules import from `config.settings` — never import `settings` from a relative path inside `src/`.

---

## What NOT to Generate

- No Flask/FastAPI web server.
- No Docker configuration.
- No training scripts — inference only.
- No multi-camera implementation (architecture must be extensible: `AsyncVideoReader` accepts one source; `main.py` runs one pipeline. Extensibility comment in `stream_reader.py` is sufficient).
- No ONNX export scripts.

---

## Delivery Instructions for Claude

Generate files in this order to minimize forward references:
1. `.gitignore`, `.env.example`, `LICENSE`, `CHANGELOG.md`
2. `config/settings.py`
3. `requirements.txt`, `Makefile`
4. `src/__init__.py`, `src/utils.py`, `src/metrics.py`
5. `src/stream_reader.py`, `src/detector.py`, `src/reid_encoder.py`
6. `src/feature_bank.py`, `src/database.py`, `src/analytics.py`
7. `tests/__init__.py` + all test files
8. `.github/workflows/ci.yml`
9. `main.py`
10. `README.md`

After generating all files, output a **verification checklist** confirming:
- [ ] Every file in the tree exists in your output
- [ ] No file contains `# TODO`, `pass` (except `__init__.py`), or `...` as a body placeholder
- [ ] `tests/` contains at least 3 test functions per module
- [ ] `main.py` imports from all 7 `src/` modules
- [ ] `settings.py` uses `python-dotenv` for secrets
