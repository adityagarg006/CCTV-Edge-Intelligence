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
_raw_source = os.getenv("VIDEO_SOURCE", "0")
try:
    VIDEO_SOURCE: int | str = int(_raw_source)
except ValueError:
    VIDEO_SOURCE = _raw_source

# --- Detection ---
YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
# Confidence threshold [0.0–1.0]. Lower = more detections, more false positives.
YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.4"))
# IoU threshold for NMS [0.0–1.0]. Higher = more overlapping boxes kept.
YOLO_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.5"))
# Frame skip for CPU-mode graceful degradation. 1 = every frame. 3 = ~3× faster, ~3× lower FPS.
FRAME_SKIP: int = int(os.getenv("FRAME_SKIP", "1"))

# --- Async capture ---
# Queue depth. Higher = more memory, larger latency buffer. Lower = more frame drops under load.
FRAME_QUEUE_MAXSIZE: int = int(os.getenv("FRAME_QUEUE_MAXSIZE", "32"))

# --- Re-ID ---
# L2 distance threshold for gallery matching.
# With OSNet x0.25 (torchreid installed):
#   same person ~0.25–0.45, different people ~0.55–1.10 → threshold 0.55 is safe.
# With MobileNetV3 fallback (no torchreid):
#   same person ~0.40–0.60, different people ~0.55–1.10 → overlap! Use 0.45.
#   Active-exclusion filtering handles co-visible people; this threshold only
#   affects returning-person matching against the exited-track gallery.
REID_DISTANCE_THRESHOLD: float = float(os.getenv("REID_DISTANCE_THRESHOLD", "0.55"))
# EMA alpha for gallery updates [0–1]. Higher = gallery changes slowly (stable but slow to adapt).
# 0.7 (vs old 0.9) lets the gallery adapt faster to pose/lighting changes.
REID_EMA_ALPHA: float = float(os.getenv("REID_EMA_ALPHA", "0.7"))
# Update gallery every N frames per track. Lower = fresher embeddings, higher CPU.
REID_GALLERY_UPDATE_FREQ: int = int(os.getenv("REID_GALLERY_UPDATE_FREQ", "5"))

# --- Database ---
DB_PATH: str = os.getenv("DB_PATH", "surveillance.db")
# Frames between write buffer flushes. Higher = fewer transactions, higher crash data-loss window.
DB_FLUSH_INTERVAL: int = int(os.getenv("DB_FLUSH_INTERVAL", "30"))

# --- Output ---
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "reports")
DISPLAY_OUTPUT: bool = os.getenv("DISPLAY_OUTPUT", "true").lower() in ("1", "true", "yes")
LOG_FILE: str | None = os.getenv("LOG_FILE") or None  # None = console only
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
