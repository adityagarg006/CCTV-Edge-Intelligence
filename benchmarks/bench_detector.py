"""Detector benchmark: cold-start, warm latency percentiles, and throughput.

Measures the full PersonDetector.detect() call at the Python boundary because
we care about pipeline cost, not ultralytics-reported inference cost. The
ultralytics timer excludes pre/post-processing (resize, NMS, result parsing)
which can add 2-8ms on CPU. Measuring at the Python call boundary captures the
true per-frame budget consumed by the detection stage.

persist=True is required for ByteTrack state continuity across frames.
Without persist=True, ByteTrack resets its track history on every call and
cannot associate detections across frames — track_ids change every frame,
making downstream Re-ID impossible. The benchmark passes persist=True to match
production behaviour exactly.

p95 is the SLA metric, not mean. Mean latency can look healthy while the
pipeline violates real-time constraints on every 20th frame. A 30fps stream
requires each frame processed within 33ms. If p95=45ms, then 5% of frames
exceed budget — at 30fps that is 90 frames/minute arriving late, causing
visible stuttering. Mean is misleading because a few very slow frames raise
the mean only slightly, masking the tail behaviour that matters for real-time
compliance.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import numpy as np

# Ensure the project root is on sys.path so 'src' and 'config' are importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.profiler import GPUProfiler, Timer, warmup_model
from benchmarks.synthetic_video import generate_synthetic_frame

logger = logging.getLogger(__name__)

# Number of warm measurement iterations.
_N_WARM = 100
# Frame dimensions for the benchmark.
_WIDTH = 1280
_HEIGHT = 720
# Number of persons rendered in each synthetic frame.
_N_PERSONS = 5


def run(device: str = "cpu", frame_skip: int = 1) -> dict[str, Any]:
    """Run the detector benchmark and return a results dict.

    Steps:
    1. Load PersonDetector (measures cold-start / model load time).
    2. Generate a pool of synthetic frames once (generation cost excluded).
    3. Warm up with 10 calls so CUDA JIT and cuDNN auto-tuner settle.
    4. Measure _N_WARM iterations with GPUProfiler.
    5. Compute throughput as frames-per-second from total wall time.

    Args:
        device: 'cpu' or 'cuda'.
        frame_skip: frame_skip parameter forwarded to PersonDetector.

    Returns:
        Dict with keys:
            device: str
            frame_skip: int
            cold_start_ms: float  -- model load time in ms
            latency: dict  -- p50/p90/p95/p99/mean/std in ms
            throughput_fps: float
            n_detections_last_frame: int
            memory: dict  -- process_rss_mb, gpu_allocated_mb, gpu_reserved_mb
    """
    from benchmarks.profiler import get_memory_mb

    # --- Cold start: measure model load time ---
    cold_timer = Timer()
    cold_timer.start()
    try:
        from src.detector import PersonDetector
        detector = PersonDetector(device=device, frame_skip=frame_skip)
    except Exception as exc:
        logger.error("PersonDetector load failed: %s", exc)
        return {
            "device": device,
            "frame_skip": frame_skip,
            "cold_start_ms": -1.0,
            "latency": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "std": 0.0},
            "throughput_fps": 0.0,
            "n_detections_last_frame": 0,
            "memory": get_memory_mb(),
            "error": str(exc),
        }
    cold_timer.stop()
    cold_start_ms = cold_timer.elapsed_ms

    # --- Generate synthetic frames pool (not timed) ---
    frames = [
        generate_synthetic_frame(
            width=_WIDTH,
            height=_HEIGHT,
            n_persons=_N_PERSONS,
            frame_idx=i,
            seed=42,
        )
        for i in range(_N_WARM)
    ]

    # --- Warmup: 10 calls so CUDA JIT / cuDNN settle ---
    # persist=True is used in production so we warm with the same call.
    # warmup_model calls detector.detect with a single frame arg.
    warmup_model(detector.detect, frames[0], n=10)

    # --- Warm measurement loop ---
    profiler = GPUProfiler(use_gpu=(device == "cuda"))
    latencies: list[float] = []
    last_detections: list[Any] = []

    total_start = time.perf_counter()
    for i in range(_N_WARM):
        frame = frames[i % len(frames)]
        profiler.record_start()
        result = detector.detect(frame)
        profiler.record_end()
        latencies.append(profiler.elapsed_ms())
        last_detections = result
    total_elapsed = time.perf_counter() - total_start

    throughput_fps = _N_WARM / total_elapsed if total_elapsed > 0 else 0.0

    return {
        "device": device,
        "frame_skip": frame_skip,
        "cold_start_ms": round(cold_start_ms, 2),
        "latency": Timer.percentiles(latencies),
        "throughput_fps": round(throughput_fps, 2),
        "n_detections_last_frame": len(last_detections),
        "memory": get_memory_mb(),
    }


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Detector benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--frame-skip", type=int, default=1)
    args = parser.parse_args()

    results = run(device=args.device, frame_skip=args.frame_skip)
    print(json.dumps(results, indent=2))
