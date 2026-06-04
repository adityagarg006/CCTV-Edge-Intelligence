"""AsyncVideoReader benchmark: queue sizes vs consumer delays, drop-oldest strategy.

Drop-oldest strategy justification:
For live video sources (RTSP, webcam), the consumer's job is to process the
current state of the scene, not a historical record. When the consumer falls
behind (GPU inference takes longer than one frame period), the queue fills with
stale frames. If we drop the newest incoming frame (queue stays full of old
frames), the consumer drains aged frames and the display lags by queue_size *
frame_period seconds. If instead we drop the oldest queued frame and enqueue
the newest, the consumer always sees the most recent frame as soon as it
catches up — staleness is bounded by one frame_period regardless of how far
behind the consumer got. This matters especially on CPU (slow consumer) where
queue overflow is frequent.

For file sources, drop-oldest is intentionally disabled: blocking put() ensures
every frame is processed. Skipping frames in a recorded video loses detections.

Queue size impact: A deeper queue buffers more frames during burst GPU slowdowns
(e.g., a large scene with 20 persons), allowing the pipeline to absorb spikes
without dropping frames. But a deeper queue also increases worst-case staleness
when the consumer is slower than the producer.
"""
from __future__ import annotations

import itertools
import logging
import os
import queue
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.profiler import Timer, get_memory_mb
from benchmarks.synthetic_video import generate_synthetic_video

logger = logging.getLogger(__name__)

# Queue sizes to benchmark.
_QUEUE_SIZES = [8, 16, 32, 64]
# Consumer delays in milliseconds (simulating slow inference).
_CONSUMER_DELAYS_MS = [0, 10, 33, 100]
# Number of frames to process per combination.
_N_FRAMES = 200
# Number of frames in the synthetic video file.
_VIDEO_FRAMES = 300
# Frame dimensions for the benchmark video.
_VIDEO_WIDTH = 640
_VIDEO_HEIGHT = 480


def _bench_file_source(
    video_path: str,
    queue_size: int,
    consumer_delay_ms: int,
) -> dict[str, Any]:
    """Benchmark AsyncVideoReader on a file source.

    File sources use blocking put() — no frames are dropped. This test measures
    end-to-end throughput and read latency. Consumer delay simulates GPU
    inference time.

    Returns:
        Dict with throughput_fps, read_latency_ms, frames_read, frames_dropped.
    """
    try:
        from src.stream_reader import AsyncVideoReader
    except Exception as exc:
        return {"error": str(exc)}

    reader = AsyncVideoReader(source=video_path, maxsize=queue_size)
    consumer_delay_s = consumer_delay_ms / 1000.0

    read_latencies: list[float] = []
    frames_read = 0
    max_frames = min(_N_FRAMES, _VIDEO_FRAMES)

    total_start = time.perf_counter()
    while frames_read < max_frames:
        t0 = time.perf_counter()
        ok, frame = reader.read()
        t1 = time.perf_counter()

        if ok and frame is not None:
            frames_read += 1
            read_latencies.append((t1 - t0) * 1000.0)
            if consumer_delay_s > 0:
                time.sleep(consumer_delay_s)
        elif reader.is_done():
            break

    total_elapsed = time.perf_counter() - total_start
    reader.stop()

    throughput = frames_read / total_elapsed if total_elapsed > 0 else 0.0

    a = np.array(read_latencies, dtype=np.float64) if read_latencies else np.zeros(1)
    return {
        "source_type": "file",
        "queue_size": queue_size,
        "consumer_delay_ms": consumer_delay_ms,
        "frames_read": frames_read,
        "frames_dropped": 0,  # File sources never drop.
        "throughput_fps": round(throughput, 2),
        "read_latency_ms": {
            "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "mean": round(float(np.mean(a)), 3),
        },
        "total_time_s": round(total_elapsed, 3),
    }


def _bench_drop_oldest_strategy(
    queue_size: int,
    consumer_delay_ms: int,
    n_frames: int = 500,
) -> dict[str, Any]:
    """Test the drop-oldest strategy via direct queue simulation.

    Simulates a live producer (no sleep between enqueues) against a slow
    consumer. Counts how many frames are dropped and measures the staleness
    (age of frames when consumed) in terms of frame slots.

    The live drop-oldest strategy is implemented directly here (not through
    AsyncVideoReader) to isolate the queue behaviour from file I/O timing.

    Returns:
        Dict with frames_produced, frames_consumed, frames_dropped,
        avg_queue_depth_at_consume, drop_rate_pct.
    """
    q: queue.Queue[int] = queue.Queue(maxsize=queue_size)
    frames_dropped = 0
    frames_consumed = 0
    queue_depths_at_consume: list[int] = []
    consumer_delay_s = consumer_delay_ms / 1000.0

    stop_event = threading.Event()

    def producer() -> None:
        nonlocal frames_dropped
        for frame_idx in range(n_frames):
            if stop_event.is_set():
                break
            # Drop-oldest: if full, remove the oldest frame first.
            if q.full():
                try:
                    q.get_nowait()
                    frames_dropped += 1
                except queue.Empty:
                    pass
            try:
                q.put_nowait(frame_idx)
            except queue.Full:
                frames_dropped += 1

    def consumer() -> None:
        nonlocal frames_consumed
        # Consumer runs until producer finishes and queue is drained.
        producer_done = threading.Event()
        # We signal producer done externally via stop_event; monitor queue.
        while not (stop_event.is_set() and q.empty()):
            try:
                _ = q.get(timeout=0.1)
                queue_depths_at_consume.append(q.qsize())
                frames_consumed += 1
                if consumer_delay_s > 0:
                    time.sleep(consumer_delay_s)
            except queue.Empty:
                if stop_event.is_set():
                    break

    consumer_thread = threading.Thread(target=consumer, daemon=True)
    consumer_thread.start()

    # Run producer in main thread.
    producer()
    stop_event.set()
    consumer_thread.join(timeout=max(10.0, n_frames * consumer_delay_s + 2.0))

    drop_rate = (frames_dropped / n_frames * 100.0) if n_frames > 0 else 0.0
    avg_depth = (
        float(np.mean(queue_depths_at_consume)) if queue_depths_at_consume else 0.0
    )

    return {
        "strategy": "drop_oldest",
        "queue_size": queue_size,
        "consumer_delay_ms": consumer_delay_ms,
        "frames_produced": n_frames,
        "frames_consumed": frames_consumed,
        "frames_dropped": frames_dropped,
        "drop_rate_pct": round(drop_rate, 2),
        "avg_queue_depth_at_consume": round(avg_depth, 2),
    }


def run(skip_video_gen: bool = False) -> dict[str, Any]:
    """Run the stream reader benchmark suite.

    Tests all combinations of queue_sizes x consumer_delays using
    itertools.product.

    Args:
        skip_video_gen: If True, attempt to use an existing synthetic video.
            If no video exists and skip_video_gen=True, file source tests
            are skipped.

    Returns:
        Dict with keys:
            file_source_results: list[dict]  -- one per (queue_size, delay) combo
            drop_oldest_results: list[dict]  -- one per (queue_size, delay) combo
            summary: dict
            memory: dict
    """
    file_results: list[dict[str, Any]] = []
    drop_oldest_results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "bench_stream.mp4")

        # Generate synthetic video unless skipped.
        video_available = False
        if not skip_video_gen:
            try:
                generate_synthetic_video(
                    path=video_path,
                    n_frames=_VIDEO_FRAMES,
                    width=_VIDEO_WIDTH,
                    height=_VIDEO_HEIGHT,
                    fps=30,
                    n_persons=3,
                    seed=42,
                )
                video_available = True
            except Exception as exc:
                logger.warning("Video generation failed: %s — skipping file source tests.", exc)
        else:
            logger.info("skip_video_gen=True — skipping file source tests.")

        # --- File source benchmark: all combinations ---
        if video_available:
            for queue_size, delay_ms in itertools.product(_QUEUE_SIZES, _CONSUMER_DELAYS_MS):
                result = _bench_file_source(video_path, queue_size, delay_ms)
                file_results.append(result)
                logger.info(
                    "file  q=%2d  delay=%3dms  fps=%6.1f  reads=%d",
                    queue_size, delay_ms,
                    result.get("throughput_fps", 0),
                    result.get("frames_read", 0),
                )

        # --- Drop-oldest simulation: all combinations ---
        for queue_size, delay_ms in itertools.product(_QUEUE_SIZES, _CONSUMER_DELAYS_MS):
            result = _bench_drop_oldest_strategy(queue_size, delay_ms)
            drop_oldest_results.append(result)
            logger.info(
                "drop_oldest  q=%2d  delay=%3dms  drop_rate=%.1f%%  consumed=%d",
                queue_size, delay_ms,
                result.get("drop_rate_pct", 0),
                result.get("frames_consumed", 0),
            )

    # Summary: best (queue_size, delay) for minimal drops and acceptable throughput.
    best_fps = max((r.get("throughput_fps", 0) for r in file_results), default=0.0)
    total_combos = len(_QUEUE_SIZES) * len(_CONSUMER_DELAYS_MS)

    return {
        "queue_sizes": _QUEUE_SIZES,
        "consumer_delays_ms": _CONSUMER_DELAYS_MS,
        "n_combinations": total_combos,
        "file_source_results": file_results,
        "drop_oldest_results": drop_oldest_results,
        "summary": {
            "best_file_source_fps": round(best_fps, 2),
            "file_source_tests_run": len(file_results),
            "drop_oldest_tests_run": len(drop_oldest_results),
        },
        "memory": get_memory_mb(),
    }


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Stream reader benchmark")
    parser.add_argument("--skip-video-gen", action="store_true")
    args = parser.parse_args()

    results = run(skip_video_gen=args.skip_video_gen)
    print(json.dumps(results, indent=2))
