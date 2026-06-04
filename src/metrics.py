"""Per-frame latency and throughput metrics collector."""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np


class MetricsCollector:
    """Thread-safe collector for pipeline throughput and Re-ID statistics.

    Stores a rolling window of frame latencies and FPS samples. All public
    methods are guarded by a single lock so callers on different threads
    can record and query concurrently without data races.
    """

    def __init__(self) -> None:
        self._frame_latencies: deque[float] = deque(maxlen=500)
        self._fps_samples: deque[float] = deque(maxlen=100)
        self._queue_depth_samples: deque[int] = deque(maxlen=100)
        self._reid_hit_count: int = 0
        self._reid_miss_count: int = 0
        self._total_frames: int = 0
        self._start_time: float = time.monotonic()
        self._lock: threading.Lock = threading.Lock()

    def record_frame(self, latency_ms: float, fps: float, queue_depth: int) -> None:
        """Record metrics for one processed frame.

        Args:
            latency_ms: End-to-end processing time for this frame in milliseconds.
            fps: Rolling FPS estimate at the time of recording.
            queue_depth: Current frame queue occupancy.
        """
        with self._lock:
            self._frame_latencies.append(latency_ms)
            self._fps_samples.append(fps)
            self._queue_depth_samples.append(queue_depth)
            self._total_frames += 1

    def record_reid(self, hit: bool) -> None:
        """Record a Re-ID gallery lookup outcome.

        Args:
            hit: ``True`` if an existing identity was matched; ``False`` if a
                new identity was assigned.
        """
        with self._lock:
            if hit:
                self._reid_hit_count += 1
            else:
                self._reid_miss_count += 1

    def summary(self) -> dict[str, float]:
        """Return a snapshot of aggregated pipeline metrics.

        Returns:
            Dictionary with keys:
            - ``avg_fps``: Mean FPS over the rolling window.
            - ``min_fps``: Minimum observed FPS.
            - ``max_fps``: Maximum observed FPS.
            - ``p95_latency_ms``: 95th-percentile frame latency in milliseconds.
            - ``reid_hit_rate``: Fraction of Re-ID queries that matched an
              existing gallery entry.
            - ``total_frames_processed``: Cumulative frame count.
            - ``runtime_seconds``: Wall-clock seconds since instantiation.
        """
        with self._lock:
            fps_list = list(self._fps_samples)
            latency_list = list(self._frame_latencies)
            total = self._reid_hit_count + self._reid_miss_count
            reid_hit_rate = self._reid_hit_count / total if total > 0 else 0.0
            runtime = time.monotonic() - self._start_time

            avg_fps = float(np.mean(fps_list)) if fps_list else 0.0
            min_fps = float(np.min(fps_list)) if fps_list else 0.0
            max_fps = float(np.max(fps_list)) if fps_list else 0.0

            # Handle empty deque gracefully — np.percentile raises on empty arrays.
            p95 = float(np.percentile(latency_list, 95)) if latency_list else 0.0

        return {
            "avg_fps": avg_fps,
            "min_fps": min_fps,
            "max_fps": max_fps,
            "p95_latency_ms": p95,
            "reid_hit_rate": reid_hit_rate,
            "total_frames_processed": float(self._total_frames),
            "runtime_seconds": runtime,
        }
