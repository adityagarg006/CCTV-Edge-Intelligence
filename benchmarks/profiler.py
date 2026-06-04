"""Timing and resource-measurement utilities for the benchmark suite."""
from __future__ import annotations
import time, logging, os
from typing import Callable
import numpy as np
import torch

logger = logging.getLogger(__name__)

class Timer:
    """High-resolution wall-clock timer usable as context manager or standalone."""
    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self._end = time.perf_counter()

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Wall-clock elapsed time in milliseconds."""
        return (self._end - self._start) * 1000.0

    @staticmethod
    def percentiles(latencies: list[float]) -> dict[str, float]:
        """Return p50/p90/p95/p99/mean/std from a list of ms latencies."""
        if not latencies:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "std": 0.0}
        a = np.array(latencies, dtype=np.float64)
        return {
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
        }


class GPUProfiler:
    """
    GPU-side latency measurement using torch.cuda.Event.

    Why CUDA Events instead of time.perf_counter?
    time.perf_counter measures CPU wall time. GPU kernels execute asynchronously —
    the CPU call returns before the GPU finishes. Calling time.perf_counter after
    model() records when the CPU *submitted* work, not when the GPU *finished* it.
    CUDA Events are recorded directly into the GPU command stream and elapsed time
    is computed on the GPU timeline, giving accurate kernel execution time.

    CRITICAL: torch.cuda.synchronize() MUST be called before reading elapsed_ms.
    The most common ML benchmark mistake is reading CUDA event time without
    synchronizing first. This causes wildly optimistic (20-100×) speedups that
    vanish in production because the CPU gets ahead of the GPU work queue.

    Falls back to CPU Timer when CUDA is unavailable.
    """

    def __init__(self, use_gpu: bool = True) -> None:
        self._use_gpu = use_gpu and torch.cuda.is_available()
        if self._use_gpu:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._timer = Timer()

    def record_start(self) -> None:
        if self._use_gpu:
            self._start_event.record()
        else:
            self._timer.start()

    def record_end(self) -> None:
        if self._use_gpu:
            self._end_event.record()
        else:
            self._timer.stop()

    def elapsed_ms(self) -> float:
        """Synchronise the GPU then return elapsed milliseconds."""
        if self._use_gpu:
            torch.cuda.synchronize()  # mandatory before reading CUDA event time
            return self._start_event.elapsed_time(self._end_event)
        return self._timer.elapsed_ms


def get_memory_mb() -> dict[str, float]:
    """Return process RSS and GPU memory in MB.

    Uses psutil if available; falls back to /proc/self/status on Linux or
    returns 0 on platforms where neither is available. psutil is NOT added
    to requirements.txt — it is treated as an optional enhancement.

    Returns:
        Dict with keys: process_rss_mb, gpu_allocated_mb, gpu_reserved_mb.
    """
    rss_mb = 0.0
    try:
        import psutil
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_mb = int(line.split()[1]) / 1024.0
                        break
        except OSError:
            pass

    gpu_alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    gpu_res = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0.0

    return {
        "process_rss_mb": round(rss_mb, 2),
        "gpu_allocated_mb": round(gpu_alloc, 2),
        "gpu_reserved_mb": round(gpu_res, 2),
    }


def warmup_model(fn: Callable, *args: object, n: int = 10) -> None:
    """Run fn(*args) n times and discard results.

    Why warmup matters:
    1. GPU kernels are JIT-compiled on the first call (CUDA PTX -> SASS).
       First-call latency can be 5-20x higher than steady state.
    2. cuDNN auto-tuner runs on first call to select the optimal convolution
       algorithm for the given input shape.
    3. PyTorch CUDA memory allocator caches from previous allocations only
       after the first forward pass — first call pays allocation overhead.
    4. CPU BLAS thread pools spin up on first GEMM call.
    Omitting warmup makes benchmarks non-representative of production steady-state.

    Args:
        fn: Callable to warm up.
        *args: Positional arguments forwarded to fn.
        n: Number of warmup iterations.
    """
    for _ in range(n):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
