"""Re-ID encoder benchmark: batch-size sweep with correctness verification.

Why batching helps GPU utilisation:
A single 256x128 crop forward pass occupies only a fraction of the GPU's
streaming multiprocessors. The GPU scheduler fills remaining SM capacity with
other work, but with only one crop the occupancy ceiling is hit immediately.
Batching N crops into a single forward pass enables the GPU to execute N
feature maps in parallel across all SMs, amortising kernel-launch overhead
(~5-20us per launch) and maximising tensor-core utilisation. On an RTX 3060,
encode_batch(16) is typically 3-5x faster per crop than encode_batch(1).

Crossover point:
The crossover is the batch size where per-crop latency stops improving
significantly (diminishing returns). For most Re-ID models on mid-range GPUs
this occurs around batch=8-16. Beyond the crossover, additional crops increase
total latency proportionally without per-crop speedup — the GPU is fully
saturated. Knowing the crossover guides production batching strategy: wait for
N crops per frame before calling encode_batch, or flush immediately if N < 3.

L2 norm check:
ReIDEncoder.encode_batch applies F.normalize(p=2, dim=1). Every output row
must lie on the unit hypersphere (L2 norm = 1.0). Deviation > 0.001 indicates
a bug in the normalisation step or numerical overflow in the model.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.profiler import GPUProfiler, Timer, warmup_model, get_memory_mb

logger = logging.getLogger(__name__)

# Batch sizes to sweep.
_BATCH_SIZES = [1, 2, 4, 8, 16]
# Repetitions per batch size.
_N_REPS = 100
# Crop dimensions match OSNet training protocol.
_CROP_H = 256
_CROP_W = 128
# L2 norm tolerance — embeddings must satisfy |norm - 1.0| < _NORM_TOL.
_NORM_TOL = 0.001


def _make_random_crops(batch_size: int, seed: int = 42) -> list[np.ndarray]:
    """Generate batch_size random BGR crops of shape (_CROP_H, _CROP_W, 3)."""
    rng = np.random.default_rng(seed)
    crops = []
    for _ in range(batch_size):
        crop = rng.integers(0, 255, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        crops.append(crop)
    return crops


def run(device: str = "cpu") -> dict[str, Any]:
    """Run the Re-ID encoder benchmark across batch sizes.

    Steps:
    1. Load ReIDEncoder once (cold start timed separately).
    2. For each batch size in _BATCH_SIZES:
       a. Warm up with 10 calls.
       b. Run _N_REPS iterations with GPUProfiler.
       c. Assert L2 norm of each output row is within _NORM_TOL of 1.0.
    3. Return per-batch-size latency stats and per-crop throughput.

    Args:
        device: 'cpu' or 'cuda'.

    Returns:
        Dict with keys:
            device: str
            cold_start_ms: float
            batch_results: dict[str, dict]  -- keyed by str(batch_size)
                Each inner dict has:
                    batch_size: int
                    latency: dict  -- p50/p90/p95/p99/mean/std in ms (full batch)
                    per_crop_ms: dict  -- p50/p90/p95/p99/mean/std per crop
                    throughput_crops_per_sec: float
                    norm_check_passed: bool
                    embed_dim: int
            memory: dict
    """
    # --- Cold start ---
    cold_timer = Timer()
    cold_timer.start()
    try:
        from src.reid_encoder import ReIDEncoder
        encoder = ReIDEncoder(device=device)
    except Exception as exc:
        logger.error("ReIDEncoder load failed: %s", exc)
        return {
            "device": device,
            "cold_start_ms": -1.0,
            "batch_results": {},
            "memory": get_memory_mb(),
            "error": str(exc),
        }
    cold_timer.stop()
    cold_start_ms = cold_timer.elapsed_ms

    batch_results: dict[str, dict[str, Any]] = {}

    for batch_size in _BATCH_SIZES:
        crops = _make_random_crops(batch_size, seed=42)

        # Warmup: 10 calls to settle CUDA JIT / cuDNN.
        warmup_model(encoder.encode_batch, crops, n=10)

        profiler = GPUProfiler(use_gpu=(device == "cuda"))
        latencies: list[float] = []
        norm_check_passed = True

        total_start = time.perf_counter()
        for rep in range(_N_REPS):
            profiler.record_start()
            embeddings = encoder.encode_batch(crops)
            profiler.record_end()
            latencies.append(profiler.elapsed_ms())

            # L2 norm assertion on every output row.
            norms = np.linalg.norm(embeddings, axis=1)
            if np.any(np.abs(norms - 1.0) > _NORM_TOL):
                norm_check_passed = False
                logger.error(
                    "L2 norm check FAILED at batch_size=%d rep=%d: norms=%s",
                    batch_size, rep, norms.tolist(),
                )
        total_elapsed = time.perf_counter() - total_start

        total_crops = _N_REPS * batch_size
        throughput = total_crops / total_elapsed if total_elapsed > 0 else 0.0

        # Per-crop latencies (divide each batch latency by batch_size).
        per_crop_latencies = [ms / batch_size for ms in latencies]

        batch_results[str(batch_size)] = {
            "batch_size": batch_size,
            "latency": Timer.percentiles(latencies),
            "per_crop_ms": Timer.percentiles(per_crop_latencies),
            "throughput_crops_per_sec": round(throughput, 2),
            "norm_check_passed": norm_check_passed,
            "embed_dim": encoder.embed_dim,
        }

        logger.info(
            "batch=%d  p95=%.2fms  per_crop_p95=%.2fms  throughput=%.1f crops/s  norm_ok=%s",
            batch_size,
            batch_results[str(batch_size)]["latency"]["p95"],
            batch_results[str(batch_size)]["per_crop_ms"]["p95"],
            throughput,
            norm_check_passed,
        )

    return {
        "device": device,
        "cold_start_ms": round(cold_start_ms, 2),
        "batch_results": batch_results,
        "memory": get_memory_mb(),
    }


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Re-ID encoder benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    results = run(device=args.device)
    print(json.dumps(results, indent=2))
