"""FeatureBank (FAISS gallery) benchmark: update and query latency vs gallery size.

IVFFlat training overhead at N=500:
IVFFlat requires training a k-means clustering (nlist centroids) on at least
39*nlist training vectors before any vector can be added. At N=500 with
nlist=32 that means 1248 training vectors are needed — more than the entire
gallery. Training alone takes 50-200ms, after which search is approximate.
For gallery sizes below ~5000, IndexFlatL2 (exact) is faster end-to-end than
IVFFlat (approximate) because the training and coarse-quantisation overhead
exceeds the savings from reduced scan width. This benchmark confirms that
FlatL2 is the correct choice in the <1000 gallery regime.

Memory estimation uses float32 (4 bytes per component):
    embed_dim * gallery_size * 4 bytes / 1024 = KB
For embed_dim=512, gallery_size=1000: 512*1000*4/1024 = 2000 KB = ~2 MB.
FAISS IndexFlatL2 stores vectors contiguously so this is exact (no overhead).
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

from benchmarks.profiler import Timer, get_memory_mb

logger = logging.getLogger(__name__)

# Gallery sizes to sweep.
_GALLERY_SIZES = [10, 50, 100, 250, 500, 1000]
# Repetitions for update and query measurements.
_N_UPDATE_REPS = 100
_N_QUERY_REPS = 1000


def _random_embedding(embed_dim: int, seed: int = 0) -> np.ndarray:
    """Generate a unit-normalised random float32 embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(embed_dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def run(embed_dim: int = 512) -> dict[str, Any]:
    """Run the FeatureBank benchmark across gallery sizes.

    Steps:
    1. For each gallery_size in _GALLERY_SIZES:
       a. Create a FeatureBank with embed_dim.
       b. Pre-populate the gallery with gallery_size entries.
       c. Time _N_UPDATE_REPS update() calls (EMA update path, existing IDs).
       d. Time _N_QUERY_REPS query() calls against the populated gallery.
       e. Compute memory estimate from embed_dim * gallery_size * 4 / 1024.

    Args:
        embed_dim: Embedding dimensionality (must match the encoder in use).

    Returns:
        Dict with keys:
            embed_dim: int
            gallery_results: dict[str, dict]  -- keyed by str(gallery_size)
                Each inner dict has:
                    gallery_size: int
                    update_latency_us: dict  -- p50/p90/p95/p99/mean/std in microseconds
                    query_latency_us: dict   -- p50/p90/p95/p99/mean/std in microseconds
                    estimated_memory_kb: float
                    actual_gallery_size: int
            memory: dict
    """
    try:
        from src.feature_bank import FeatureBank
    except Exception as exc:
        logger.error("FeatureBank import failed: %s", exc)
        return {
            "embed_dim": embed_dim,
            "gallery_results": {},
            "memory": get_memory_mb(),
            "error": str(exc),
        }

    gallery_results: dict[str, dict[str, Any]] = {}

    for gallery_size in _GALLERY_SIZES:
        bank = FeatureBank(embed_dim=embed_dim)

        # Pre-populate the gallery with gallery_size entries (IDs 0..N-1).
        for reid_id in range(gallery_size):
            emb = _random_embedding(embed_dim, seed=reid_id)
            bank.update(reid_id, emb)

        # Verify population.
        actual_size = bank.size()

        # --- Update benchmark: EMA update on existing IDs ---
        # Cycling through existing IDs exercises the EMA code path (remove + re-add).
        update_latencies_us: list[float] = []
        for rep in range(_N_UPDATE_REPS):
            reid_id = rep % gallery_size
            emb = _random_embedding(embed_dim, seed=1000 + rep)
            t_start = time.perf_counter()
            bank.update(reid_id, emb)
            t_end = time.perf_counter()
            # Store in microseconds for sub-millisecond resolution.
            update_latencies_us.append((t_end - t_start) * 1e6)

        # --- Query benchmark: nearest-neighbour search ---
        query_latencies_us: list[float] = []
        for rep in range(_N_QUERY_REPS):
            query_emb = _random_embedding(embed_dim, seed=2000 + rep)
            t_start = time.perf_counter()
            _ = bank.query(query_emb, k=5)
            t_end = time.perf_counter()
            query_latencies_us.append((t_end - t_start) * 1e6)

        # Memory estimate: float32 = 4 bytes per component.
        estimated_memory_kb = embed_dim * gallery_size * 4 / 1024

        # Convert percentile dicts from ms (Timer.percentiles uses float arrays).
        def _us_percentiles(us_list: list[float]) -> dict[str, float]:
            if not us_list:
                return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "std": 0.0}
            a = np.array(us_list, dtype=np.float64)
            return {
                "p50": round(float(np.percentile(a, 50)), 3),
                "p90": round(float(np.percentile(a, 90)), 3),
                "p95": round(float(np.percentile(a, 95)), 3),
                "p99": round(float(np.percentile(a, 99)), 3),
                "mean": round(float(np.mean(a)), 3),
                "std": round(float(np.std(a)), 3),
            }

        gallery_results[str(gallery_size)] = {
            "gallery_size": gallery_size,
            "update_latency_us": _us_percentiles(update_latencies_us),
            "query_latency_us": _us_percentiles(query_latencies_us),
            "estimated_memory_kb": round(estimated_memory_kb, 2),
            "actual_gallery_size": actual_size,
        }

        logger.info(
            "gallery=%4d  update_p95=%.1fus  query_p95=%.1fus  mem=%.1fkb",
            gallery_size,
            gallery_results[str(gallery_size)]["update_latency_us"]["p95"],
            gallery_results[str(gallery_size)]["query_latency_us"]["p95"],
            estimated_memory_kb,
        )

    return {
        "embed_dim": embed_dim,
        "gallery_results": gallery_results,
        "memory": get_memory_mb(),
    }


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="FeatureBank benchmark")
    parser.add_argument("--embed-dim", type=int, default=512)
    args = parser.parse_args()

    results = run(embed_dim=args.embed_dim)
    print(json.dumps(results, indent=2))
