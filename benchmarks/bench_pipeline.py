"""End-to-end pipeline benchmark on synthetic video.

Measures per-stage latency with 5 named timers:
  1. detection       -- PersonDetector.detect()
  2. crop_extract    -- utils.extract_crops()
  3. reid_encode     -- ReIDEncoder.encode_batch()
  4. faiss_query     -- FeatureBank.query() x N detections
  5. db_write        -- TrackingDatabase.log_detection() x N detections

SLA targets:
  - Total pipeline latency <= 150ms for graceful degradation on CPU.
  - Total pipeline latency <= 50ms for real-time 20fps on GPU.

Per-stage breakdown reveals which stage dominates the budget. In a typical
GPU run: detection ~10ms, reid_encode ~8ms, faiss_query <1ms, db_write <1ms.
On CPU: detection ~120ms is the bottleneck — frame_skip is the mitigation.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from typing import Any

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.profiler import Timer, GPUProfiler, warmup_model, get_memory_mb
from benchmarks.synthetic_video import generate_synthetic_frame, generate_synthetic_video

logger = logging.getLogger(__name__)

# SLA thresholds in milliseconds.
_SLA_CPU_MS = 150.0
_SLA_GPU_MS = 50.0

# Number of frames to process in the benchmark.
_N_FRAMES = 50
# Warmup frames (excluded from measurement).
_N_WARMUP = 5
# Frame dimensions.
_WIDTH = 1280
_HEIGHT = 720
_N_PERSONS = 5


def run(device: str = "cpu", frame_skip: int = 1) -> dict[str, Any]:
    """Run the end-to-end pipeline benchmark on synthetic frames.

    Loads all pipeline components (PersonDetector, ReIDEncoder, FeatureBank,
    TrackingDatabase) and processes _N_FRAMES synthetic frames, recording
    per-stage latency for each frame.

    Args:
        device: 'cpu' or 'cuda'.
        frame_skip: Frame skip for PersonDetector.

    Returns:
        Dict with keys:
            device: str
            frame_skip: int
            n_frames: int
            per_stage_latency_ms: dict  -- keyed by stage name, value is percentile dict
            total_latency_ms: dict      -- p50/p90/p95/p99/mean/std for full frame
            sla_150ms_compliance_pct: float
            sla_50ms_compliance_pct: float
            throughput_fps: float
            avg_detections_per_frame: float
            memory: dict
            component_load_ms: dict
    """
    load_timers: dict[str, float] = {}

    # --- Load components ---
    t = Timer()

    t.start()
    try:
        from src.detector import PersonDetector
        detector = PersonDetector(device=device, frame_skip=frame_skip)
    except Exception as exc:
        logger.error("PersonDetector load failed: %s", exc)
        return {"error": f"PersonDetector: {exc}", "device": device}
    t.stop()
    load_timers["detector_ms"] = round(t.elapsed_ms, 2)

    t.start()
    try:
        from src.reid_encoder import ReIDEncoder
        encoder = ReIDEncoder(device=device)
    except Exception as exc:
        logger.error("ReIDEncoder load failed: %s", exc)
        return {"error": f"ReIDEncoder: {exc}", "device": device}
    t.stop()
    load_timers["reid_encoder_ms"] = round(t.elapsed_ms, 2)

    t.start()
    try:
        from src.feature_bank import FeatureBank
        bank = FeatureBank(embed_dim=encoder.embed_dim)
    except Exception as exc:
        logger.error("FeatureBank load failed: %s", exc)
        return {"error": f"FeatureBank: {exc}", "device": device}
    t.stop()
    load_timers["feature_bank_ms"] = round(t.elapsed_ms, 2)

    # --- Open a temporary database ---
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "bench_pipeline.db")

    t.start()
    try:
        from src.database import TrackingDatabase
        from src.utils import extract_crops
        db = TrackingDatabase(db_path=db_path, flush_interval=30)
    except Exception as exc:
        logger.error("TrackingDatabase load failed: %s", exc)
        return {"error": f"TrackingDatabase: {exc}", "device": device}
    t.stop()
    load_timers["database_ms"] = round(t.elapsed_ms, 2)

    # --- Pre-generate synthetic frames ---
    frames = [
        generate_synthetic_frame(
            width=_WIDTH, height=_HEIGHT, n_persons=_N_PERSONS, frame_idx=i, seed=42
        )
        for i in range(_N_FRAMES + _N_WARMUP)
    ]

    # Stage-level timers (CPU wall clock; GPU timer reserved for full-frame only).
    stage_names = ["detection", "crop_extract", "reid_encode", "faiss_query", "db_write"]
    stage_latencies: dict[str, list[float]] = {s: [] for s in stage_names}
    total_latencies: list[float] = []
    detection_counts: list[int] = []
    base_ts = time.time()

    # --- Warmup ---
    for wi in range(_N_WARMUP):
        detector.detect(frames[wi])

    # --- Measurement loop ---
    for fi in range(_N_FRAMES):
        frame = frames[_N_WARMUP + fi]
        frame_id = _N_WARMUP + fi
        timestamp = base_ts + frame_id * 0.033

        frame_total_start = time.perf_counter()

        # 1. Detection
        t_det_start = time.perf_counter()
        detections = detector.detect(frame)
        t_det_end = time.perf_counter()
        stage_latencies["detection"].append((t_det_end - t_det_start) * 1000.0)

        # 2. Crop extraction
        t_crop_start = time.perf_counter()
        valid_dets, crops = extract_crops(frame, detections)
        t_crop_end = time.perf_counter()
        stage_latencies["crop_extract"].append((t_crop_end - t_crop_start) * 1000.0)

        # 3. Re-ID encoding
        t_reid_start = time.perf_counter()
        embeddings = encoder.encode_batch(crops) if crops else np.zeros((0, encoder.embed_dim), dtype=np.float32)
        t_reid_end = time.perf_counter()
        stage_latencies["reid_encode"].append((t_reid_end - t_reid_start) * 1000.0)

        # 4. FAISS query (and gallery update)
        t_faiss_start = time.perf_counter()
        reid_assignments: dict[int, int] = {}
        for det, emb in zip(valid_dets, embeddings):
            matches = bank.query(emb, k=1)
            if matches:
                reid_id = matches[0][0]
            else:
                reid_id = det.track_id  # New identity.
            reid_assignments[det.track_id] = reid_id
            # Update gallery with EMA.
            bank.update(reid_id, emb)
        t_faiss_end = time.perf_counter()
        stage_latencies["faiss_query"].append((t_faiss_end - t_faiss_start) * 1000.0)

        # 5. Database write
        t_db_start = time.perf_counter()
        for det in valid_dets:
            reid_id = reid_assignments.get(det.track_id)
            db.log_detection(
                frame_id=frame_id,
                timestamp=timestamp,
                detection=det,
                reid_id=reid_id,
            )
        t_db_end = time.perf_counter()
        stage_latencies["db_write"].append((t_db_end - t_db_start) * 1000.0)

        frame_total_end = time.perf_counter()
        total_latencies.append((frame_total_end - frame_total_start) * 1000.0)
        detection_counts.append(len(detections))

    db.flush()
    db.close()

    # --- Compute results ---
    total_elapsed_s = sum(total_latencies) / 1000.0
    throughput_fps = _N_FRAMES / total_elapsed_s if total_elapsed_s > 0 else 0.0

    total_stats = Timer.percentiles(total_latencies)
    sla_cpu = sum(1 for ms in total_latencies if ms <= _SLA_CPU_MS) / len(total_latencies) * 100.0
    sla_gpu = sum(1 for ms in total_latencies if ms <= _SLA_GPU_MS) / len(total_latencies) * 100.0
    avg_detections = float(np.mean(detection_counts)) if detection_counts else 0.0

    per_stage_stats: dict[str, dict[str, float]] = {}
    for stage, lats in stage_latencies.items():
        per_stage_stats[stage] = Timer.percentiles(lats)

    return {
        "device": device,
        "frame_skip": frame_skip,
        "n_frames": _N_FRAMES,
        "per_stage_latency_ms": per_stage_stats,
        "total_latency_ms": total_stats,
        "sla_150ms_compliance_pct": round(sla_cpu, 2),
        "sla_50ms_compliance_pct": round(sla_gpu, 2),
        "throughput_fps": round(throughput_fps, 2),
        "avg_detections_per_frame": round(avg_detections, 2),
        "component_load_ms": load_timers,
        "memory": get_memory_mb(),
    }


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="End-to-end pipeline benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--frame-skip", type=int, default=1)
    args = parser.parse_args()

    results = run(device=args.device, frame_skip=args.frame_skip)
    print(json.dumps(results, indent=2))
