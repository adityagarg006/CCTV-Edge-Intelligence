"""Database benchmark: flush interval sweep, WAL vs no-WAL, concurrent reads.

Measures write throughput and read latency for TrackingDatabase under various
flush intervals. Also benchmarks WAL mode vs default journal mode to quantify
the write-throughput gain from synchronous=NORMAL + WAL.

Concurrent read test uses threading to simulate the analytics reader running
while the writer is active. Without WAL, the reader blocks on every write
transaction (SQLite's default exclusive write lock). With WAL, readers and
the writer operate on separate WAL and database files concurrently.
"""
from __future__ import annotations

import logging
import os
import sqlite3
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

logger = logging.getLogger(__name__)

# Flush intervals to benchmark.
_FLUSH_INTERVALS = [1, 10, 30, 100]
# Total detection rows to insert per benchmark run.
_N_DETECTIONS = 5000
# Number of concurrent reader threads for the concurrency test.
_N_READER_THREADS = 4
# Number of SELECT queries each reader thread issues.
_N_READER_QUERIES = 100


def _make_fake_detection() -> Any:
    """Create a minimal Detection-like object without importing the full src stack."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class FakeDetection:
        track_id: int
        bbox_xyxy: np.ndarray
        confidence: float
        class_id: int

    return FakeDetection(
        track_id=1,
        bbox_xyxy=np.array([10.0, 20.0, 110.0, 220.0], dtype=np.float32),
        confidence=0.85,
        class_id=0,
    )


def _bench_flush_interval(db_path: str, flush_interval: int, wal: bool) -> dict[str, Any]:
    """Measure write throughput for a given flush_interval and WAL setting.

    Args:
        db_path: Path to a fresh temporary SQLite file.
        flush_interval: Rows between automatic flushes.
        wal: If True, configure WAL + synchronous=NORMAL. If False, default journal.

    Returns:
        Dict with total_time_s, throughput_rows_per_sec, rows_inserted.
    """
    try:
        from src.database import TrackingDatabase
    except Exception as exc:
        logger.error("TrackingDatabase import failed: %s", exc)
        return {"error": str(exc)}

    if not wal:
        # Override journal mode by patching the connection after open.
        # We create the DB with WAL disabled by setting journal_mode=DELETE.
        db = TrackingDatabase(db_path=db_path, flush_interval=flush_interval)
        # Force-switch journal mode to DELETE (no WAL) via direct connection.
        db._conn.execute("PRAGMA journal_mode=DELETE;")
        db._conn.execute("PRAGMA synchronous=FULL;")
        db._conn.commit()
    else:
        db = TrackingDatabase(db_path=db_path, flush_interval=flush_interval)

    det = _make_fake_detection()
    base_ts = time.time()

    start = time.perf_counter()
    for i in range(_N_DETECTIONS):
        db.log_detection(
            frame_id=i,
            timestamp=base_ts + i * 0.033,
            detection=det,
            reid_id=i % 50,
        )
    db.flush()
    elapsed = time.perf_counter() - start

    db.close()

    throughput = _N_DETECTIONS / elapsed if elapsed > 0 else 0.0
    return {
        "flush_interval": flush_interval,
        "wal": wal,
        "total_time_s": round(elapsed, 4),
        "throughput_rows_per_sec": round(throughput, 1),
        "rows_inserted": _N_DETECTIONS,
    }


def _bench_concurrent_reads(db_path: str) -> dict[str, Any]:
    """Measure concurrent read latency while a writer is active.

    Spawns _N_READER_THREADS reader threads that each issue _N_READER_QUERIES
    SELECT COUNT(*) queries. A writer thread inserts rows concurrently.
    Measures per-query read latency with WAL enabled.

    Returns:
        Dict with read_latency_ms (p50/p95/mean) and write_throughput_rows_per_sec.
    """
    # Pre-populate the database with some rows.
    try:
        from src.database import TrackingDatabase
    except Exception as exc:
        return {"error": str(exc)}

    db = TrackingDatabase(db_path=db_path, flush_interval=30)
    det = _make_fake_detection()
    base_ts = time.time()
    for i in range(500):
        db.log_detection(frame_id=i, timestamp=base_ts + i * 0.033, detection=det, reid_id=i % 10)
    db.flush()

    read_latencies: list[float] = []
    read_lock = threading.Lock()
    writer_done = threading.Event()

    def reader_worker() -> None:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        for _ in range(_N_READER_QUERIES):
            t0 = time.perf_counter()
            conn.execute("SELECT COUNT(*) FROM detections;").fetchone()
            t1 = time.perf_counter()
            with read_lock:
                read_latencies.append((t1 - t0) * 1000.0)
        conn.close()

    def writer_worker() -> None:
        for i in range(500, 500 + _N_DETECTIONS):
            db.log_detection(
                frame_id=i,
                timestamp=base_ts + i * 0.033,
                detection=det,
                reid_id=i % 50,
            )
        db.flush()
        writer_done.set()

    # Launch readers and writer concurrently.
    reader_threads = [threading.Thread(target=reader_worker, daemon=True) for _ in range(_N_READER_THREADS)]
    writer_thread = threading.Thread(target=writer_worker, daemon=True)

    w_start = time.perf_counter()
    for t in reader_threads:
        t.start()
    writer_thread.start()

    for t in reader_threads:
        t.join()
    writer_thread.join()
    w_elapsed = time.perf_counter() - w_start

    db.close()

    if not read_latencies:
        return {"read_latency_ms": {}, "write_throughput_rows_per_sec": 0.0}

    a = np.array(read_latencies, dtype=np.float64)
    write_throughput = _N_DETECTIONS / w_elapsed if w_elapsed > 0 else 0.0

    return {
        "n_reader_threads": _N_READER_THREADS,
        "n_reader_queries_per_thread": _N_READER_QUERIES,
        "read_latency_ms": {
            "p50": round(float(np.percentile(a, 50)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "mean": round(float(np.mean(a)), 4),
            "max": round(float(np.max(a)), 4),
        },
        "write_throughput_rows_per_sec": round(write_throughput, 1),
        "total_reads": len(read_latencies),
    }


def run() -> dict[str, Any]:
    """Run the full database benchmark suite.

    Returns:
        Dict with keys:
            flush_interval_results: list[dict]  -- one per (interval, wal) combination
            wal_vs_no_wal: dict  -- head-to-head at flush_interval=30
            concurrent_read_test: dict
            memory: dict
    """
    flush_results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Flush interval sweep ---
        for flush_interval in _FLUSH_INTERVALS:
            for wal in (True, False):
                db_path = os.path.join(tmpdir, f"bench_fi{flush_interval}_wal{int(wal)}.db")
                result = _bench_flush_interval(db_path, flush_interval, wal)
                flush_results.append(result)
                logger.info(
                    "flush_interval=%3d  wal=%s  throughput=%.0f rows/s  time=%.3fs",
                    flush_interval, wal,
                    result.get("throughput_rows_per_sec", 0),
                    result.get("total_time_s", 0),
                )

        # --- WAL vs no-WAL head-to-head at flush_interval=30 ---
        wal_result = next(
            (r for r in flush_results if r.get("flush_interval") == 30 and r.get("wal") is True), {}
        )
        no_wal_result = next(
            (r for r in flush_results if r.get("flush_interval") == 30 and r.get("wal") is False), {}
        )
        wal_tput = wal_result.get("throughput_rows_per_sec", 0)
        no_wal_tput = no_wal_result.get("throughput_rows_per_sec", 1)
        speedup = wal_tput / no_wal_tput if no_wal_tput > 0 else 0.0

        wal_vs_no_wal = {
            "flush_interval": 30,
            "wal_throughput_rows_per_sec": wal_tput,
            "no_wal_throughput_rows_per_sec": no_wal_tput,
            "wal_speedup_factor": round(speedup, 2),
        }

        # --- Concurrent read test ---
        conc_db_path = os.path.join(tmpdir, "bench_concurrent.db")
        concurrent_result = _bench_concurrent_reads(conc_db_path)

    return {
        "n_detections": _N_DETECTIONS,
        "flush_interval_results": flush_results,
        "wal_vs_no_wal": wal_vs_no_wal,
        "concurrent_read_test": concurrent_result,
        "memory": get_memory_mb(),
    }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    results = run()
    print(json.dumps(results, indent=2))
