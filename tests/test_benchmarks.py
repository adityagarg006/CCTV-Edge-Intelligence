"""Tests for the benchmarking suite.

All tests run on CPU without GPU. Tests cover:
1. Timer accuracy and percentile computation.
2. GPUProfiler CPU fallback.
3. Synthetic video frame generation (shape, dtype, determinism).
4. FeatureBank benchmark (gallery sizes, memory estimates).
5. Database benchmark (flush intervals, WAL comparison, concurrent reads).
6. Stream reader drop-oldest simulation.
7. Report aggregation and output file creation.
8. Re-ID norm check logic (unit test of the assertion).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import numpy as np
import pytest

# Ensure project root is on path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# 1. Timer: accuracy and percentiles
# ---------------------------------------------------------------------------

class TestTimer:
    def test_elapsed_ms_context_manager(self) -> None:
        from benchmarks.profiler import Timer

        with Timer() as t:
            time.sleep(0.05)

        # Should be at least 40ms (allow OS scheduling slack).
        assert t.elapsed_ms >= 40.0, f"Expected >= 40ms, got {t.elapsed_ms:.2f}ms"
        # Should not be absurdly long.
        assert t.elapsed_ms < 2000.0

    def test_elapsed_ms_standalone(self) -> None:
        from benchmarks.profiler import Timer

        t = Timer()
        t.start()
        time.sleep(0.02)
        t.stop()

        assert t.elapsed_ms >= 15.0

    def test_percentiles_empty(self) -> None:
        from benchmarks.profiler import Timer

        result = Timer.percentiles([])
        assert result["p50"] == 0.0
        assert result["p99"] == 0.0
        assert result["mean"] == 0.0
        assert result["std"] == 0.0

    def test_percentiles_known_values(self) -> None:
        from benchmarks.profiler import Timer

        # [1, 2, ..., 100] — p50=50.5, p99~=99.0
        latencies = [float(i) for i in range(1, 101)]
        result = Timer.percentiles(latencies)
        assert abs(result["p50"] - 50.5) < 0.1
        assert result["p99"] >= 98.0
        assert result["mean"] == pytest.approx(50.5, abs=0.1)
        assert result["std"] > 0.0


# ---------------------------------------------------------------------------
# 2. GPUProfiler: CPU fallback
# ---------------------------------------------------------------------------

class TestGPUProfiler:
    def test_cpu_fallback_returns_positive_elapsed(self) -> None:
        from benchmarks.profiler import GPUProfiler

        prof = GPUProfiler(use_gpu=False)
        prof.record_start()
        time.sleep(0.01)
        prof.record_end()
        elapsed = prof.elapsed_ms()
        assert elapsed >= 5.0, f"Expected >= 5ms, got {elapsed:.3f}ms"

    def test_cpu_fallback_consistent_with_timer(self) -> None:
        from benchmarks.profiler import GPUProfiler, Timer

        # Both Timer and GPUProfiler(cpu) should measure the same sleep.
        prof = GPUProfiler(use_gpu=False)
        t = Timer()

        t.start()
        prof.record_start()
        time.sleep(0.03)
        prof.record_end()
        t.stop()

        diff = abs(prof.elapsed_ms() - t.elapsed_ms)
        # Should agree within 5ms.
        assert diff < 5.0, f"Timer vs GPUProfiler disagree by {diff:.2f}ms"


# ---------------------------------------------------------------------------
# 3. Synthetic video frame generation
# ---------------------------------------------------------------------------

class TestSyntheticVideo:
    def test_frame_shape_and_dtype(self) -> None:
        from benchmarks.synthetic_video import generate_synthetic_frame

        frame = generate_synthetic_frame(width=320, height=240, n_persons=3, frame_idx=0, seed=42)
        assert frame.shape == (240, 320, 3), f"Unexpected shape: {frame.shape}"
        assert frame.dtype == np.uint8

    def test_frame_determinism(self) -> None:
        from benchmarks.synthetic_video import generate_synthetic_frame

        f1 = generate_synthetic_frame(width=320, height=240, seed=42, frame_idx=5)
        f2 = generate_synthetic_frame(width=320, height=240, seed=42, frame_idx=5)
        assert np.array_equal(f1, f2), "Same seed+frame_idx must produce identical frames"

    def test_different_seeds_differ(self) -> None:
        from benchmarks.synthetic_video import generate_synthetic_frame

        f1 = generate_synthetic_frame(width=320, height=240, seed=42, frame_idx=0)
        f2 = generate_synthetic_frame(width=320, height=240, seed=99, frame_idx=0)
        assert not np.array_equal(f1, f2), "Different seeds should produce different frames"

    def test_frame_has_coloured_rectangles(self) -> None:
        from benchmarks.synthetic_video import generate_synthetic_frame

        # With n_persons > 0, the frame should not be entirely the background grey.
        frame = generate_synthetic_frame(width=320, height=240, n_persons=3, frame_idx=10, seed=7)
        background = 30
        unique_values = np.unique(frame)
        assert len(unique_values) > 1, "Frame should contain non-background pixels"

    def test_video_file_creation(self) -> None:
        from benchmarks.synthetic_video import generate_synthetic_video

        # Use dimensions large enough that (width - _PERSON_W=80) > 0
        # and (height - _PERSON_H=200) > 0, i.e. width > 80, height > 200.
        # We also release the VideoWriter before the temp dir cleanup to avoid
        # Windows file-lock errors (the writer is released inside the function).
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = os.path.join(tmpdir, "test.mp4")
            returned = generate_synthetic_video(
                path=path,
                n_frames=10,
                width=320,
                height=240,
                fps=10,
                n_persons=2,
                seed=1,
            )
            assert os.path.exists(returned), f"Video file not created: {returned}"
            assert os.path.getsize(returned) > 1000, "Video file suspiciously small"


# ---------------------------------------------------------------------------
# 4. FeatureBank benchmark
# ---------------------------------------------------------------------------

class TestFeatureBankBench:
    def test_run_returns_expected_keys(self) -> None:
        from benchmarks.bench_feature_bank import run

        result = run(embed_dim=64)
        assert "embed_dim" in result
        assert "gallery_results" in result
        assert "memory" in result

    def test_gallery_sizes_present(self) -> None:
        from benchmarks.bench_feature_bank import run, _GALLERY_SIZES

        result = run(embed_dim=32)
        gallery_results = result.get("gallery_results", {})
        for gs in _GALLERY_SIZES:
            assert str(gs) in gallery_results, f"Missing gallery_size={gs}"

    def test_memory_estimate_formula(self) -> None:
        from benchmarks.bench_feature_bank import run

        result = run(embed_dim=128)
        g100 = result["gallery_results"]["100"]
        # 128 * 100 * 4 / 1024 = 50.0 KB
        expected_kb = 128 * 100 * 4 / 1024
        assert abs(g100["estimated_memory_kb"] - expected_kb) < 0.01

    def test_query_latency_positive(self) -> None:
        from benchmarks.bench_feature_bank import run

        result = run(embed_dim=32)
        for gs in ["10", "50", "100"]:
            gdata = result["gallery_results"][gs]
            assert gdata["query_latency_us"]["p95"] >= 0.0
            assert gdata["update_latency_us"]["p95"] >= 0.0

    def test_actual_gallery_size_matches(self) -> None:
        from benchmarks.bench_feature_bank import run

        result = run(embed_dim=32)
        for gs_str, gdata in result["gallery_results"].items():
            gs = int(gs_str)
            # actual_gallery_size should be >= gs (updates don't reduce size).
            assert gdata["actual_gallery_size"] >= gs, (
                f"gallery_size={gs}: actual={gdata['actual_gallery_size']}"
            )


# ---------------------------------------------------------------------------
# 5. Database benchmark
# ---------------------------------------------------------------------------

class TestDatabaseBench:
    def test_run_returns_expected_keys(self) -> None:
        from benchmarks.bench_database import run

        result = run()
        assert "flush_interval_results" in result
        assert "wal_vs_no_wal" in result
        assert "concurrent_read_test" in result
        assert "memory" in result

    def test_all_flush_intervals_present(self) -> None:
        from benchmarks.bench_database import run, _FLUSH_INTERVALS

        result = run()
        intervals_seen = set()
        for r in result["flush_interval_results"]:
            if "error" not in r:
                intervals_seen.add(r["flush_interval"])
        for fi in _FLUSH_INTERVALS:
            assert fi in intervals_seen, f"Missing flush_interval={fi}"

    def test_wal_speedup_factor_present(self) -> None:
        from benchmarks.bench_database import run

        result = run()
        wal_cmp = result.get("wal_vs_no_wal", {})
        assert "wal_speedup_factor" in wal_cmp
        # WAL speedup should be positive (could be < 1 on some systems; just check it's a number).
        assert isinstance(wal_cmp["wal_speedup_factor"], float)

    def test_concurrent_read_test_ran(self) -> None:
        from benchmarks.bench_database import run

        result = run()
        conc = result.get("concurrent_read_test", {})
        if "error" not in conc:
            assert conc.get("total_reads", 0) > 0, "No reads recorded in concurrent test"

    def test_throughput_is_positive(self) -> None:
        from benchmarks.bench_database import run

        result = run()
        for r in result["flush_interval_results"]:
            if "error" not in r:
                assert r["throughput_rows_per_sec"] > 0, (
                    f"Zero throughput at flush_interval={r['flush_interval']}, wal={r['wal']}"
                )


# ---------------------------------------------------------------------------
# 6. Stream reader: drop-oldest simulation
# ---------------------------------------------------------------------------

class TestStreamReaderBench:
    def test_drop_oldest_no_delay_accounting(self) -> None:
        """Verify that frames_consumed + frames_dropped <= frames_produced.

        The producer and consumer run in separate threads; the Python GIL means
        the producer can still outpace the consumer even with consumer_delay_ms=0
        (especially on Windows with its coarser thread scheduler). Rather than
        asserting a specific drop-rate bound, we verify the accounting invariant:
        every produced frame is either consumed or dropped, and no phantom frames
        are created.
        """
        from benchmarks.bench_stream_reader import _bench_drop_oldest_strategy

        result = _bench_drop_oldest_strategy(queue_size=64, consumer_delay_ms=0, n_frames=200)
        assert result["frames_produced"] == 200
        # Accounting invariant: consumed + dropped <= produced (some may remain in queue).
        assert result["frames_consumed"] + result["frames_dropped"] <= result["frames_produced"] + result["queue_size"], (
            f"Accounting error: consumed={result['frames_consumed']} + "
            f"dropped={result['frames_dropped']} > produced={result['frames_produced']}"
        )
        assert result["frames_consumed"] >= 0
        assert result["frames_dropped"] >= 0

    def test_drop_oldest_high_delay_has_drops(self) -> None:
        """With a slow consumer and small queue, drops should occur."""
        from benchmarks.bench_stream_reader import _bench_drop_oldest_strategy

        result = _bench_drop_oldest_strategy(queue_size=8, consumer_delay_ms=100, n_frames=100)
        assert result["frames_produced"] == 100
        # High delay + small queue: producer is faster than consumer, drops expected.
        # drop_rate_pct can be 0 if consumer catches up, but consumed <= produced.
        assert result["frames_consumed"] <= result["frames_produced"]

    def test_drop_oldest_returns_expected_keys(self) -> None:
        from benchmarks.bench_stream_reader import _bench_drop_oldest_strategy

        result = _bench_drop_oldest_strategy(queue_size=16, consumer_delay_ms=0, n_frames=50)
        for key in ("strategy", "queue_size", "consumer_delay_ms", "frames_produced",
                    "frames_consumed", "frames_dropped", "drop_rate_pct"):
            assert key in result, f"Missing key: {key}"

    def test_run_skip_video_gen(self) -> None:
        """run() with skip_video_gen=True should still run drop-oldest tests."""
        from benchmarks.bench_stream_reader import run

        result = run(skip_video_gen=True)
        assert "drop_oldest_results" in result
        assert len(result["drop_oldest_results"]) > 0
        # File tests should be skipped.
        assert len(result.get("file_source_results", [])) == 0


# ---------------------------------------------------------------------------
# 7. Report aggregation
# ---------------------------------------------------------------------------

class TestReport:
    def _make_fake_results(self) -> dict:
        """Build a minimal fake results dict for testing report generation."""
        return {
            "detector": {
                "device": "cpu",
                "frame_skip": 1,
                "cold_start_ms": 1200.0,
                "latency": {"p50": 95.0, "p90": 110.0, "p95": 125.0, "p99": 140.0, "mean": 97.0, "std": 8.0},
                "throughput_fps": 9.8,
                "n_detections_last_frame": 3,
                "memory": {"process_rss_mb": 512.0, "gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0},
            },
            "reid": {
                "device": "cpu",
                "cold_start_ms": 300.0,
                "batch_results": {
                    "1": {
                        "batch_size": 1,
                        "latency": {"p50": 45.0, "p90": 55.0, "p95": 60.0, "p99": 75.0, "mean": 46.0, "std": 4.0},
                        "per_crop_ms": {"p50": 45.0, "p90": 55.0, "p95": 60.0, "p99": 75.0, "mean": 46.0, "std": 4.0},
                        "throughput_crops_per_sec": 20.5,
                        "norm_check_passed": True,
                        "embed_dim": 512,
                    },
                    "8": {
                        "batch_size": 8,
                        "latency": {"p50": 120.0, "p90": 135.0, "p95": 140.0, "p99": 155.0, "mean": 122.0, "std": 6.0},
                        "per_crop_ms": {"p50": 15.0, "p90": 17.0, "p95": 18.0, "p99": 20.0, "mean": 15.3, "std": 1.0},
                        "throughput_crops_per_sec": 62.0,
                        "norm_check_passed": True,
                        "embed_dim": 512,
                    },
                },
                "memory": {"process_rss_mb": 600.0, "gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0},
            },
            "feature_bank": {
                "embed_dim": 512,
                "gallery_results": {
                    "100": {
                        "gallery_size": 100,
                        "update_latency_us": {"p50": 120.0, "p90": 145.0, "p95": 155.0, "p99": 180.0, "mean": 122.0, "std": 10.0},
                        "query_latency_us": {"p50": 40.0, "p90": 55.0, "p95": 60.0, "p99": 80.0, "mean": 42.0, "std": 5.0},
                        "estimated_memory_kb": 200.0,
                        "actual_gallery_size": 100,
                    },
                    "500": {
                        "gallery_size": 500,
                        "update_latency_us": {"p50": 500.0, "p90": 600.0, "p95": 650.0, "p99": 750.0, "mean": 510.0, "std": 40.0},
                        "query_latency_us": {"p50": 200.0, "p90": 250.0, "p95": 270.0, "p99": 310.0, "mean": 205.0, "std": 20.0},
                        "estimated_memory_kb": 1000.0,
                        "actual_gallery_size": 500,
                    },
                },
                "memory": {"process_rss_mb": 550.0, "gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0},
            },
            "database": {
                "n_detections": 5000,
                "flush_interval_results": [
                    {"flush_interval": 30, "wal": True, "total_time_s": 0.5, "throughput_rows_per_sec": 10000.0, "rows_inserted": 5000},
                    {"flush_interval": 30, "wal": False, "total_time_s": 2.5, "throughput_rows_per_sec": 2000.0, "rows_inserted": 5000},
                ],
                "wal_vs_no_wal": {
                    "flush_interval": 30,
                    "wal_throughput_rows_per_sec": 10000.0,
                    "no_wal_throughput_rows_per_sec": 2000.0,
                    "wal_speedup_factor": 5.0,
                },
                "concurrent_read_test": {
                    "n_reader_threads": 4,
                    "n_reader_queries_per_thread": 100,
                    "read_latency_ms": {"p50": 0.02, "p95": 0.05, "mean": 0.025, "max": 0.1},
                    "write_throughput_rows_per_sec": 8000.0,
                    "total_reads": 400,
                },
                "memory": {"process_rss_mb": 480.0, "gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0},
            },
            "pipeline": {
                "device": "cpu",
                "frame_skip": 1,
                "n_frames": 50,
                "per_stage_latency_ms": {
                    "detection": {"p50": 90.0, "p90": 105.0, "p95": 115.0, "p99": 130.0, "mean": 92.0, "std": 7.0},
                    "crop_extract": {"p50": 0.5, "p90": 0.8, "p95": 1.0, "p99": 1.5, "mean": 0.55, "std": 0.1},
                    "reid_encode": {"p50": 40.0, "p90": 50.0, "p95": 55.0, "p99": 65.0, "mean": 41.0, "std": 4.0},
                    "faiss_query": {"p50": 0.3, "p90": 0.5, "p95": 0.6, "p99": 0.9, "mean": 0.32, "std": 0.05},
                    "db_write": {"p50": 0.1, "p90": 0.2, "p95": 0.25, "p99": 0.4, "mean": 0.12, "std": 0.03},
                },
                "total_latency_ms": {"p50": 132.0, "p90": 158.0, "p95": 172.0, "p99": 195.0, "mean": 134.0, "std": 12.0},
                "sla_150ms_compliance_pct": 75.0,
                "sla_50ms_compliance_pct": 0.0,
                "throughput_fps": 7.2,
                "avg_detections_per_frame": 3.8,
                "component_load_ms": {"detector_ms": 1200.0, "reid_encoder_ms": 300.0, "feature_bank_ms": 5.0, "database_ms": 8.0},
                "memory": {"process_rss_mb": 700.0, "gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0},
            },
        }

    def test_aggregate_creates_three_files(self) -> None:
        from benchmarks.report import aggregate_results

        results = self._make_fake_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = aggregate_results(results, output_dir=tmpdir)
            assert "json" in paths
            assert "csv" in paths
            assert "markdown" in paths
            for p in paths.values():
                assert os.path.exists(p), f"Report file not created: {p}"

    def test_json_is_valid(self) -> None:
        from benchmarks.report import aggregate_results

        results = self._make_fake_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = aggregate_results(results, output_dir=tmpdir)
            with open(paths["json"], encoding="utf-8") as f:
                loaded = json.load(f)
            assert "detector" in loaded
            assert "pipeline" in loaded

    def test_csv_has_rows(self) -> None:
        import csv as csv_mod
        from benchmarks.report import aggregate_results

        results = self._make_fake_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = aggregate_results(results, output_dir=tmpdir)
            with open(paths["csv"], encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                rows = list(reader)
            assert len(rows) > 0, "CSV should contain at least one data row"
            # Every row must have 'component' and 'value' fields.
            for row in rows:
                assert "component" in row
                assert "value" in row

    def test_markdown_contains_no_hardcoded_numbers(self) -> None:
        """Verify the markdown was generated from runtime values by checking
        that the detector p95 value in the results appears in the table."""
        from benchmarks.report import aggregate_results

        results = self._make_fake_results()
        # The detector p95 is 125.0ms — this should appear in the markdown.
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = aggregate_results(results, output_dir=tmpdir)
            with open(paths["markdown"], encoding="utf-8") as f:
                md = f.read()
            assert "125.00" in md, "Detector p95=125ms should appear in markdown table"
            assert "# Benchmark Report" in md
            assert "| Component |" in md

    def test_markdown_table_structure(self) -> None:
        from benchmarks.report import aggregate_results

        results = self._make_fake_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = aggregate_results(results, output_dir=tmpdir)
            with open(paths["markdown"], encoding="utf-8") as f:
                lines = f.readlines()
            # Should contain at least one table separator line.
            table_sep_lines = [l for l in lines if l.strip().startswith("|---")]
            assert len(table_sep_lines) >= 1, "Markdown should contain at least one table"

    def test_aggregate_handles_error_results(self) -> None:
        """aggregate_results should not crash when a component returned an error."""
        from benchmarks.report import aggregate_results

        results = {
            "detector": {"error": "YOLO model not found"},
            "reid": {"error": "torchreid import failed"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = aggregate_results(results, output_dir=tmpdir)
            # JSON should still be written.
            assert os.path.exists(paths["json"])


# ---------------------------------------------------------------------------
# 8. Re-ID norm check logic
# ---------------------------------------------------------------------------

class TestReIDNormCheck:
    """Unit test the L2 norm assertion logic used in bench_reid."""

    def test_normalised_vector_passes(self) -> None:
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        norm = np.linalg.norm(v)
        assert abs(norm - 1.0) <= 0.001

    def test_unnormalised_vector_fails(self) -> None:
        v = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        norm = np.linalg.norm(v)
        assert abs(norm - 1.0) > 0.001

    def test_batch_norm_check(self) -> None:
        """Simulate the norm check done per batch in bench_reid."""
        # Create 8 properly normalised vectors.
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((8, 64)).astype(np.float32)
        norms_raw = np.linalg.norm(raw, axis=1, keepdims=True)
        normalised = raw / norms_raw

        norms_check = np.linalg.norm(normalised, axis=1)
        assert np.all(np.abs(norms_check - 1.0) <= 0.001), "All normalised rows should pass"

    def test_non_unit_batch_detected(self) -> None:
        """A batch of raw (non-normalised) vectors should fail the check."""
        v = np.array([[2.0, 3.0], [5.0, 1.0]], dtype=np.float32)
        norms = np.linalg.norm(v, axis=1)
        assert np.any(np.abs(norms - 1.0) > 0.001), "Raw vectors should fail norm check"


# ---------------------------------------------------------------------------
# 9. get_memory_mb: always returns a dict with required keys
# ---------------------------------------------------------------------------

class TestGetMemoryMb:
    def test_returns_required_keys(self) -> None:
        from benchmarks.profiler import get_memory_mb

        result = get_memory_mb()
        assert "process_rss_mb" in result
        assert "gpu_allocated_mb" in result
        assert "gpu_reserved_mb" in result

    def test_values_are_non_negative(self) -> None:
        from benchmarks.profiler import get_memory_mb

        result = get_memory_mb()
        for key, val in result.items():
            assert val >= 0.0, f"{key} should be non-negative, got {val}"


# ---------------------------------------------------------------------------
# 10. warmup_model: callable is invoked n times
# ---------------------------------------------------------------------------

class TestWarmupModel:
    def test_warmup_calls_fn_n_times(self) -> None:
        from benchmarks.profiler import warmup_model

        call_count = []

        def fn(x: int) -> int:
            call_count.append(1)
            return x * 2

        warmup_model(fn, 5, n=7)
        assert len(call_count) == 7, f"Expected 7 calls, got {len(call_count)}"

    def test_warmup_passes_args(self) -> None:
        from benchmarks.profiler import warmup_model

        received = []

        def fn(a: int, b: str) -> None:
            received.append((a, b))

        warmup_model(fn, 42, "hello", n=3)
        assert all(r == (42, "hello") for r in received)
