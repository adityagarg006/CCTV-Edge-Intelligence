"""Tests for MetricsCollector — pure Python, no external dependencies."""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Test: summary() returns 0.0 for all fields on fresh instance
# ---------------------------------------------------------------------------

class TestFreshInstance:
    def test_all_summary_fields_are_zero_on_init(self):
        """A new MetricsCollector must report 0.0 for every summary field."""
        mc = MetricsCollector()
        summary = mc.summary()
        for key, value in summary.items():
            if key == "runtime_seconds":
                # runtime increases immediately after instantiation; allow a small window.
                assert value >= 0.0
            else:
                assert value == pytest.approx(0.0), f"Expected 0.0 for {key!r}, got {value}"

    def test_summary_returns_expected_keys(self):
        """summary() must return all documented keys."""
        mc = MetricsCollector()
        required_keys = {
            "avg_fps",
            "min_fps",
            "max_fps",
            "p95_latency_ms",
            "reid_hit_rate",
            "total_frames_processed",
            "runtime_seconds",
        }
        assert required_keys <= set(mc.summary().keys())


# ---------------------------------------------------------------------------
# Test: record_frame() updates avg_fps correctly
# ---------------------------------------------------------------------------

class TestRecordFrame:
    def test_avg_fps_reflects_recorded_samples(self):
        """avg_fps must equal the mean of all recorded fps samples."""
        mc = MetricsCollector()
        fps_values = [10.0, 20.0, 30.0, 40.0]
        for fps in fps_values:
            mc.record_frame(latency_ms=25.0, fps=fps, queue_depth=0)

        summary = mc.summary()
        assert summary["avg_fps"] == pytest.approx(float(np.mean(fps_values)), abs=1e-5)

    def test_min_max_fps_tracked(self):
        mc = MetricsCollector()
        for fps in [5.0, 15.0, 25.0, 35.0]:
            mc.record_frame(latency_ms=30.0, fps=fps, queue_depth=0)

        summary = mc.summary()
        assert summary["min_fps"] == pytest.approx(5.0, abs=1e-5)
        assert summary["max_fps"] == pytest.approx(35.0, abs=1e-5)

    def test_total_frames_increments(self):
        """total_frames_processed must equal the number of record_frame() calls."""
        mc = MetricsCollector()
        for _ in range(7):
            mc.record_frame(latency_ms=20.0, fps=30.0, queue_depth=5)
        assert mc.summary()["total_frames_processed"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Test: record_reid(hit=True) increments hit count
# ---------------------------------------------------------------------------

class TestRecordReid:
    def test_all_hits_produces_hit_rate_one(self):
        """5 hits and 0 misses must yield reid_hit_rate == 1.0."""
        mc = MetricsCollector()
        for _ in range(5):
            mc.record_reid(hit=True)
        assert mc.summary()["reid_hit_rate"] == pytest.approx(1.0)

    def test_all_misses_produces_hit_rate_zero(self):
        """0 hits and 5 misses must yield reid_hit_rate == 0.0."""
        mc = MetricsCollector()
        for _ in range(5):
            mc.record_reid(hit=False)
        assert mc.summary()["reid_hit_rate"] == pytest.approx(0.0)

    def test_mixed_hit_rate_is_accurate(self):
        """3 hits + 1 miss must yield reid_hit_rate == 0.75."""
        mc = MetricsCollector()
        mc.record_reid(hit=True)
        mc.record_reid(hit=True)
        mc.record_reid(hit=True)
        mc.record_reid(hit=False)
        assert mc.summary()["reid_hit_rate"] == pytest.approx(0.75, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: P95 latency with a known distribution
# ---------------------------------------------------------------------------

class TestP95Latency:
    def test_p95_latency_matches_numpy_percentile(self):
        """p95_latency_ms must match numpy's 95th percentile of the recorded values."""
        mc = MetricsCollector()
        latencies = [float(x) for x in range(1, 101)]  # 1, 2, ..., 100
        for lat in latencies:
            mc.record_frame(latency_ms=lat, fps=30.0, queue_depth=0)

        expected_p95 = float(np.percentile(latencies, 95))
        assert mc.summary()["p95_latency_ms"] == pytest.approx(expected_p95, abs=0.1)

    def test_p95_latency_is_zero_when_no_frames_recorded(self):
        """p95_latency_ms must be 0.0 when no frames have been recorded yet."""
        mc = MetricsCollector()
        assert mc.summary()["p95_latency_ms"] == pytest.approx(0.0)

    def test_p95_single_frame(self):
        """With a single frame, p95_latency_ms must equal that frame's latency."""
        mc = MetricsCollector()
        mc.record_frame(latency_ms=42.7, fps=30.0, queue_depth=0)
        assert mc.summary()["p95_latency_ms"] == pytest.approx(42.7, abs=0.01)
