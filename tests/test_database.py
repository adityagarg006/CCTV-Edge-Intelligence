"""Tests for TrackingDatabase — uses pytest tmp_path, no real camera."""
from __future__ import annotations

import sqlite3
import time

import numpy as np
import pytest

from src.database import TrackingDatabase
from src.detector import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(track_id: int = 1, conf: float = 0.85) -> Detection:
    return Detection(
        track_id=track_id,
        bbox_xyxy=np.array([10.0, 20.0, 100.0, 200.0], dtype=np.float32),
        confidence=conf,
        class_id=0,
    )


def _open_raw(db_path: str) -> sqlite3.Connection:
    """Open the SQLite file directly to verify content without TrackingDatabase."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Test: log_detection() + flush() → row appears in DB
# ---------------------------------------------------------------------------

class TestLogDetectionAndFlush:
    def test_row_appears_after_flush(self, tmp_path):
        """A logged detection must appear in the detections table after flush()."""
        db_path = str(tmp_path / "test.db")
        db = TrackingDatabase(db_path=db_path, flush_interval=100)

        ts = time.time()
        det = _make_detection(track_id=7)
        db.log_detection(frame_id=1, timestamp=ts, detection=det, reid_id=42)
        db.flush()
        db.close()

        raw = _open_raw(db_path)
        row = raw.execute("SELECT * FROM detections WHERE track_id = 7").fetchone()
        assert row is not None
        assert row["reid_id"] == 42
        assert row["frame_id"] == 1
        assert pytest.approx(row["confidence"], abs=1e-3) == 0.85
        raw.close()

    def test_multiple_rows_after_flush(self, tmp_path):
        """All buffered detections must be persisted after a single flush()."""
        db_path = str(tmp_path / "test2.db")
        db = TrackingDatabase(db_path=db_path, flush_interval=100)
        ts = time.time()

        for i in range(5):
            db.log_detection(frame_id=i, timestamp=ts + i, detection=_make_detection(i), reid_id=i)
        db.flush()
        db.close()

        raw = _open_raw(db_path)
        count = raw.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        assert count == 5
        raw.close()

    def test_no_rows_before_flush(self, tmp_path):
        """Rows must not be written to the DB until flush() is called."""
        db_path = str(tmp_path / "test3.db")
        db = TrackingDatabase(db_path=db_path, flush_interval=100)
        ts = time.time()
        db.log_detection(frame_id=1, timestamp=ts, detection=_make_detection(), reid_id=1)

        # Do NOT call flush — query directly (buffer not yet committed).
        raw = _open_raw(db_path)
        count = raw.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        raw.close()

        assert count == 0

        db.close()


# ---------------------------------------------------------------------------
# Test: upsert_identity() updates last_seen and increments total_detections
# ---------------------------------------------------------------------------

class TestUpsertIdentity:
    def test_first_upsert_inserts_row(self, tmp_path):
        """First upsert_identity call must create an identities row."""
        db_path = str(tmp_path / "identity.db")
        db = TrackingDatabase(db_path=db_path, flush_interval=100)
        ts = time.time()
        db.upsert_identity(reid_id=10, timestamp=ts, confidence=0.9)
        db._conn.commit()

        row = db._conn.execute(
            "SELECT * FROM identities WHERE reid_id = 10"
        ).fetchone()
        assert row is not None
        db.close()

    def test_second_upsert_increments_total_detections(self, tmp_path):
        """Second call to upsert_identity must increment total_detections."""
        db_path = str(tmp_path / "identity2.db")
        db = TrackingDatabase(db_path=db_path, flush_interval=100)
        ts = time.time()

        db.upsert_identity(reid_id=5, timestamp=ts, confidence=0.7)
        db.upsert_identity(reid_id=5, timestamp=ts + 1.0, confidence=0.8)
        db._conn.commit()

        row = db._conn.execute(
            "SELECT total_detections, best_confidence FROM identities WHERE reid_id = 5"
        ).fetchone()
        # row is a plain tuple — (total_detections, best_confidence)
        assert row[0] == 2
        assert pytest.approx(row[1], abs=1e-3) == 0.8
        db.close()

    def test_second_upsert_updates_last_seen(self, tmp_path):
        """last_seen must be updated to the newer timestamp on second upsert."""
        db_path = str(tmp_path / "identity3.db")
        db = TrackingDatabase(db_path=db_path, flush_interval=100)
        ts = time.time()

        db.upsert_identity(reid_id=3, timestamp=ts, confidence=0.75)
        later_ts = ts + 30.0
        db.upsert_identity(reid_id=3, timestamp=later_ts, confidence=0.6)
        db._conn.commit()

        row = db._conn.execute(
            "SELECT first_seen, last_seen FROM identities WHERE reid_id = 3"
        ).fetchone()
        # row is a plain tuple — (first_seen, last_seen)
        assert pytest.approx(row[0], abs=0.01) == ts
        assert pytest.approx(row[1], abs=0.01) == later_ts
        db.close()


# ---------------------------------------------------------------------------
# Test: context manager closes connection on exit
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_connection_closed_after_exit(self, tmp_path):
        """TrackingDatabase.__exit__ must close the underlying SQLite connection."""
        db_path = str(tmp_path / "ctx.db")

        with TrackingDatabase(db_path=db_path, flush_interval=100) as db:
            ts = time.time()
            db.log_detection(frame_id=1, timestamp=ts, detection=_make_detection(), reid_id=1)

        # Accessing the closed connection must raise.
        with pytest.raises(Exception):
            db._conn.execute("SELECT 1")

    def test_rows_flushed_on_exit(self, tmp_path):
        """Rows buffered inside the context must be persisted after __exit__."""
        db_path = str(tmp_path / "ctx2.db")
        ts = time.time()

        with TrackingDatabase(db_path=db_path, flush_interval=100) as db:
            db.log_detection(frame_id=1, timestamp=ts, detection=_make_detection(), reid_id=99)

        raw = _open_raw(db_path)
        count = raw.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        raw.close()
        assert count == 1
