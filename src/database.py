"""SQLite WAL persistence layer for tracking detections and identities.

Design decisions:
-----------------
``PRAGMA journal_mode=WAL``: Write-Ahead Logging allows concurrent reads while
a write is in progress. Without WAL, any reader would block on an active write
transaction, causing dropped frames in a real-time pipeline.

``PRAGMA synchronous=NORMAL``: Skips fsync after every write (but not after a
checkpoint). For surveillance data, losing the last few buffered frames on an
OS crash is acceptable. This pragma yields 3–5× write throughput improvement
over ``synchronous=FULL``.

Write buffer (deque + flush interval): Accumulating row insertions and
committing them in a single transaction every N frames amortises
transaction-open/commit overhead. Individual per-frame commits would dominate
CPU time at high frame rates.

Indexes on ``track_id`` and ``frame_id``: Enable O(log N) lookup during
analytics aggregation (GROUP BY, WHERE track_id = ?) vs O(N) full-table scan.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from collections import deque
from typing import Any

from src.detector import Detection
from config import settings

logger = logging.getLogger(__name__)

# SQL schema — created once at startup.
_CREATE_DETECTIONS = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,
    frame_id    INTEGER NOT NULL,
    track_id    INTEGER NOT NULL,
    reid_id     INTEGER,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    confidence  REAL
);
"""

_CREATE_IDENTITIES = """
CREATE TABLE IF NOT EXISTS identities (
    reid_id          INTEGER PRIMARY KEY,
    first_seen       REAL NOT NULL,
    last_seen        REAL NOT NULL,
    total_detections INTEGER DEFAULT 1,
    best_confidence  REAL
);
"""

_CREATE_IDX_TRACK = (
    "CREATE INDEX IF NOT EXISTS idx_detections_track_id ON detections(track_id);"
)
_CREATE_IDX_FRAME = (
    "CREATE INDEX IF NOT EXISTS idx_detections_frame_id ON detections(frame_id);"
)


class TrackingDatabase:
    """Buffered SQLite database for storing detections and identity profiles.

    Args:
        db_path: File-system path for the SQLite database. Created if it does
            not exist.
        flush_interval: Number of ``log_detection`` calls between automatic
            buffer flushes (i.e., transaction commits). Defaults to
            ``settings.DB_FLUSH_INTERVAL``.

    Raises:
        OSError: If the database file cannot be opened or created.
    """

    def __init__(
        self,
        db_path: str = settings.DB_PATH,
        flush_interval: int = settings.DB_FLUSH_INTERVAL,
    ) -> None:
        self._db_path = db_path
        self._flush_interval = flush_interval
        self._detection_count = 0

        # Write buffer: each element is (args_tuple) for the INSERT statement.
        self._buffer: deque[tuple[Any, ...]] = deque()

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(
            _CREATE_DETECTIONS
            + _CREATE_IDENTITIES
            + _CREATE_IDX_TRACK
            + _CREATE_IDX_FRAME
        )
        self._conn.commit()
        logger.info("TrackingDatabase opened at %r (flush_interval=%d)", db_path, flush_interval)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log_detection(
        self,
        frame_id: int,
        timestamp: float,
        detection: Detection,
        reid_id: int | None,
    ) -> None:
        """Buffer one detection row for insertion.

        Triggers a flush when the buffer reaches ``flush_interval`` entries.

        Args:
            frame_id: Monotonically increasing frame counter.
            timestamp: UNIX timestamp (``time.time()``) at frame capture.
            detection: Detection dataclass instance.
            reid_id: Resolved Re-ID identity (``None`` if Re-ID was skipped).
        """
        x1, y1, x2, y2 = detection.bbox_xyxy
        self._buffer.append((
            timestamp,
            frame_id,
            detection.track_id,
            reid_id,
            float(x1), float(y1), float(x2), float(y2),
            detection.confidence,
        ))
        self._detection_count += 1
        if self._detection_count % self._flush_interval == 0:
            self.flush()

    def upsert_identity(
        self,
        reid_id: int,
        timestamp: float,
        confidence: float,
    ) -> None:
        """Insert or update a row in the ``identities`` table.

        On first insert, sets both ``first_seen`` and ``last_seen``. On
        subsequent calls, updates ``last_seen``, increments
        ``total_detections``, and updates ``best_confidence`` if the new
        value is higher.

        Args:
            reid_id: Stable Re-ID integer identity.
            timestamp: UNIX timestamp of this detection.
            confidence: YOLO detection confidence score.
        """
        self._conn.execute(
            """
            INSERT INTO identities (reid_id, first_seen, last_seen, total_detections, best_confidence)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(reid_id) DO UPDATE SET
                last_seen        = MAX(last_seen, excluded.last_seen),
                total_detections = total_detections + 1,
                best_confidence  = MAX(best_confidence, excluded.best_confidence)
            """,
            (reid_id, timestamp, timestamp, confidence),
        )

    def flush(self) -> None:
        """Commit all buffered detection rows to the database.

        Called automatically every ``flush_interval`` detections and on
        ``close()`` / ``__exit__``.
        """
        if not self._buffer:
            return

        rows = list(self._buffer)
        self._buffer.clear()

        try:
            self._conn.executemany(
                """
                INSERT INTO detections
                    (timestamp, frame_id, track_id, reid_id, x1, y1, x2, y2, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
            logger.debug("Flushed %d detection rows to DB.", len(rows))
        except sqlite3.Error as exc:
            logger.error("DB flush failed: %s", exc)
            # Re-queue rows so they are not silently lost.
            self._buffer.extendleft(reversed(rows))

    def close(self) -> None:
        """Flush remaining buffer and close the database connection.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._conn:
            self.flush()
            self._conn.close()
            logger.info("TrackingDatabase closed.")

    def __enter__(self) -> "TrackingDatabase":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
