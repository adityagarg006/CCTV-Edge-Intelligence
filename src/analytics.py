"""Analytics reporter: generates JSON summary and CSV exports from SQLite data."""
from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime

from src.database import TrackingDatabase
from src.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class AnalyticsReporter:
    """Generate post-session surveillance reports from the tracking database.

    Produces three files per session:
    - ``surveillance_report_<ts>.json``: System metadata and throughput summary.
    - ``tracking_summary_<ts>.csv``:  Per-identity statistics.
    - ``frame_density_<ts>.csv``: Per-frame detection counts.

    All throughput values are sourced from ``MetricsCollector.summary()`` — none
    are hard-coded.

    Args:
        db: Open ``TrackingDatabase`` instance for SQL queries.
        metrics: ``MetricsCollector`` accumulating per-frame statistics.
        source: Video source string (RTSP URL, file path, or webcam index).
        model_name: YOLO model name (e.g., ``"yolov8n.pt"``).
        index_type: FAISS index description (e.g., ``"IndexFlatL2+IndexIDMap"``).
    """

    def __init__(
        self,
        db: TrackingDatabase,
        metrics: MetricsCollector,
        source: str,
        model_name: str,
        index_type: str,
    ) -> None:
        self._db = db
        self._metrics = metrics
        self._source = source
        self._model_name = model_name
        self._index_type = index_type

    def generate_report(self, output_dir: str) -> dict[str, str]:
        """Write all report files and return a mapping of type → file path.

        Flushes the database buffer before querying so no in-flight detections
        are missed.

        Args:
            output_dir: Directory for report output. Created if absent.

        Returns:
            Dictionary with keys ``"json"``, ``"tracking_csv"``, and
            ``"density_csv"`` mapping to the absolute file paths written.

        Raises:
            OSError: If ``output_dir`` cannot be created or a file cannot be
                written.
        """
        os.makedirs(output_dir, exist_ok=True)
        self._db.flush()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths: dict[str, str] = {
            "json": os.path.join(output_dir, f"surveillance_report_{ts}.json"),
            "tracking_csv": os.path.join(output_dir, f"tracking_summary_{ts}.csv"),
            "density_csv": os.path.join(output_dir, f"frame_density_{ts}.csv"),
        }

        self._write_json(paths["json"])
        self._write_tracking_csv(paths["tracking_csv"])
        self._write_density_csv(paths["density_csv"])

        logger.info("Reports written to %r: %s", output_dir, list(paths.values()))
        return paths

    # ------------------------------------------------------------------
    # Private writers
    # ------------------------------------------------------------------

    def _write_json(self, path: str) -> None:
        summary = self._metrics.summary()
        conn = self._db._conn

        unique_track_ids = conn.execute(
            "SELECT COUNT(DISTINCT track_id) FROM detections"
        ).fetchone()[0] or 0

        unique_reid_ids = conn.execute(
            "SELECT COUNT(DISTINCT reid_id) FROM detections WHERE reid_id IS NOT NULL"
        ).fetchone()[0] or 0

        total_detections = conn.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0] or 0

        total_frames = int(summary["total_frames_processed"]) or 1
        avg_det_per_frame = total_detections / total_frames

        identity_rows = conn.execute(
            """
            SELECT reid_id, first_seen, last_seen, total_detections, best_confidence
            FROM identities
            ORDER BY first_seen
            """
        ).fetchall()

        identity_profiles = [
            {
                "reid_id": row[0],
                "first_seen": datetime.fromtimestamp(row[1]).isoformat(),
                "last_seen": datetime.fromtimestamp(row[2]).isoformat(),
                "total_detections": row[3],
                "best_confidence": row[4],
            }
            for row in identity_rows
        ]

        report = {
            "system_metadata": {
                "model_name": self._model_name,
                "reid_model": "osnet_x0_25",
                "index_type": self._index_type,
                "source": self._source,
                "generated_at": datetime.now().isoformat(),
                "runtime_seconds": summary["runtime_seconds"],
            },
            "throughput": {
                "total_frames": total_frames,
                "avg_fps": round(summary["avg_fps"], 2),
                "min_fps": round(summary["min_fps"], 2),
                "max_fps": round(summary["max_fps"], 2),
                "p95_frame_latency_ms": round(summary["p95_latency_ms"], 2),
            },
            "tracking_summary": {
                "unique_track_ids": unique_track_ids,
                "unique_reid_ids": unique_reid_ids,
                "total_detections": total_detections,
                "avg_detections_per_frame": round(avg_det_per_frame, 2),
            },
            "identity_profiles": identity_profiles,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def _write_tracking_csv(self, path: str) -> None:
        conn = self._db._conn
        rows = conn.execute(
            """
            SELECT reid_id, first_seen, last_seen, total_detections, best_confidence
            FROM identities
            ORDER BY first_seen
            """
        ).fetchall()

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "reid_id",
                "first_seen_iso",
                "first_seen_ts",
                "last_seen_iso",
                "last_seen_ts",
                "duration_seconds",
                "total_detections",
                "best_confidence",
            ])
            for row in rows:
                reid_id, first_ts, last_ts, total_det, best_conf = row
                writer.writerow([
                    reid_id,
                    datetime.fromtimestamp(first_ts).isoformat(),
                    first_ts,
                    datetime.fromtimestamp(last_ts).isoformat(),
                    last_ts,
                    round(last_ts - first_ts, 3),
                    total_det,
                    best_conf,
                ])

    def _write_density_csv(self, path: str) -> None:
        conn = self._db._conn
        rows = conn.execute(
            """
            SELECT frame_id, timestamp, COUNT(*) AS detection_count
            FROM detections
            GROUP BY frame_id
            ORDER BY frame_id
            """
        ).fetchall()

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "timestamp_iso", "timestamp_ts", "detection_count"])
            for row in rows:
                frame_id, ts, count = row
                writer.writerow([
                    frame_id,
                    datetime.fromtimestamp(ts).isoformat(),
                    ts,
                    count,
                ])
