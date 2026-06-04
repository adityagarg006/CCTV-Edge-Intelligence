"""CCTV-Edge-Intelligence — main orchestration entry point.

Wires together all pipeline stages:
    AsyncVideoReader → PersonDetector → ReIDEncoder → FeatureBank
        → TrackingDatabase + MetricsCollector → AnalyticsReporter

Signal handling (SIGINT/SIGTERM) is idempotent: a double-signal does not
trigger a second shutdown sequence.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections import deque
from threading import Event

import cv2

from config import settings
from src.analytics import AnalyticsReporter
from src.database import TrackingDatabase
from src.detector import PersonDetector
from src.feature_bank import FeatureBank
from src.metrics import MetricsCollector
from src.reid_encoder import ReIDEncoder
from src.stream_reader import AsyncVideoReader
from src.utils import annotate_frame, auto_select_device, extract_crops, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CCTV-Edge-Intelligence: real-time person detection and Re-ID pipeline."
    )
    parser.add_argument(
        "--source",
        default=str(settings.VIDEO_SOURCE),
        help="Video source: webcam index (int), file path, or RTSP URL. (default: %(default)s)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the cv2 display window (headless / server mode).",
    )
    parser.add_argument(
        "--output-dir",
        default=settings.OUTPUT_DIR,
        help="Directory for generated reports. (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Compute device for YOLO and Re-ID inference. (default: %(default)s)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=settings.FRAME_SKIP,
        help="Run inference every N frames (CPU degradation mode). (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default=settings.LOG_LEVEL,
        help="Logging verbosity. (default: %(default)s)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shutdown handler
# ---------------------------------------------------------------------------

_shutdown_event = Event()
_shutdown_triggered = False


def _build_signal_handler(
    reader: AsyncVideoReader,
    db: TrackingDatabase,
    reporter: AnalyticsReporter,
    output_dir: str,
    metrics: MetricsCollector,
):
    def handler(signum: int, frame: object) -> None:
        global _shutdown_triggered
        if _shutdown_triggered:
            return  # Idempotent — ignore double-signal.
        _shutdown_triggered = True
        logger.info("Shutdown signal %d received. Flushing and generating report …", signum)
        _shutdown_event.set()
        reader.stop()
        db.flush()
        try:
            paths = reporter.generate_report(output_dir)
            print("\n--- Final Stats ---")
            for key, value in metrics.summary().items():
                print(f"  {key}: {value}")
            print(f"\nReports written to {output_dir}:")
            for kind, path in paths.items():
                print(f"  [{kind}] {path}")
        except Exception as exc:
            logger.error("Error generating final report: %s", exc)

    return handler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    # Resolve source type: coerce to int if numeric string.
    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    setup_logging(log_level=args.log_level, log_file=settings.LOG_FILE)
    logger.info("Starting CCTV-Edge-Intelligence | source=%r device=%s", source, args.device)

    device = auto_select_device() if args.device == "auto" else args.device
    display = not args.no_display and settings.DISPLAY_OUTPUT

    # Initialise all components.
    metrics = MetricsCollector()
    detector = PersonDetector(device=device, frame_skip=args.frame_skip)
    reid_encoder = ReIDEncoder(device=device)
    feature_bank = FeatureBank(embed_dim=reid_encoder.embed_dim)

    with TrackingDatabase(db_path=settings.DB_PATH) as db:
        reporter = AnalyticsReporter(
            db=db,
            metrics=metrics,
            source=str(source),
            model_name=settings.YOLO_MODEL,
            index_type="IndexFlatL2+IndexIDMap",
        )

        handler = _build_signal_handler(
            reader=None,  # type: ignore[arg-type]  # assigned below
            db=db,
            reporter=reporter,
            output_dir=args.output_dir,
            metrics=metrics,
        )

        with AsyncVideoReader(source=source, maxsize=settings.FRAME_QUEUE_MAXSIZE) as reader:
            # Patch handler with the real reader now that it's constructed.
            handler = _build_signal_handler(reader, db, reporter, args.output_dir, metrics)
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)

            frame_id = 0
            next_reid_id = 1
            reid_assignments: dict[int, int] = {}  # track_id → reid_id

            # Rolling FPS window over the last 30 frame timestamps.
            ts_window: deque[float] = deque(maxlen=30)

            while not _shutdown_event.is_set():
                ok, frame = reader.read()

                if not ok:
                    if _shutdown_event.is_set():
                        break
                    # Source exhausted or frame queue empty — distinguish.
                    if isinstance(source, str) and not source.lower().startswith("rtsp://"):
                        logger.info("Video file ended. Exiting main loop.")
                        break
                    elif isinstance(source, str) and source.lower().startswith("rtsp://"):
                        logger.warning("RTSP frame unavailable (may be reconnecting). Waiting …")
                        time.sleep(0.05)
                        continue
                    else:
                        # Empty queue on webcam — brief back-off.
                        time.sleep(0.01)
                        continue

                t_start = time.perf_counter()
                timestamp = time.time()
                frame_id += 1

                # 1. Detection + tracking.
                detections = detector.detect(frame)

                # 2. Crop extraction and Re-ID encoding.
                valid_detections, crops = extract_crops(frame, detections)
                embeddings = reid_encoder.encode_batch(crops)

                # 3. Gallery update, identity assignment, DB logging.
                for det, embedding in zip(valid_detections, embeddings):
                    if frame_id % settings.REID_GALLERY_UPDATE_FREQ == 0:
                        feature_bank.update(det.track_id, embedding)

                    results = feature_bank.query(embedding, k=1)

                    if results and results[0][1] <= settings.REID_DISTANCE_THRESHOLD:
                        reid_id = results[0][0]
                        metrics.record_reid(hit=True)
                    else:
                        reid_id = next_reid_id
                        next_reid_id += 1
                        feature_bank.update(reid_id, embedding)
                        metrics.record_reid(hit=False)

                    reid_assignments[det.track_id] = reid_id
                    db.log_detection(frame_id, timestamp, det, reid_id)
                    db.upsert_identity(reid_id, timestamp, det.confidence)

                # 4. Metrics.
                latency_ms = (time.perf_counter() - t_start) * 1000
                now = time.monotonic()
                ts_window.append(now)
                fps = len(ts_window) / (ts_window[-1] - ts_window[0] + 1e-9) if len(ts_window) > 1 else 0.0
                metrics.record_frame(latency_ms, fps, reader.qsize())

                # 5. Display.
                if display:
                    annotated = annotate_frame(
                        frame.copy(),
                        detections,
                        reid_assignments,
                        fps,
                        reader.qsize(),
                        settings.FRAME_QUEUE_MAXSIZE,
                        gallery_size=feature_bank.size(),
                    )
                    cv2.imshow("CCTV-Edge-Intelligence", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("User pressed 'q'. Shutting down.")
                        _shutdown_event.set()
                        break

                # 6. Console heartbeat every 30 frames.
                if frame_id % 30 == 0:
                    print(
                        f"Frame {frame_id:>6} | FPS: {fps:5.1f} "
                        f"| Tracked: {len(detections)} "
                        f"| Gallery: {feature_bank.size()} "
                        f"| Queue: {reader.qsize()}/{settings.FRAME_QUEUE_MAXSIZE}"
                    )

        if display:
            cv2.destroyAllWindows()

        # Final report (if not already triggered by signal handler).
        if not _shutdown_triggered:
            try:
                paths = reporter.generate_report(args.output_dir)
                print("\n--- Session Complete ---")
                for key, value in metrics.summary().items():
                    print(f"  {key}: {value}")
                print(f"\nReports: {list(paths.values())}")
            except Exception as exc:
                logger.error("Failed to generate final report: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
