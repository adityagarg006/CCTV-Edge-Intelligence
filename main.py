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
import numpy as np

from config import settings
from src.analytics import AnalyticsReporter
from src.database import TrackingDatabase
from src.detector import Detection, PersonDetector
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
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated output to an MP4 file inside --output-dir.",
    )
    parser.add_argument(
        "--inference-width",
        type=int,
        default=1280,
        help=(
            "Resize frames to this width before YOLO+ReID inference. "
            "Dramatically reduces inference time on high-res sources (e.g. 4K). "
            "Bounding boxes are scaled back to original coordinates automatically. "
            "Use 0 to disable. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=1280,
        help=(
            "Resize annotated frames to this width before saving. "
            "Height is scaled proportionally. Use 0 to keep original resolution. "
            "(default: %(default)s)"
        ),
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
# Helpers
# ---------------------------------------------------------------------------

def _resize_for_output(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Resize frame to target_width, preserving aspect ratio.

    Args:
        frame: BGR frame.
        target_width: Desired output width in pixels. 0 = no resize.

    Returns:
        Resized frame, or the original if target_width is 0 or already smaller.
    """
    if target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def _probe_fps(source: int | str, default: float = 30.0) -> float:
    """Return the native FPS of a video source for VideoWriter initialisation.

    Args:
        source: Webcam index, file path, or RTSP URL.
        default: Fallback FPS when the source reports 0 or is unavailable.

    Returns:
        Frames-per-second as a float.
    """
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default


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

    # Probe source FPS for the VideoWriter (needed before frames arrive).
    source_fps = _probe_fps(source)

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

            # VideoWriter — lazily initialised on the first annotated frame
            # so we know the exact frame dimensions.
            video_writer: cv2.VideoWriter | None = None
            video_out_path: str | None = None
            if args.save_video:
                import os as _os
                _os.makedirs(args.output_dir, exist_ok=True)
                from datetime import datetime as _dt
                _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                video_out_path = _os.path.join(args.output_dir, f"annotated_{_ts}.mp4")

            while not _shutdown_event.is_set():
                ok, frame = reader.read()

                if not ok:
                    if _shutdown_event.is_set():
                        break
                    if isinstance(source, str) and not source.lower().startswith("rtsp://"):
                        # File source: only exit when the producer has finished AND
                        # the queue is drained. A False read while the producer is
                        # still running just means the queue is momentarily empty
                        # (e.g. the VideoCapture open took longer than the 50ms
                        # read() timeout on the first call).
                        if reader.is_done():
                            logger.info("Video file ended. Exiting main loop.")
                            break
                        continue
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

                # 1. Optionally resize for inference.
                # High-res sources (e.g. 4K) cause the queue to drop frames because
                # inference is slower than the source frame rate. Resizing to ~1280px
                # wide brings inference time under the inter-frame interval so the
                # consumer can keep up with the producer.
                orig_h, orig_w = frame.shape[:2]
                if args.inference_width > 0 and orig_w > args.inference_width:
                    scale_x = orig_w / args.inference_width
                    scale_y = orig_h / int(orig_h * args.inference_width / orig_w)
                    inf_frame = cv2.resize(
                        frame,
                        (args.inference_width, int(orig_h * args.inference_width / orig_w)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    inf_frame = frame
                    scale_x = 1.0
                    scale_y = 1.0

                # 2. Detection + tracking (on inference-resolution frame).
                detections = detector.detect(inf_frame)

                # Scale bboxes back to original-frame coordinates so DB logs and
                # annotations are in the correct pixel space.
                if scale_x != 1.0:
                    detections = [
                        Detection(
                            track_id=d.track_id,
                            bbox_xyxy=np.array(
                                [
                                    d.bbox_xyxy[0] * scale_x,
                                    d.bbox_xyxy[1] * scale_y,
                                    d.bbox_xyxy[2] * scale_x,
                                    d.bbox_xyxy[3] * scale_y,
                                ],
                                dtype=np.float32,
                            ),
                            confidence=d.confidence,
                            class_id=d.class_id,
                        )
                        for d in detections
                    ]

                # 3. Crop extraction and Re-ID encoding (on original-res frame).
                valid_detections, crops = extract_crops(frame, detections)
                embeddings = reid_encoder.encode_batch(crops)

                # 4. Identity assignment and gallery management.
                #
                # Gallery keys are always reid_ids — never track_ids.
                #
                # Two-path logic:
                #   Continuing track  → reid_id already known, refresh gallery entry.
                #   New/returning track → query gallery with active-exclusion filter.
                #
                # Active-exclusion: when a new track_id appears, we exclude from the
                # query any reid_id that is already claimed by a *currently visible*
                # person in this frame. Without this, T5 (new person) could match
                # T1's gallery entry even though T1 is standing right next to them —
                # two simultaneously visible people would share the same reid_id.
                # A returning person's reid_id is safe to match because their track
                # was lost (they are NOT in the current frame's active set).
                active_reid_ids: set[int] = {
                    reid_assignments[tid]
                    for tid in (d.track_id for d in valid_detections)
                    if tid in reid_assignments
                }

                for det, embedding in zip(valid_detections, embeddings):
                    if det.track_id in reid_assignments:
                        reid_id = reid_assignments[det.track_id]
                        if frame_id % settings.REID_GALLERY_UPDATE_FREQ == 0:
                            feature_bank.update(reid_id, embedding)
                        metrics.record_reid(hit=True)
                    else:
                        # Query gallery, then filter out reid_ids that are already
                        # claimed by someone currently visible in this frame.
                        candidates = feature_bank.query(embedding, k=5)
                        candidates = [
                            (rid, dist) for rid, dist in candidates
                            if rid not in active_reid_ids
                        ]

                        if candidates and candidates[0][1] <= settings.REID_DISTANCE_THRESHOLD:
                            reid_id = candidates[0][0]
                            metrics.record_reid(hit=True)
                        else:
                            reid_id = next_reid_id
                            next_reid_id += 1
                            metrics.record_reid(hit=False)

                        feature_bank.update(reid_id, embedding)
                        reid_assignments[det.track_id] = reid_id
                        # Add this new assignment to the active set so later
                        # iterations in this same frame respect it.
                        active_reid_ids.add(reid_id)

                    db.log_detection(frame_id, timestamp, det, reid_id)
                    db.upsert_identity(reid_id, timestamp, det.confidence)

                # 4. Metrics.
                latency_ms = (time.perf_counter() - t_start) * 1000
                now = time.monotonic()
                ts_window.append(now)
                fps = len(ts_window) / (ts_window[-1] - ts_window[0] + 1e-9) if len(ts_window) > 1 else 0.0
                metrics.record_frame(latency_ms, fps, reader.qsize())

                # 5. Annotate, display, and/or save video.
                if display or args.save_video:
                    annotated = annotate_frame(
                        frame.copy(),
                        detections,
                        reid_assignments,
                        fps,
                        reader.qsize(),
                        settings.FRAME_QUEUE_MAXSIZE,
                        gallery_size=feature_bank.size(),
                    )

                    if args.save_video and video_out_path is not None:
                        out_frame = _resize_for_output(annotated, args.output_width)
                        # Initialise writer on the first frame once we know exact dimensions.
                        if video_writer is None:
                            oh, ow = out_frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(
                                video_out_path, fourcc, source_fps, (ow, oh)
                            )
                            logger.info(
                                "VideoWriter opened: %s  (%dx%d @ %.1f fps)",
                                video_out_path, ow, oh, source_fps,
                            )
                        video_writer.write(out_frame)

                    if display:
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

        if video_writer is not None:
            video_writer.release()
            logger.info("Annotated video saved to %s", video_out_path)

        # Final report (if not already triggered by signal handler).
        if not _shutdown_triggered:
            try:
                paths = reporter.generate_report(args.output_dir)
                print("\n--- Session Complete ---")
                for key, value in metrics.summary().items():
                    print(f"  {key}: {value}")
                print(f"\nReports: {list(paths.values())}")
                if video_out_path:
                    print(f"  Annotated video: {video_out_path}")
            except Exception as exc:
                logger.error("Failed to generate final report: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
