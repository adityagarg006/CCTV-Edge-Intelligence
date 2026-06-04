"""YOLOv8 + ByteTrack person detector for CCTV-Edge-Intelligence.

Why ByteTrack over DeepSORT / StrongSORT?
------------------------------------------
1. Two-stage association: ByteTrack matches high-confidence detections first,
   then re-associates low-confidence boxes with unmatched tracks. This
   preserves identity across partial occlusions (e.g., a person half-hidden
   by a column keeps their track_id rather than spawning a new one).

2. No appearance model: DeepSORT runs a re-identification CNN on every frame
   for its cascade-matching step, adding O(N²) appearance-feature computation.
   ByteTrack uses only IoU for association, keeping the detector self-contained
   and inference time predictable.

3. MOTA on MOT17: ByteTrack ≈77.8 vs DeepSORT ≈74.5 at comparable cost.
"""
from __future__ import annotations

import dataclasses
import logging
import threading

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Detection:
    """Immutable result of one tracked person detection.

    Attributes:
        track_id: ByteTrack-assigned persistent integer identity for this
            detection across frames.
        bbox_xyxy: Bounding box as ``[x1, y1, x2, y2]`` in pixel coordinates,
            dtype float32, shape (4,).
        confidence: YOLO detection confidence in [0, 1].
        class_id: COCO class id (0 = person).
    """

    track_id: int
    bbox_xyxy: np.ndarray  # shape (4,), dtype float32
    confidence: float
    class_id: int


class PersonDetector:
    """Thread-safe YOLOv8 person detector with ByteTrack tracking.

    Args:
        device: Compute device — ``'cuda'``, ``'cpu'``, or ``'auto'``.
            ``'auto'`` calls ``torch.cuda.is_available()`` to decide.
        frame_skip: Run inference only every ``frame_skip``-th call. On
            intermediate calls, return the previous result. Use ``frame_skip=3``
            on CPU-only hardware for ~3× throughput at the cost of tracking
            freshness. This is the CPU graceful-degradation mechanism — the
            architecture does not change, only the inference cadence.

    Raises:
        RuntimeError: If the YOLO model file cannot be loaded.
    """

    def __init__(
        self,
        device: str = "auto",
        frame_skip: int = settings.FRAME_SKIP,
    ) -> None:
        from src.utils import auto_select_device

        self._device = auto_select_device() if device == "auto" else device
        self._frame_skip = max(1, frame_skip)
        self._lock = threading.Lock()
        self._call_count = 0
        self._last_result: list[Detection] = []

        try:
            from ultralytics import YOLO

            self._model = YOLO(settings.YOLO_MODEL)
            logger.info(
                "PersonDetector loaded model=%r on device=%r frame_skip=%d",
                settings.YOLO_MODEL,
                self._device,
                self._frame_skip,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model {settings.YOLO_MODEL!r}: {exc}"
            ) from exc

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection and tracking on one frame.

        Args:
            frame: BGR frame as an (H, W, 3) numpy array.

        Returns:
            List of ``Detection`` objects for all tracked persons in the frame.
            Returns an empty list when no persons are detected. On frame-skip
            frames, returns the cached result from the previous inference call.
        """
        with self._lock:
            self._call_count += 1

            # Frame-skip: return cached result on non-inference frames.
            if self._frame_skip > 1 and self._call_count % self._frame_skip != 1:
                return self._last_result

            try:
                results = self._model.track(
                    frame,
                    tracker="bytetrack.yaml",
                    persist=True,
                    classes=[0],  # person only
                    conf=settings.YOLO_CONFIDENCE,
                    iou=settings.YOLO_IOU_THRESHOLD,
                    device=self._device,
                    verbose=False,
                )
            except Exception as exc:
                logger.error("YOLO inference error: %s", exc)
                return self._last_result

            # model.track() returns None when no objects are detected.
            if results is None or len(results) == 0:
                self._last_result = []
                return []

            result = results[0]

            # Boxes may be None if tracking returns no persons.
            if result.boxes is None or result.boxes.id is None:
                self._last_result = []
                return []

            detections: list[Detection] = []
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, tid, conf, cid in zip(boxes_xyxy, track_ids, confidences, class_ids):
                detections.append(
                    Detection(
                        track_id=int(tid),
                        bbox_xyxy=box.astype(np.float32),
                        confidence=float(conf),
                        class_id=int(cid),
                    )
                )

            self._last_result = detections
            return detections
