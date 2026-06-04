"""Shared utility functions for the CCTV-Edge-Intelligence pipeline."""
from __future__ import annotations

import logging
import logging.handlers
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.detector import Detection

logger = logging.getLogger(__name__)


def extract_crops(
    frame: np.ndarray,
    detections: list[Detection],
    min_crop_area: int = 500,
) -> tuple[list[Detection], list[np.ndarray]]:
    """Extract and validate bounding box crops from a frame.

    Filters detections whose crop area is below ``min_crop_area`` to avoid
    encoding noise from tiny or partially visible persons.

    Args:
        frame: Full BGR frame as an (H, W, 3) numpy array.
        detections: List of ``Detection`` objects from the detector.
        min_crop_area: Minimum pixel area (width × height) for a crop to be
            retained. Crops below this threshold are discarded.

    Returns:
        A tuple of (valid_detections, crops) where both lists are parallel —
        element i of crops corresponds to element i of valid_detections.
    """
    h, w = frame.shape[:2]
    valid_detections: list[Detection] = []
    crops: list[np.ndarray] = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        # Clamp to frame boundaries to handle detections that extend beyond edges.
        x1c = max(0, int(x1))
        y1c = max(0, int(y1))
        x2c = min(w, int(x2))
        y2c = min(h, int(y2))

        crop_w = x2c - x1c
        crop_h = y2c - y1c

        if crop_w * crop_h < min_crop_area:
            continue

        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue

        valid_detections.append(det)
        crops.append(crop)

    return valid_detections, crops


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    reid_assignments: dict[int, int],
    fps: float,
    queue_depth: int,
    queue_max: int,
    gallery_size: int = 0,
) -> np.ndarray:
    """Draw bounding boxes, track/reid labels, and a HUD overlay on the frame.

    Modifies ``frame`` in-place and returns it for chaining.

    Args:
        frame: BGR frame to annotate (modified in-place).
        detections: Detected persons for this frame.
        reid_assignments: Mapping of track_id → reid_id for display.
        fps: Current rolling FPS estimate.
        queue_depth: Current frame queue depth.
        queue_max: Maximum frame queue capacity.
        gallery_size: Number of identities in the FAISS feature bank.

    Returns:
        The annotated frame (same object as ``frame``).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    box_color = (0, 255, 0)
    text_color = (255, 255, 255)
    bg_color = (0, 128, 0)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        reid_id = reid_assignments.get(det.track_id, -1)
        label = f"T:{det.track_id} R:{reid_id} ({det.confidence:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y1 - 5, lh + 5)
        cv2.rectangle(
            frame,
            (x1, label_y - lh - baseline - 2),
            (x1 + lw, label_y + baseline),
            bg_color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x1, label_y - baseline),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    # HUD: semi-transparent background rectangle in top-left corner.
    hud_text = (
        f"FPS: {fps:5.1f} | Tracked: {len(detections)} "
        f"| Gallery: {gallery_size} | Queue: {queue_depth}/{queue_max}"
    )
    (hw, hh), hb = cv2.getTextSize(hud_text, font, 0.6, 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (hw + 15, hh + hb + 15), (0, 0, 0), cv2.FILLED)
    # Alpha blend for semi-transparency.
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(
        frame,
        hud_text,
        (10, hh + 10),
        font,
        0.6,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger with a standard format.

    Args:
        log_level: One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        log_file: Optional path for a rotating file handler. When ``None``,
            only a console handler is attached.

    Raises:
        ValueError: If ``log_level`` is not a valid logging level name.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level!r}")

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any handlers already attached (e.g., from a prior call in tests).
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)


def auto_select_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, else 'cpu'.

    Returns:
        'cuda' or 'cpu' as a string suitable for passing to PyTorch.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
