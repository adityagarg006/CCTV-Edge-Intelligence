"""Tests for PersonDetector — mocks ultralytics YOLO, no GPU required."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector import Detection, PersonDetector


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_fake_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _mock_yolo_result(
    track_ids: list[int],
    bboxes: list[list[float]],
    confs: list[float],
    class_ids: list[int],
) -> MagicMock:
    """Build a minimal ultralytics Results mock."""
    import torch

    boxes = MagicMock()
    boxes.id = torch.tensor(track_ids, dtype=torch.float32) if track_ids else None
    boxes.xyxy = torch.tensor(bboxes, dtype=torch.float32)
    boxes.conf = torch.tensor(confs, dtype=torch.float32)
    boxes.cls = torch.tensor(class_ids, dtype=torch.float32)

    result = MagicMock()
    result.boxes = boxes if track_ids else None

    mock_results = [result]
    return mock_results


def _build_detector_with_mock_yolo(track_results=None, frame_skip: int = 1) -> PersonDetector:
    """Return a PersonDetector with a mocked YOLO model."""
    mock_model = MagicMock()
    mock_model.track.return_value = track_results

    with patch("src.detector.YOLO", return_value=mock_model):
        detector = PersonDetector(device="cpu", frame_skip=frame_skip)
        detector._model = mock_model  # ensure the right mock is in place

    return detector


# ---------------------------------------------------------------------------
# Test: detect() returns [] when model.track() returns None
# ---------------------------------------------------------------------------

class TestDetectReturnsEmptyOnNone:
    def test_returns_empty_list_when_track_returns_none(self):
        """detect() must return [] when model.track() returns None."""
        detector = _build_detector_with_mock_yolo(track_results=None)
        result = detector.detect(_make_fake_frame())
        assert result == []

    def test_returns_empty_list_when_track_returns_empty_list(self):
        """detect() must handle an empty list from model.track()."""
        detector = _build_detector_with_mock_yolo(track_results=[])
        result = detector.detect(_make_fake_frame())
        assert result == []


# ---------------------------------------------------------------------------
# Test: returned objects are Detection dataclass instances
# ---------------------------------------------------------------------------

class TestDetectionDataclass:
    def test_returns_detection_instances(self):
        """detect() must return a list of Detection dataclass objects."""
        results = _mock_yolo_result(
            track_ids=[1],
            bboxes=[[10.0, 20.0, 100.0, 200.0]],
            confs=[0.85],
            class_ids=[0],
        )
        detector = _build_detector_with_mock_yolo(track_results=results)
        detections = detector.detect(_make_fake_frame())

        assert len(detections) == 1
        det = detections[0]
        assert isinstance(det, Detection)
        assert det.track_id == 1
        assert det.confidence == pytest.approx(0.85, abs=1e-3)
        assert det.class_id == 0
        assert det.bbox_xyxy.shape == (4,)
        assert det.bbox_xyxy.dtype == np.float32

    def test_multiple_detections_returned(self):
        """detect() must return all detections from a multi-person frame."""
        results = _mock_yolo_result(
            track_ids=[1, 2, 3],
            bboxes=[
                [0.0, 0.0, 50.0, 100.0],
                [50.0, 0.0, 150.0, 200.0],
                [200.0, 10.0, 300.0, 300.0],
            ],
            confs=[0.9, 0.75, 0.6],
            class_ids=[0, 0, 0],
        )
        detector = _build_detector_with_mock_yolo(track_results=results)
        detections = detector.detect(_make_fake_frame())
        assert len(detections) == 3


# ---------------------------------------------------------------------------
# Test: frame_skip=3 returns cached result on frames 2 and 3
# ---------------------------------------------------------------------------

class TestFrameSkip:
    def test_frame_skip_3_uses_cache_on_frames_2_and_3(self):
        """With frame_skip=3, only frame 1 triggers inference; 2 and 3 return cache."""
        results = _mock_yolo_result(
            track_ids=[42],
            bboxes=[[5.0, 5.0, 100.0, 200.0]],
            confs=[0.88],
            class_ids=[0],
        )
        detector = _build_detector_with_mock_yolo(track_results=results, frame_skip=3)
        frame = _make_fake_frame()

        first = detector.detect(frame)    # frame 1 — runs inference
        second = detector.detect(frame)   # frame 2 — uses cache
        third = detector.detect(frame)    # frame 3 — uses cache

        assert len(first) == 1
        # track() should have been called exactly once across the 3 calls.
        assert detector._model.track.call_count == 1
        # All three results should be identical (same cached list object).
        assert second == first
        assert third == first

    def test_frame_skip_1_calls_inference_every_frame(self):
        """With frame_skip=1 (default), every detect() call triggers inference."""
        results = _mock_yolo_result(
            track_ids=[1],
            bboxes=[[0.0, 0.0, 50.0, 100.0]],
            confs=[0.7],
            class_ids=[0],
        )
        detector = _build_detector_with_mock_yolo(track_results=results, frame_skip=1)
        frame = _make_fake_frame()

        detector.detect(frame)
        detector.detect(frame)
        detector.detect(frame)

        assert detector._model.track.call_count == 3
