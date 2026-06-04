"""Tests for AsyncVideoReader — no real camera or GPU required."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np

from src.stream_reader import AsyncVideoReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_frame(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _build_reader_with_mock_cap(frames: list[tuple[bool, np.ndarray | None]]):
    """Return (reader, mock_cap) where the mock returns ``frames`` sequentially."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = frames + [(False, None)] * 100  # tail of failures

    with patch("cv2.VideoCapture", return_value=mock_cap):
        reader = AsyncVideoReader(source="fake_file.mp4", maxsize=4)
    return reader, mock_cap


# ---------------------------------------------------------------------------
# Test: read() returns (False, None) when queue is empty and timeout elapses
# ---------------------------------------------------------------------------

class TestReadOnEmptyQueue:
    def test_returns_false_none_when_queue_is_empty(self):
        """read() must return (False, None) with no frame available."""
        frames = [(True, _make_fake_frame())]
        reader, _ = _build_reader_with_mock_cap(frames)

        # Drain whatever the producer might have pushed.
        time.sleep(0.15)
        while True:
            ok, _ = reader.read()
            if not ok:
                break

        # Queue is now empty; next read must time out.
        ok, frame = reader.read()
        assert ok is False
        assert frame is None

        reader.stop()

    def test_returns_true_frame_when_frame_available(self):
        """read() returns (True, ndarray) when the queue has a frame."""
        frame_data = _make_fake_frame()
        frames = [(True, frame_data)] * 5
        reader, _ = _build_reader_with_mock_cap(frames)
        time.sleep(0.15)

        ok, frame = reader.read()
        assert ok is True
        assert isinstance(frame, np.ndarray)

        reader.stop()


# ---------------------------------------------------------------------------
# Test: producer drops oldest frame on queue.Full
# ---------------------------------------------------------------------------

class TestDropOldestOnFull:
    def test_producer_drops_oldest_frame_when_queue_full(self):
        """When maxsize=1, the producer must evict the stale frame and push the new one."""
        old_frame = np.full((32, 32, 3), 10, dtype=np.uint8)
        new_frame = np.full((32, 32, 3), 99, dtype=np.uint8)

        # Feed exactly two frames so we can observe the drop behaviour.
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, old_frame),
            (True, new_frame),
            (False, None),  # signal EOF
        ] + [(False, None)] * 100

        with patch("cv2.VideoCapture", return_value=mock_cap):
            reader = AsyncVideoReader(source="test.mp4", maxsize=1)

        # Allow both frames to be produced.
        time.sleep(0.3)

        ok, frame = reader.read()
        assert ok is True
        # Because maxsize=1, the old_frame was evicted; we should see the newer one.
        assert frame is not None
        # The newest frame has pixel value 99; the oldest has 10.
        assert int(frame[0, 0, 0]) == 99, "Expected oldest frame to be dropped in favour of newest"

        reader.stop()


# ---------------------------------------------------------------------------
# Test: context manager calls stop()
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_exit_calls_stop(self):
        """__exit__ must invoke stop(), causing the producer thread to join."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with AsyncVideoReader(source="test.mp4", maxsize=4) as reader:
                thread = reader._thread

        # After __exit__, the thread should have terminated.
        thread.join(timeout=3.0)
        assert not thread.is_alive(), "Producer thread still alive after __exit__"

    def test_enter_returns_self(self):
        """__enter__ must return the AsyncVideoReader instance itself."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            reader = AsyncVideoReader(source="test.mp4", maxsize=4)

        result = reader.__enter__()
        assert result is reader
        reader.stop()


# ---------------------------------------------------------------------------
# Test: qsize reflects queue depth
# ---------------------------------------------------------------------------

class TestQsize:
    def test_qsize_returns_non_negative_int(self):
        """qsize() must return a non-negative integer at all times."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            reader = AsyncVideoReader(source="test.mp4", maxsize=4)

        assert isinstance(reader.qsize(), int)
        assert reader.qsize() >= 0
        reader.stop()
