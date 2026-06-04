"""Asynchronous video frame producer for CCTV-Edge-Intelligence.

Why a dedicated producer thread?
---------------------------------
Synchronous ``cv2.VideoCapture.read()`` in the main loop blocks for ≈33ms at
30 fps (or longer on RTSP due to network jitter). During that block, the GPU
inference pipeline sits completely idle. By moving capture to a background
thread, the queue stays pre-filled and the GPU pipeline can pull the next
frame immediately after finishing inference — eliminating I/O-induced GPU
starvation.

Extensibility note: This class manages one source. For multi-camera
deployments, instantiate one ``AsyncVideoReader`` per source and merge their
outputs in the main loop or a dedicated multiplexer thread.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Union

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class AsyncVideoReader:
    """Background-thread video reader with configurable frame-drop strategy.

    A producer thread calls ``cv2.VideoCapture.read()`` in a tight loop and
    pushes decoded frames into an internal ``queue.Queue``. The main
    (consumer) thread calls ``read()`` which is non-blocking.

    Drop strategy — oldest frame, not newest:
        When the queue is full, the *oldest* frame is dropped before the new
        one is enqueued. Dropping the oldest frame ensures the consumer always
        gets the most recent available frame. Dropping the newest (i.e.,
        silently discarding the incoming frame) would leave the consumer
        draining a queue full of stale frames while live data is thrown away.

    Args:
        source: Webcam index (``int``), local file path (``str``), or RTSP URL
            (``str`` starting with ``rtsp://``).
        maxsize: Maximum frames buffered in the queue before the drop policy
            activates. Defaults to ``settings.FRAME_QUEUE_MAXSIZE``.
    """

    def __init__(
        self,
        source: Union[int, str],
        maxsize: int = settings.FRAME_QUEUE_MAXSIZE,
    ) -> None:
        self._source = source
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._producer,
            daemon=True,
            name="frame-producer",
        )
        self._thread.start()
        logger.info("AsyncVideoReader started for source=%r qsize=%d", source, maxsize)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Non-blocking frame read with a short timeout.

        Returns:
            ``(True, frame)`` if a frame was available, or ``(False, None)``
            if the queue was empty within the timeout window.
        """
        try:
            frame = self._queue.get(timeout=0.05)
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        """Signal the producer thread to stop and wait for it to join.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if not self._stop_event.is_set():
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            logger.info("AsyncVideoReader stopped.")

    def qsize(self) -> int:
        """Return the approximate number of frames currently in the queue.

        Returns:
            Current queue depth (0 … maxsize).
        """
        return self._queue.qsize()

    def __enter__(self) -> "AsyncVideoReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal producer
    # ------------------------------------------------------------------

    def _is_rtsp(self) -> bool:
        return isinstance(self._source, str) and self._source.lower().startswith("rtsp://")

    def _is_file(self) -> bool:
        return isinstance(self._source, str) and not self._is_rtsp()

    def _producer(self) -> None:
        """Main loop of the background producer thread."""
        cap = cv2.VideoCapture(self._source)

        if not cap.isOpened():
            logger.error("Cannot open source: %r", self._source)
            self._stop_event.set()
            return

        while not self._stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                # Distinguish between source types for appropriate recovery.
                if self._is_file():
                    logger.info("End of file reached for source=%r. Stopping.", self._source)
                    self._stop_event.set()
                    break
                elif self._is_rtsp():
                    self._reconnect_rtsp(cap)
                    if self._stop_event.is_set():
                        break
                    continue
                else:
                    # Integer webcam — hardware disconnect or driver error.
                    logger.error("Webcam read failed for source=%r. Stopping.", self._source)
                    self._stop_event.set()
                    break
                continue

            # Drop-oldest strategy: if the queue is full, remove the stale
            # front element before inserting the fresh frame.
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass  # Another consumer drained it between our check and get.
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass  # Extremely rare race; skip this frame rather than block.

        cap.release()
        logger.debug("Producer thread exiting.")

    def _reconnect_rtsp(self, cap: cv2.VideoCapture) -> None:
        """Exponential-backoff reconnect loop for RTSP stream drops.

        Args:
            cap: The ``cv2.VideoCapture`` object to reopen in-place.
        """
        delay = 1.0
        max_delay = 30.0
        attempt = 0

        while not self._stop_event.is_set():
            attempt += 1
            logger.warning(
                "RTSP stream lost for %r. Reconnect attempt #%d in %.0fs.",
                self._source,
                attempt,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, max_delay)

            if cap.open(self._source):
                logger.info("RTSP reconnected to %r after %d attempt(s).", self._source, attempt)
                return

        logger.warning("Reconnect loop exited because stop was requested.")
