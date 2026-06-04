"""Deterministic synthetic video generator for reproducible benchmarks.

Why synthetic video?
- Avoids copyright/licensing issues with real surveillance footage.
- Guarantees deterministic ground truth: same seed -> identical video bytes,
  enabling regression testing and CI reproducibility.
- Enables benchmarking without camera hardware or network streams.
- Isolates the pipeline from real-world variability (lighting, occlusion,
  resolution variation) so benchmark numbers reflect code performance.
"""
from __future__ import annotations
import logging, os
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Person rectangle dimensions (w, h). Tall-and-narrow to resemble human silhouettes.
_PERSON_W = 80
_PERSON_H = 200


def _person_positions(
    width: int,
    height: int,
    n_persons: int,
    frame_idx: int,
    rng: np.random.Generator,
    speeds: np.ndarray,
    origins: np.ndarray,
) -> list[tuple[int, int]]:
    """Return top-left (x, y) for each person at frame_idx using linear motion."""
    positions = []
    for i in range(n_persons):
        ox, oy = int(origins[i, 0]), int(origins[i, 1])
        sx, sy = float(speeds[i, 0]), float(speeds[i, 1])
        x = int((ox + sx * frame_idx) % (width - _PERSON_W))
        y = int((oy + sy * frame_idx) % (height - _PERSON_H))
        positions.append((max(0, x), max(0, y)))
    return positions


def generate_synthetic_frame(
    width: int = 1920,
    height: int = 1080,
    n_persons: int = 5,
    frame_idx: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """Generate one synthetic BGR frame with n_persons moving colored rectangles.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        n_persons: Number of person-like rectangles to render.
        frame_idx: Frame index used to compute trajectory position.
        seed: RNG seed for full reproducibility.

    Returns:
        BGR uint8 numpy array of shape (height, width, 3).
    """
    rng = np.random.default_rng(seed)
    # Each person gets a distinct saturated colour for visual contrast.
    colours = [tuple(int(c) for c in rng.integers(80, 230, size=3)) for _ in range(n_persons)]
    speeds = rng.uniform(1.0, 4.0, size=(n_persons, 2))
    origins = rng.integers(0, [width - _PERSON_W, height - _PERSON_H], size=(n_persons, 2))

    # Dark grey background gives strong contrast with coloured rectangles.
    frame = np.full((height, width, 3), 30, dtype=np.uint8)

    positions = _person_positions(width, height, n_persons, frame_idx, rng, speeds, origins)
    for (x, y), colour in zip(positions, colours):
        cv2.rectangle(frame, (x, y), (x + _PERSON_W, y + _PERSON_H), colour, cv2.FILLED)
        # Inner darker stripe to add texture so Re-ID embeddings are non-trivial.
        cv2.rectangle(
            frame,
            (x + 20, y + 40),
            (x + _PERSON_W - 20, y + _PERSON_H - 40),
            tuple(max(0, c - 60) for c in colour),
            cv2.FILLED,
        )

    return frame


def generate_synthetic_video(
    path: str,
    n_frames: int = 300,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    n_persons: int = 5,
    seed: int = 42,
) -> str:
    """Write a deterministic synthetic .mp4 to path and return the path.

    Args:
        path: Output file path (must end in .mp4).
        n_frames: Total number of frames to write.
        width: Frame width.
        height: Frame height.
        fps: Playback frame rate encoded into the container.
        n_persons: Number of moving person rectangles per frame.
        seed: RNG seed; same seed always produces identical bytes.

    Returns:
        Absolute path to the written file.

    Raises:
        RuntimeError: If cv2.VideoWriter fails to open the output file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path!r}. Check codec/path.")

    rng = np.random.default_rng(seed)
    colours = [tuple(int(c) for c in rng.integers(80, 230, size=3)) for _ in range(n_persons)]
    speeds = rng.uniform(1.0, 4.0, size=(n_persons, 2))
    origins = rng.integers(0, [width - _PERSON_W, height - _PERSON_H], size=(n_persons, 2))

    for idx in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        positions = _person_positions(width, height, n_persons, idx, rng, speeds, origins)
        for (x, y), colour in zip(positions, colours):
            cv2.rectangle(frame, (x, y), (x + _PERSON_W, y + _PERSON_H), colour, cv2.FILLED)
            cv2.rectangle(
                frame,
                (x + 20, y + 40),
                (x + _PERSON_W - 20, y + _PERSON_H - 40),
                tuple(max(0, c - 60) for c in colour),
                cv2.FILLED,
            )
        writer.write(frame)

    writer.release()
    logger.info("Synthetic video written: %s (%d frames, %dx%d @ %dfps)", path, n_frames, width, height, fps)
    return os.path.abspath(path)
