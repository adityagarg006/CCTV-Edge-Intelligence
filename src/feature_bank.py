"""FAISS-backed Re-ID feature gallery with EMA update strategy.

Index design rationale:
-----------------------
``IndexFlatL2`` performs exact nearest-neighbour search. For a gallery of
10–500 identities, exact search is definitively faster than approximate indexes
(IVFFlat, HNSW) because ANN indexes incur training and clustering overhead that
only pays off beyond ~100K vectors. Exact search on 500 float32 256-dim vectors
takes <0.1ms — the search step is negligible.

``IndexIDMap`` wraps the flat index and maps arbitrary integer IDs (track_ids /
reid_ids) to FAISS internal row indices. This eliminates the need for a
secondary mapping table and allows direct lookup and deletion by external ID.

EMA update rationale:
---------------------
Raw embeddings vary across poses, lighting angles, and partial occlusions.
Replacing the gallery vector on every update would thrash the representation.
Computing the mean over all observed embeddings requires storing all of them.
EMA (α=0.9) smooths the gallery toward a stable centroid with O(1) storage:
    new = α * old + (1 - α) * incoming
After re-normalisation this approximates the running mean direction in the
embedding hypersphere.

remove_ids API note:
--------------------
``IndexIDMap(IndexFlatL2).remove_ids`` requires an ``IDSelectorBatch`` object,
NOT a raw numpy array. Passing a numpy array raises a FAISS TypeError at
runtime — a common mistake that is silently accepted by some FAISS builds but
fails on others. Always use ``IDSelectorBatch``.
"""
from __future__ import annotations

import logging
import threading

import faiss
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class FeatureBank:
    """Thread-safe FAISS gallery for person Re-ID embeddings.

    Args:
        embed_dim: Dimensionality of the embedding vectors (must match the
            ``ReIDEncoder.embed_dim`` used to produce them).

    Raises:
        ValueError: If ``embed_dim`` is not a positive integer.
    """

    def __init__(self, embed_dim: int) -> None:
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")

        self._embed_dim = embed_dim
        self._lock = threading.RLock()

        flat = faiss.IndexFlatL2(embed_dim)
        self._index = faiss.IndexIDMap(flat)

        # Maintain a Python-side dict for EMA retrieval without a second FAISS
        # search: reid_id → current gallery embedding (numpy float32 vector).
        self._gallery: dict[int, np.ndarray] = {}

        logger.info("FeatureBank initialised with embed_dim=%d", embed_dim)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        reid_id: int,
        embedding: np.ndarray,
        alpha: float = settings.REID_EMA_ALPHA,
    ) -> None:
        """Insert or EMA-update a gallery entry for ``reid_id``.

        If ``reid_id`` already exists in the gallery, the stored embedding is
        updated via EMA and the FAISS index entry is replaced. If new, the
        embedding is inserted directly.

        Args:
            reid_id: Integer identity key (may be a ByteTrack track_id or a
                Re-ID-assigned stable id).
            embedding: L2-normalised float32 vector of shape ``(embed_dim,)``.
            alpha: EMA smoothing factor in [0, 1]. Higher = more weight on the
                existing gallery representation (slower adaptation).

        Raises:
            ValueError: If ``embedding`` has the wrong shape.
        """
        emb = embedding.flatten().astype(np.float32)
        if emb.shape[0] != self._embed_dim:
            raise ValueError(
                f"Expected embedding dim={self._embed_dim}, got {emb.shape[0]}"
            )

        with self._lock:
            if reid_id in self._gallery:
                old_emb = self._gallery[reid_id]
                new_emb = alpha * old_emb + (1.0 - alpha) * emb
                # Re-normalise to keep the gallery on the L2 unit sphere.
                norm = np.linalg.norm(new_emb)
                if norm > 0:
                    new_emb = new_emb / norm

                # ``remove_ids`` on IndexIDMap(IndexFlatL2) requires
                # ``IDSelectorBatch``; a plain numpy array is NOT accepted.
                ids_to_remove = np.array([reid_id], dtype=np.int64)
                selector = faiss.IDSelectorBatch(ids_to_remove)
                self._index.remove_ids(selector)

                self._gallery[reid_id] = new_emb
                vec = new_emb.reshape(1, -1)
            else:
                self._gallery[reid_id] = emb
                vec = emb.reshape(1, -1)

            ids = np.array([reid_id], dtype=np.int64)
            self._index.add_with_ids(vec, ids)

    def query(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find the k nearest gallery entries to ``embedding``.

        Filters results whose L2 distance exceeds
        ``settings.REID_DISTANCE_THRESHOLD``.

        Args:
            embedding: Query vector of shape ``(embed_dim,)``, float32,
                L2-normalised.
            k: Maximum number of nearest neighbours to return.

        Returns:
            List of ``(reid_id, l2_distance)`` tuples sorted by ascending
            distance. Empty if the gallery is empty or all results exceed the
            distance threshold.
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []

            k_actual = min(k, self._index.ntotal)
            vec = embedding.flatten().astype(np.float32).reshape(1, -1)
            distances, ids = self._index.search(vec, k_actual)

            results: list[tuple[int, float]] = []
            for dist, rid in zip(distances[0], ids[0]):
                if rid == -1:
                    continue  # FAISS sentinel for "no result"
                if dist <= settings.REID_DISTANCE_THRESHOLD:
                    results.append((int(rid), float(dist)))

        return results

    def size(self) -> int:
        """Return the number of identities currently in the gallery.

        Returns:
            Non-negative integer count.
        """
        with self._lock:
            return self._index.ntotal

    def clear(self) -> None:
        """Remove all entries from the gallery and reset the FAISS index.

        After calling ``clear()``, ``size()`` returns 0.
        """
        with self._lock:
            self._index.reset()
            self._gallery.clear()
            logger.info("FeatureBank cleared.")
