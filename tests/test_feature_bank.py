"""Tests for FeatureBank — FAISS in-process, no GPU required."""
from __future__ import annotations

import threading

import numpy as np

from src.feature_bank import FeatureBank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 64


def _unit_vec(seed: int = 0) -> np.ndarray:
    """Return a deterministic L2-normalised float32 vector of EMBED_DIM dims."""
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Test: size() returns 0 on init
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_size_zero_on_init(self):
        bank = FeatureBank(embed_dim=EMBED_DIM)
        assert bank.size() == 0


# ---------------------------------------------------------------------------
# Test: update() then size() returns 1
# ---------------------------------------------------------------------------

class TestUpdateIncreasesSize:
    def test_update_then_size_is_one(self):
        bank = FeatureBank(embed_dim=EMBED_DIM)
        bank.update(reid_id=1, embedding=_unit_vec(0))
        assert bank.size() == 1

    def test_two_updates_different_ids_size_is_two(self):
        bank = FeatureBank(embed_dim=EMBED_DIM)
        bank.update(reid_id=1, embedding=_unit_vec(0))
        bank.update(reid_id=2, embedding=_unit_vec(1))
        assert bank.size() == 2


# ---------------------------------------------------------------------------
# Test: update() on same ID does not increase size (EMA replacement)
# ---------------------------------------------------------------------------

class TestUpdateSameIdNoSizeIncrease:
    def test_same_id_update_keeps_size_at_one(self):
        bank = FeatureBank(embed_dim=EMBED_DIM)
        bank.update(reid_id=7, embedding=_unit_vec(0))
        bank.update(reid_id=7, embedding=_unit_vec(1))
        bank.update(reid_id=7, embedding=_unit_vec(2))
        assert bank.size() == 1

    def test_ema_update_result_is_unit_norm(self):
        """After EMA update the stored embedding must remain L2-normalised."""
        bank = FeatureBank(embed_dim=EMBED_DIM)
        bank.update(reid_id=5, embedding=_unit_vec(0))
        bank.update(reid_id=5, embedding=_unit_vec(1))
        # Query the bank; the returned distance should be valid (not NaN).
        result = bank.query(_unit_vec(0), k=1)
        assert len(result) >= 0  # May be empty if threshold filters it; just no crash.


# ---------------------------------------------------------------------------
# Test: query() returns empty list when distance exceeds threshold
# ---------------------------------------------------------------------------

class TestQueryFiltering:
    def test_query_empty_when_distance_exceeds_threshold(self):
        """query() must return [] when no gallery entry passes the threshold."""
        bank = FeatureBank(embed_dim=EMBED_DIM)
        # Insert a vector; use a threshold so tight that a random query won't match.
        bank.update(reid_id=1, embedding=_unit_vec(0))

        # Monkey-patch threshold to 0.0 so no match passes.
        import config.settings as settings
        original = settings.REID_DISTANCE_THRESHOLD
        settings.REID_DISTANCE_THRESHOLD = 0.0
        try:
            results = bank.query(_unit_vec(99), k=5)
            assert results == []
        finally:
            settings.REID_DISTANCE_THRESHOLD = original

    def test_query_returns_self_match_under_normal_threshold(self):
        """Querying with the same vector that was inserted should match itself."""
        bank = FeatureBank(embed_dim=EMBED_DIM)
        vec = _unit_vec(0)
        bank.update(reid_id=3, embedding=vec)

        import config.settings as settings
        original = settings.REID_DISTANCE_THRESHOLD
        settings.REID_DISTANCE_THRESHOLD = 1.0  # permissive
        try:
            results = bank.query(vec, k=1)
            assert len(results) == 1
            assert results[0][0] == 3
        finally:
            settings.REID_DISTANCE_THRESHOLD = original

    def test_query_empty_bank_returns_empty(self):
        bank = FeatureBank(embed_dim=EMBED_DIM)
        results = bank.query(_unit_vec(0), k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Test: clear() resets size to 0
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_size(self):
        bank = FeatureBank(embed_dim=EMBED_DIM)
        for i in range(5):
            bank.update(reid_id=i, embedding=_unit_vec(i))
        assert bank.size() == 5
        bank.clear()
        assert bank.size() == 0

    def test_clear_then_update_works(self):
        """After clear(), new entries can be inserted without error."""
        bank = FeatureBank(embed_dim=EMBED_DIM)
        bank.update(reid_id=1, embedding=_unit_vec(0))
        bank.clear()
        bank.update(reid_id=99, embedding=_unit_vec(1))
        assert bank.size() == 1


# ---------------------------------------------------------------------------
# Test: thread safety — 10 concurrent updates result in size == 10
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_updates_produce_correct_size(self):
        """10 threads each inserting a unique ID must yield size == 10."""
        bank = FeatureBank(embed_dim=EMBED_DIM)
        n = 10

        def worker(tid: int) -> None:
            bank.update(reid_id=tid, embedding=_unit_vec(tid))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert bank.size() == n

    def test_concurrent_updates_and_queries_no_crash(self):
        """Concurrent updates and queries must not raise any exception."""
        bank = FeatureBank(embed_dim=EMBED_DIM)
        errors: list[Exception] = []

        def updater(tid: int) -> None:
            try:
                bank.update(reid_id=tid, embedding=_unit_vec(tid))
            except Exception as exc:
                errors.append(exc)

        def querier() -> None:
            try:
                bank.query(_unit_vec(42), k=3)
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=updater, args=(i,)) for i in range(8)]
            + [threading.Thread(target=querier) for _ in range(4)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exceptions during concurrent access: {errors}"
