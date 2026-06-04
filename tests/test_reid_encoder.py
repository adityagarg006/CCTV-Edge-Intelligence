"""Tests for ReIDEncoder — runs on CPU without torchreid installed."""
from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bgr_crop(h: int = 256, w: int = 128) -> np.ndarray:
    """Return a synthetic BGR crop filled with random uint8 values."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test: encode_batch([]) returns shape (0, embed_dim)
# ---------------------------------------------------------------------------

class TestEncodeBatchEmpty:
    def test_empty_list_returns_zero_rows(self):
        """encode_batch([]) must return an array of shape (0, embed_dim)."""
        from src.reid_encoder import ReIDEncoder

        encoder = ReIDEncoder(device="cpu")
        result = encoder.encode_batch([])
        assert result.shape == (0, encoder.embed_dim)
        assert result.dtype == np.float32

    def test_empty_list_does_not_raise(self):
        """encode_batch([]) must not raise any exception."""
        from src.reid_encoder import ReIDEncoder

        encoder = ReIDEncoder(device="cpu")
        try:
            encoder.encode_batch([])
        except Exception as exc:
            pytest.fail(f"encode_batch([]) raised {exc!r}")


# ---------------------------------------------------------------------------
# Test: output is L2-normalised
# ---------------------------------------------------------------------------

class TestL2Normalisation:
    def test_single_encode_output_is_unit_norm(self):
        """encode() must return an L2-normalised vector (‖v‖₂ ≈ 1.0)."""
        from src.reid_encoder import ReIDEncoder

        encoder = ReIDEncoder(device="cpu")
        crop = _make_bgr_crop()
        vec = encoder.encode(crop)
        norm = np.linalg.norm(vec)
        assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_batch_encode_all_outputs_are_unit_norm(self):
        """encode_batch() must return unit-norm vectors for all crops."""
        from src.reid_encoder import ReIDEncoder

        encoder = ReIDEncoder(device="cpu")
        crops = [_make_bgr_crop() for _ in range(4)]
        embeddings = encoder.encode_batch(crops)
        assert embeddings.shape == (4, encoder.embed_dim)
        for i, vec in enumerate(embeddings):
            norm = np.linalg.norm(vec)
            assert pytest.approx(norm, abs=1e-5) == 1.0, f"Row {i} not unit-norm (‖v‖={norm})"

    def test_output_dtype_is_float32(self):
        """encode_batch() must return float32 arrays."""
        from src.reid_encoder import ReIDEncoder

        encoder = ReIDEncoder(device="cpu")
        embeddings = encoder.encode_batch([_make_bgr_crop()])
        assert embeddings.dtype == np.float32


# ---------------------------------------------------------------------------
# Test: fallback path activates when torchreid is absent
# ---------------------------------------------------------------------------

class TestFallbackPath:
    def test_mobilenetv3_fallback_when_torchreid_missing(self):
        """ReIDEncoder must fall back to MobileNetV3-Small if torchreid is absent."""
        # Remove torchreid from sys.modules so the import inside _load_model fails.
        saved = sys.modules.pop("torchreid", None)
        # Also make the import itself raise ImportError.
        with patch.dict(sys.modules, {"torchreid": None}):
            # Reload to force _load_model to re-run.
            import importlib
            import src.reid_encoder as reid_mod

            # Directly test by patching builtins.
            import builtins
            real_import = builtins.__import__

            def _no_torchreid(name, *args, **kwargs):
                if name == "torchreid":
                    raise ImportError("Mocked: torchreid not installed")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_no_torchreid):
                importlib.reload(reid_mod)
                encoder = reid_mod.ReIDEncoder(device="cpu")

            # MobileNetV3-Small produces 576-dim embeddings.
            assert encoder.embed_dim == 576

            # Restore the module after the test.
            importlib.reload(reid_mod)

        if saved is not None:
            sys.modules["torchreid"] = saved

    def test_embed_dim_attribute_exists(self):
        """ReIDEncoder must expose an embed_dim integer attribute."""
        from src.reid_encoder import ReIDEncoder

        encoder = ReIDEncoder(device="cpu")
        assert isinstance(encoder.embed_dim, int)
        assert encoder.embed_dim > 0
