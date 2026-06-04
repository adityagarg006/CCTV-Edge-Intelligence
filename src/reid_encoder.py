"""OSNet Re-ID feature encoder for CCTV-Edge-Intelligence.

Why OSNet x0.25 over ResNet50?
--------------------------------
OSNet's omni-scale feature learning uses unified aggregation gates to capture
both fine-grained texture cues and coarse body-part patterns simultaneously.
This multi-scale strategy achieves mAP 73.5% on Market-1501 vs ResNet50's
68.8%, with ~4× fewer parameters (~2.2M vs ~23.5M) and ~3× faster inference on
the same GPU. For an edge deployment budget, OSNet is the dominant choice.

Latency note: End-to-end Re-ID latency is dominated by OSNet/MobileNetV3
inference (5–15ms GPU). The FAISS query step (<1ms even on CPU) is negligible.
"Sub-millisecond Re-ID" is a common overclaim that confuses the index search
with the full encode-then-search pipeline.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet normalization constants used by both OSNet and MobileNetV3.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Re-ID crop target dimensions (height × width) per OSNet's training protocol.
_CROP_H = 256
_CROP_W = 128


class ReIDEncoder:
    """Encode person crops into L2-normalised Re-ID embeddings.

    Attempts to load OSNet x0.25 via ``torchreid``. If ``torchreid`` is not
    installed, falls back to MobileNetV3-Small (576-dim output). The
    ``embed_dim`` attribute exposes the embedding dimensionality so
    ``FeatureBank`` can initialise the FAISS index correctly without
    hard-coding model-specific dimensions.

    Args:
        device: Compute device string — ``'cuda'``, ``'cpu'``, or ``'auto'``.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self._model, self.embed_dim = self._load_model()
        self._model.to(self._device)
        self._model.eval()

        self._preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((_CROP_H, _CROP_W)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_MEAN, std=_STD),
            ]
        )
        logger.info(
            "ReIDEncoder ready: embed_dim=%d device=%s", self.embed_dim, self._device
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(self, crop: np.ndarray) -> np.ndarray:
        """Encode a single BGR crop into an L2-normalised embedding.

        Args:
            crop: BGR image crop as an (H, W, 3) numpy array.

        Returns:
            L2-normalised float32 embedding of shape ``(embed_dim,)``.
        """
        batch = self.encode_batch([crop])
        return batch[0]

    def encode_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Encode a list of BGR crops into a batch of L2-normalised embeddings.

        This is the primary call path — batching amortises model forward-pass
        overhead across all detections in a single frame.

        Args:
            crops: List of BGR image crops, each an (H, W, 3) numpy array.

        Returns:
            Float32 numpy array of shape ``(N, embed_dim)``. Returns
            ``np.zeros((0, embed_dim), dtype=np.float32)`` for an empty list.
        """
        if not crops:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        tensors: list[torch.Tensor] = []
        for crop in crops:
            # Convert BGR (OpenCV) → RGB before applying transforms.
            rgb_crop = crop[:, :, ::-1].copy()
            tensors.append(self._preprocess(rgb_crop))

        batch_tensor = torch.stack(tensors).to(self._device)

        with torch.no_grad():
            features = self._model(batch_tensor)
            # L2-normalise so cosine and L2 distances are equivalent.
            features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> tuple[nn.Module, int]:
        """Load OSNet or fall back to MobileNetV3-Small.

        Returns:
            Tuple of (model, embedding_dimension).
        """
        try:
            import torchreid

            model = torchreid.models.build_model(
                name="osnet_x0_25",
                num_classes=1000,
                pretrained=True,
            )
            # Strip the classifier head; OSNet's feature extractor outputs 256-dim.
            model.classifier = nn.Identity()
            embed_dim = 256
            logger.info("Loaded OSNet x0.25 (embed_dim=256) via torchreid.")
            return model, embed_dim

        except ImportError:
            logger.warning(
                "torchreid not found; falling back to MobileNetV3-Small (576-dim). "
                "Expect ~8%% lower Re-ID accuracy. "
                "Install: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            )
            import torchvision

            model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
            # MobileNetV3-Small's classifier input is 576-dim after AdaptiveAvgPool.
            model.classifier = nn.Identity()
            embed_dim = 576
            logger.info("Loaded MobileNetV3-Small (embed_dim=576) as Re-ID fallback.")
            return model, embed_dim
