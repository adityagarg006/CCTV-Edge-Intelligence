"""
Market-1501 Re-ID evaluation script.

Computes Rank-1, Rank-5, Rank-10 CMC accuracy and mAP for our ReIDEncoder.

--- HOW TO GET THE DATASET ---
Option A (easiest): Kaggle
  1. Install kaggle CLI:  pip install kaggle
  2. Get API token from https://www.kaggle.com/settings → Create New Token
  3. Place kaggle.json in ~/.kaggle/
  4. Run: kaggle datasets download pengcw1/market-1501 -p benchmarks/data/
  5. Unzip: python -c "import zipfile; zipfile.ZipFile('benchmarks/data/market-1501.zip').extractall('benchmarks/data/')"

Option B: torchreid auto-download (requires working mirror)
  python -c "import torchreid; torchreid.data.ImageDataManager(root='benchmarks/data', sources='market1501', ...)"

Option C: Manual download
  Request at http://www.liangzheng.org/Project/project_reid.html
  Extract to benchmarks/data/Market-1501-v15.09.15/

--- HOW TO GET MARKET-1501 PRETRAINED WEIGHTS ---
1. Go to https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md
2. Download "osnet_x0_25" row → "Market1501" column (.pth.tar file)
3. Place at: pretrained/osnet_x0_25_market1501.pth.tar

Run with:
  python benchmarks/eval_reid_market1501.py --data benchmarks/data/Market-1501-v15.09.15 [--weights pretrained/osnet_x0_25_market1501.pth.tar]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_market1501(root: str) -> tuple[list, list]:
    """Return (query_items, gallery_items) where each item is (img_path, pid, cam_id)."""
    import glob

    def _parse_dir(folder: str):
        items = []
        for fpath in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
            fname = os.path.basename(fpath)
            pid = int(fname[:4])
            cam = int(fname[6])
            if pid == -1:
                continue  # junk images
            items.append((fpath, pid, cam))
        return items

    query = _parse_dir(os.path.join(root, "query"))
    gallery = _parse_dir(os.path.join(root, "bounding_box_test"))
    return query, gallery


def extract_features(items: list, encoder, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run encoder over all images; return (embeddings, pids, camids)."""
    import cv2
    embeddings, pids, camids = [], [], []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        crops = []
        for fpath, pid, cam in batch:
            img = cv2.imread(fpath)
            if img is None:
                continue
            crops.append(img)
            pids.append(pid)
            camids.append(cam)
        if crops:
            embs = encoder.encode_batch(crops)
            embeddings.append(embs)
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(items)} images…", end="\r")

    print()
    return np.vstack(embeddings), np.array(pids), np.array(camids)


def compute_cmc_map(
    q_embs: np.ndarray,
    g_embs: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 10,
) -> tuple[np.ndarray, float]:
    """Compute CMC curve and mAP."""
    num_q = q_embs.shape[0]
    all_cmc = []
    all_ap = []

    # Pairwise L2 distances (vectorised)
    dists = np.sum((q_embs[:, None] - g_embs[None]) ** 2, axis=2)  # (Q, G)

    for q_idx in range(num_q):
        q_pid, q_cam = q_pids[q_idx], q_cams[q_idx]

        order = np.argsort(dists[q_idx])
        keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))

        ranked_pids = g_pids[order][keep][:max_rank + 100]
        matches = (ranked_pids == q_pid).astype(int)

        if matches.sum() == 0:
            continue  # query not in gallery (bad data)

        # CMC
        cmc = np.cumsum(matches[:max_rank]) > 0
        all_cmc.append(cmc.astype(float))

        # AP
        num_rel = matches.sum()
        tmp_cmc = np.cumsum(matches)
        tmp_cmc = tmp_cmc / (np.arange(len(matches)) + 1) * matches
        ap = tmp_cmc.sum() / num_rel
        all_ap.append(ap)

    cmc_curve = np.mean(all_cmc, axis=0)
    m_ap = float(np.mean(all_ap))
    return cmc_curve, m_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Market-1501-v15.09.15 folder")
    parser.add_argument("--weights", default=None, help="Optional .pth.tar Market-1501 pretrained weights")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"ERROR: dataset not found at {args.data}")
        print(__doc__)
        sys.exit(1)

    from src.reid_encoder import ReIDEncoder

    # Load encoder (optionally with Market-1501 pretrained weights)
    encoder = ReIDEncoder(device=args.device)
    if args.weights:
        print(f"Loading custom weights from {args.weights}")
        from torchreid.utils import load_pretrained_weights
        load_pretrained_weights(encoder._model, args.weights)
        print("Custom weights loaded.")

    print("\nLoading Market-1501 dataset…")
    query, gallery = load_market1501(args.data)
    print(f"Query: {len(query)} images  |  Gallery: {len(gallery)} images")

    print("\nExtracting query features…")
    t0 = time.perf_counter()
    q_embs, q_pids, q_cams = extract_features(query, encoder)
    t1 = time.perf_counter()
    query_speed = len(query) / (t1 - t0)

    print("Extracting gallery features…")
    g_embs, g_pids, g_cams = extract_features(gallery, encoder)
    t2 = time.perf_counter()

    print(f"\nFeature extraction: {query_speed:.0f} images/sec")

    print("Computing CMC and mAP…")
    cmc, m_ap = compute_cmc_map(q_embs, g_embs, q_pids, g_pids, q_cams, g_cams)

    weights_label = os.path.basename(args.weights) if args.weights else "ImageNet pretrained"
    print(f"\n{'='*55}")
    print(f"Model   : OSNet x0.25")
    print(f"Weights : {weights_label}")
    print(f"Dataset : Market-1501 (query={len(query)}, gallery={len(gallery)})")
    print(f"{'='*55}")
    print(f"Rank-1  : {cmc[0]*100:.2f}%")
    print(f"Rank-5  : {cmc[4]*100:.2f}%")
    print(f"Rank-10 : {cmc[9]*100:.2f}%")
    print(f"mAP     : {m_ap*100:.2f}%")
    print(f"{'='*55}")
    print(f"\nExtraction speed: {query_speed:.0f} crops/sec on {args.device}")


if __name__ == "__main__":
    main()
