"""
Pipeline throughput benchmark.

Measures per-stage and end-to-end latency/FPS on a given video.

Run:
  python benchmarks/eval_throughput.py --source path/to/video.mp4
  python benchmarks/eval_throughput.py --source path/to/video.mp4 --inference-width 640
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def percentile(data: list[float], p: int) -> float:
    idx = max(0, int(len(data) * p / 100) - 1)
    return sorted(data)[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-width", type=int, default=0, help="Resize before inference. 0=original")
    parser.add_argument("--n-frames", type=int, default=200, help="Frames to benchmark")
    args = parser.parse_args()

    from src.detector import PersonDetector
    from src.reid_encoder import ReIDEncoder
    from src.feature_bank import FeatureBank
    from src.utils import extract_crops

    print(f"\n{'='*60}")
    print(f"Hardware : {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory//1024**3}GB VRAM)")
    print(f"CUDA     : {torch.version.cuda}")
    print(f"Source   : {args.source}")
    print(f"{'='*60}\n")

    det = PersonDetector(device=args.device)
    enc = ReIDEncoder(device=args.device)
    bank = FeatureBank(embed_dim=enc.embed_dim)

    cap = cv2.VideoCapture(args.source)
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while len(frames) < args.n_frames:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    print(f"Source FPS     : {fps_src:.1f}")
    print(f"Resolution     : {w}x{h}")
    print(f"Test frames    : {len(frames)}")
    print(f"Re-ID model    : OSNet x0.25 ({enc.embed_dim}-dim)")
    print(f"Inference width: {'original' if args.inference_width == 0 else args.inference_width}px\n")

    # Optionally resize
    if args.inference_width > 0:
        inf_frames = [
            cv2.resize(f, (args.inference_width, int(h * args.inference_width / w)))
            if w > args.inference_width else f
            for f in frames
        ]
    else:
        inf_frames = frames

    # Warm-up
    for f in inf_frames[:5]:
        det.detect(f)

    # --- Stage 1: Detection ---
    yolo_times = []
    all_dets = []
    for f in inf_frames:
        t = time.perf_counter()
        d = det.detect(f)
        yolo_times.append((time.perf_counter() - t) * 1000)
        all_dets.append(d)

    # --- Stage 2: Crop extraction + Re-ID encoding ---
    reid_times = []
    crops_per_frame = []
    all_embs = []
    for orig_f, dets in zip(frames, all_dets):
        valid, crops = extract_crops(orig_f, dets)
        crops_per_frame.append(len(crops))
        if not crops:
            all_embs.append([])
            continue
        t = time.perf_counter()
        embs = enc.encode_batch(crops)
        reid_times.append((time.perf_counter() - t) * 1000)
        all_embs.append(embs)

    # --- Stage 3: FAISS queries ---
    faiss_times = []
    for embs in all_embs:
        if len(embs) == 0:
            continue
        t = time.perf_counter()
        for emb in embs:
            bank.query(emb, k=5)
        faiss_times.append((time.perf_counter() - t) * 1000)

    # --- End-to-end ---
    e2e_times = []
    for orig_f, inf_f in zip(frames, inf_frames):
        t = time.perf_counter()
        dets = det.detect(inf_f)
        valid, crops = extract_crops(orig_f, dets)
        if crops:
            embs = enc.encode_batch(crops)
            for emb in embs:
                bank.query(emb, k=5)
        e2e_times.append((time.perf_counter() - t) * 1000)

    print(f"{'='*60}")
    print(f"{'Stage':<30} {'avg':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
    print(f"{'-'*60}")

    def row(label, data):
        avg = sum(data) / len(data)
        p50 = percentile(data, 50)
        p95 = percentile(data, 95)
        p99 = percentile(data, 99)
        print(f"  {label:<28} {avg:>7.1f} {p50:>7.1f} {p95:>7.1f} {p99:>7.1f}  ms")

    row(f"YOLOv8n detect ({w}x{h})", yolo_times)
    avg_crops = sum(crops_per_frame) / len(crops_per_frame)
    if reid_times:
        row(f"OSNet encode (avg {avg_crops:.1f} crops)", reid_times)
    if faiss_times:
        row("FAISS query", faiss_times)
    row("End-to-end", e2e_times)

    avg_e2e = sum(e2e_times) / len(e2e_times)
    achieved_fps = 1000 / avg_e2e
    print(f"{'='*60}")
    print(f"  Achieved FPS      : {achieved_fps:.1f} FPS")
    print(f"  Source FPS        : {fps_src:.1f} FPS")
    print(f"  Real-time capable : {'YES' if achieved_fps >= fps_src else f'NO (need {fps_src/achieved_fps:.1f}x speedup)'}")
    print(f"  Avg detections/frame: {avg_crops:.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
