"""Benchmark runner CLI for CCTV-Edge-Intelligence.

Usage:
    python benchmarks/run.py [--device cpu|cuda] [--output-dir DIR]
                              [--component COMP] [--frame-skip N]
                              [--skip-video-gen]

Components: detector, reid, feature_bank, database, stream_reader, pipeline, all

Progress is printed after each component in the format:
    [check] bench_X  p95=X.Xms  fps=X.X  (X.Xs)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logger = logging.getLogger(__name__)

_ALL_COMPONENTS = ["detector", "reid", "feature_bank", "database", "stream_reader", "pipeline"]


def _print_progress(name: str, result: dict[str, Any], elapsed_s: float) -> None:
    """Print a one-line progress summary after each component run.

    Format: [✓] bench_X  p95=X.Xms  fps=X.X  (X.Xs)
    """
    p95 = "N/A"
    fps = "N/A"

    if "error" in result:
        status = "[!]"
        detail = f"ERROR: {result['error']}"
        print(f"{status} bench_{name:<14}  {detail}  ({elapsed_s:.1f}s)", flush=True)
        return

    status = "[+]"

    # Extract p95 and fps depending on component type.
    if name == "detector":
        lat = result.get("latency", {})
        p95_val = lat.get("p95", None)
        fps_val = result.get("throughput_fps", None)
        if p95_val is not None:
            p95 = f"{p95_val:.1f}ms"
        if fps_val is not None:
            fps = f"{fps_val:.1f}"

    elif name == "reid":
        # Use batch=1 p95.
        b1 = result.get("batch_results", {}).get("1", {})
        p95_val = b1.get("latency", {}).get("p95", None)
        b8 = result.get("batch_results", {}).get("8", {})
        fps_val = b8.get("throughput_crops_per_sec", None)
        if p95_val is not None:
            p95 = f"{p95_val:.1f}ms"
        if fps_val is not None:
            fps = f"{fps_val:.1f} crops/s"

    elif name == "feature_bank":
        # Use gallery=500 query p95.
        g500 = result.get("gallery_results", {}).get("500", {})
        p95_val = g500.get("query_latency_us", {}).get("p95", None)
        if p95_val is not None:
            p95 = f"{p95_val:.1f}us"
        fps = "N/A"

    elif name == "database":
        # Use WAL flush_interval=30 throughput.
        wal_cmp = result.get("wal_vs_no_wal", {})
        fps_val = wal_cmp.get("wal_throughput_rows_per_sec", None)
        if fps_val is not None:
            fps = f"{fps_val:.0f} rows/s"
        p95_val = result.get("concurrent_read_test", {}).get("read_latency_ms", {}).get("p95", None)
        if p95_val is not None:
            p95 = f"{p95_val:.2f}ms"

    elif name == "stream_reader":
        fps_val = result.get("summary", {}).get("best_file_source_fps", None)
        if fps_val is not None:
            fps = f"{fps_val:.1f}"
        p95 = "N/A"

    elif name == "pipeline":
        lat = result.get("total_latency_ms", {})
        p95_val = lat.get("p95", None)
        fps_val = result.get("throughput_fps", None)
        if p95_val is not None:
            p95 = f"{p95_val:.1f}ms"
        if fps_val is not None:
            fps = f"{fps_val:.1f}"

    print(f"{status} bench_{name:<14}  p95={p95:<10}  fps={fps:<12}  ({elapsed_s:.1f}s)", flush=True)


def main() -> None:
    """Parse arguments, run selected benchmarks, and write aggregated reports."""
    parser = argparse.ArgumentParser(
        description="CCTV-Edge-Intelligence benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/run.py                          # all components, cpu
  python benchmarks/run.py --device cuda            # all, GPU
  python benchmarks/run.py --component detector     # single component
  python benchmarks/run.py --component pipeline --device cuda --frame-skip 1
        """,
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for report files (default: reports)",
    )
    parser.add_argument(
        "--component",
        default="all",
        choices=_ALL_COMPONENTS + ["all"],
        help="Which component to benchmark (default: all)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Frame skip for detector and pipeline (default: 1)",
    )
    parser.add_argument(
        "--skip-video-gen",
        action="store_true",
        help="Skip synthetic video generation (stream_reader file tests skipped)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    components = _ALL_COMPONENTS if args.component == "all" else [args.component]

    print(f"CCTV-Edge-Intelligence Benchmarks")
    print(f"Device: {args.device}  |  Components: {', '.join(components)}")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")
    print("-" * 70, flush=True)

    all_results: dict[str, Any] = {}

    for component in components:
        t_start = time.perf_counter()
        result: dict[str, Any] = {}

        try:
            if component == "detector":
                from benchmarks.bench_detector import run as bench_run
                result = bench_run(device=args.device, frame_skip=args.frame_skip)

            elif component == "reid":
                from benchmarks.bench_reid import run as bench_run
                result = bench_run(device=args.device)

            elif component == "feature_bank":
                from benchmarks.bench_feature_bank import run as bench_run
                result = bench_run()

            elif component == "database":
                from benchmarks.bench_database import run as bench_run
                result = bench_run()

            elif component == "stream_reader":
                from benchmarks.bench_stream_reader import run as bench_run
                result = bench_run(skip_video_gen=args.skip_video_gen)

            elif component == "pipeline":
                from benchmarks.bench_pipeline import run as bench_run
                result = bench_run(device=args.device, frame_skip=args.frame_skip)

        except Exception as exc:
            logger.exception("Benchmark '%s' raised an unhandled exception: %s", component, exc)
            result = {"error": str(exc)}

        elapsed_s = time.perf_counter() - t_start
        all_results[component] = result
        _print_progress(component, result, elapsed_s)

    print("-" * 70, flush=True)

    # --- Aggregate and write reports ---
    from benchmarks.report import aggregate_results

    report_paths = aggregate_results(all_results, output_dir=args.output_dir)

    print("\nReports written:")
    for report_type, path in report_paths.items():
        print(f"  {report_type:8s}: {path}")

    # Print quick JSON to stdout for CI capture.
    print("\n--- Quick JSON Summary ---")
    quick: dict[str, Any] = {}
    for comp, res in all_results.items():
        if "error" in res:
            quick[comp] = {"error": res["error"]}
        elif comp == "detector":
            quick[comp] = {
                "p95_ms": res.get("latency", {}).get("p95"),
                "fps": res.get("throughput_fps"),
            }
        elif comp == "reid":
            quick[comp] = {
                "batch1_p95_ms": res.get("batch_results", {}).get("1", {}).get("latency", {}).get("p95"),
                "batch8_p95_ms": res.get("batch_results", {}).get("8", {}).get("latency", {}).get("p95"),
            }
        elif comp == "feature_bank":
            quick[comp] = {
                "g500_query_p95_us": res.get("gallery_results", {}).get("500", {}).get("query_latency_us", {}).get("p95"),
            }
        elif comp == "database":
            quick[comp] = {
                "wal_speedup": res.get("wal_vs_no_wal", {}).get("wal_speedup_factor"),
            }
        elif comp == "pipeline":
            quick[comp] = {
                "fps": res.get("throughput_fps"),
                "total_p95_ms": res.get("total_latency_ms", {}).get("p95"),
                "sla_150ms_pct": res.get("sla_150ms_compliance_pct"),
            }
    print(json.dumps(quick, indent=2))


if __name__ == "__main__":
    main()
