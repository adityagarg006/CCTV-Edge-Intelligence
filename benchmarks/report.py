"""Benchmark result aggregation and report generation.

Collects results from all bench_*.run() calls and writes:
  - benchmark_report_<ts>.json  -- full structured results
  - benchmark_summary_<ts>.csv  -- tabular summary for spreadsheet analysis
  - benchmark_summary_<ts>.md   -- markdown table for GitHub README embedding

All values in the markdown table are runtime-computed from the results dict.
No numbers are hardcoded — the table always reflects the actual benchmark run.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logger = logging.getLogger(__name__)


def _safe_get(d: Any, *keys: str, default: Any = "N/A") -> Any:
    """Safely traverse nested dicts; return default if any key is missing."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def _fmt(value: Any, decimals: int = 2) -> str:
    """Format a numeric value to a fixed number of decimal places, or return as-is."""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def aggregate_results(
    results: dict[str, Any],
    output_dir: str = "reports",
) -> dict[str, str]:
    """Aggregate all benchmark results and write JSON, CSV, and Markdown reports.

    Args:
        results: Dict mapping component name to its bench_*.run() output.
            Expected keys (optional): 'detector', 'reid', 'feature_bank',
            'database', 'stream_reader', 'pipeline'.
        output_dir: Directory to write report files into.

    Returns:
        Dict mapping report type to absolute file path:
            {'json': ..., 'csv': ..., 'markdown': ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"benchmark_report_{ts}.json")
    csv_path = os.path.join(output_dir, f"benchmark_summary_{ts}.csv")
    md_path = os.path.join(output_dir, f"benchmark_summary_{ts}.md")

    # --- Write JSON (full results) ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("JSON report written: %s", json_path)

    # --- Build summary rows ---
    rows = _build_summary_rows(results)

    # --- Write CSV ---
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("CSV report written: %s", csv_path)
    else:
        # Write empty CSV with header only.
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("component,metric,value\n")

    # --- Write Markdown ---
    md_content = _build_markdown(results, rows, ts)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Markdown report written: %s", md_path)

    return {
        "json": os.path.abspath(json_path),
        "csv": os.path.abspath(csv_path),
        "markdown": os.path.abspath(md_path),
    }


def _build_summary_rows(results: dict[str, Any]) -> list[dict[str, str]]:
    """Extract key metrics from results into a flat list of row dicts."""
    rows: list[dict[str, str]] = []

    # --- Detector ---
    det = results.get("detector", {})
    if det and "error" not in det:
        rows.append({
            "component": "detector",
            "device": str(_safe_get(det, "device")),
            "metric": "cold_start_ms",
            "value": _fmt(_safe_get(det, "cold_start_ms", default=0.0)),
            "unit": "ms",
        })
        for pct in ("p50", "p95", "p99", "mean"):
            rows.append({
                "component": "detector",
                "device": str(_safe_get(det, "device")),
                "metric": f"latency_{pct}",
                "value": _fmt(_safe_get(det, "latency", pct, default=0.0)),
                "unit": "ms",
            })
        rows.append({
            "component": "detector",
            "device": str(_safe_get(det, "device")),
            "metric": "throughput_fps",
            "value": _fmt(_safe_get(det, "throughput_fps", default=0.0)),
            "unit": "fps",
        })

    # --- Re-ID encoder ---
    reid = results.get("reid", {})
    if reid and "error" not in reid:
        rows.append({
            "component": "reid_encoder",
            "device": str(_safe_get(reid, "device")),
            "metric": "cold_start_ms",
            "value": _fmt(_safe_get(reid, "cold_start_ms", default=0.0)),
            "unit": "ms",
        })
        for bs, bdata in _safe_get(reid, "batch_results", default={}).items():
            if not isinstance(bdata, dict):
                continue
            rows.append({
                "component": f"reid_encoder_batch{bs}",
                "device": str(_safe_get(reid, "device")),
                "metric": "latency_p95",
                "value": _fmt(_safe_get(bdata, "latency", "p95", default=0.0)),
                "unit": "ms",
            })
            rows.append({
                "component": f"reid_encoder_batch{bs}",
                "device": str(_safe_get(reid, "device")),
                "metric": "per_crop_p95_ms",
                "value": _fmt(_safe_get(bdata, "per_crop_ms", "p95", default=0.0)),
                "unit": "ms",
            })
            rows.append({
                "component": f"reid_encoder_batch{bs}",
                "device": str(_safe_get(reid, "device")),
                "metric": "throughput_crops_per_sec",
                "value": _fmt(_safe_get(bdata, "throughput_crops_per_sec", default=0.0)),
                "unit": "crops/s",
            })

    # --- Feature bank ---
    fb = results.get("feature_bank", {})
    if fb and "error" not in fb:
        for gs, gdata in _safe_get(fb, "gallery_results", default={}).items():
            if not isinstance(gdata, dict):
                continue
            rows.append({
                "component": f"feature_bank_g{gs}",
                "device": "cpu",
                "metric": "update_p95_us",
                "value": _fmt(_safe_get(gdata, "update_latency_us", "p95", default=0.0)),
                "unit": "us",
            })
            rows.append({
                "component": f"feature_bank_g{gs}",
                "device": "cpu",
                "metric": "query_p95_us",
                "value": _fmt(_safe_get(gdata, "query_latency_us", "p95", default=0.0)),
                "unit": "us",
            })
            rows.append({
                "component": f"feature_bank_g{gs}",
                "device": "cpu",
                "metric": "estimated_memory_kb",
                "value": _fmt(_safe_get(gdata, "estimated_memory_kb", default=0.0)),
                "unit": "KB",
            })

    # --- Database ---
    db = results.get("database", {})
    if db and "error" not in db:
        for r in _safe_get(db, "flush_interval_results", default=[]):
            if not isinstance(r, dict) or "error" in r:
                continue
            fi = r.get("flush_interval", "?")
            wal = r.get("wal", False)
            rows.append({
                "component": f"database_fi{fi}_wal{int(wal)}",
                "device": "cpu",
                "metric": "throughput_rows_per_sec",
                "value": _fmt(r.get("throughput_rows_per_sec", 0.0)),
                "unit": "rows/s",
            })
        wal_cmp = _safe_get(db, "wal_vs_no_wal", default={})
        if isinstance(wal_cmp, dict):
            rows.append({
                "component": "database_wal_speedup",
                "device": "cpu",
                "metric": "wal_speedup_factor",
                "value": _fmt(_safe_get(wal_cmp, "wal_speedup_factor", default=0.0)),
                "unit": "x",
            })
        conc = _safe_get(db, "concurrent_read_test", default={})
        if isinstance(conc, dict):
            rows.append({
                "component": "database_concurrent",
                "device": "cpu",
                "metric": "read_p95_ms",
                "value": _fmt(_safe_get(conc, "read_latency_ms", "p95", default=0.0)),
                "unit": "ms",
            })

    # --- Stream reader ---
    sr = results.get("stream_reader", {})
    if sr and "error" not in sr:
        best_fps = _safe_get(sr, "summary", "best_file_source_fps", default=0.0)
        rows.append({
            "component": "stream_reader",
            "device": "cpu",
            "metric": "best_file_source_fps",
            "value": _fmt(best_fps),
            "unit": "fps",
        })
        for dr in _safe_get(sr, "drop_oldest_results", default=[]):
            if not isinstance(dr, dict):
                continue
            qs = dr.get("queue_size", "?")
            dm = dr.get("consumer_delay_ms", "?")
            rows.append({
                "component": f"stream_drop_oldest_q{qs}_d{dm}ms",
                "device": "cpu",
                "metric": "drop_rate_pct",
                "value": _fmt(dr.get("drop_rate_pct", 0.0)),
                "unit": "%",
            })

    # --- Pipeline ---
    pl = results.get("pipeline", {})
    if pl and "error" not in pl:
        rows.append({
            "component": "pipeline",
            "device": str(_safe_get(pl, "device")),
            "metric": "throughput_fps",
            "value": _fmt(_safe_get(pl, "throughput_fps", default=0.0)),
            "unit": "fps",
        })
        rows.append({
            "component": "pipeline",
            "device": str(_safe_get(pl, "device")),
            "metric": "total_p95_ms",
            "value": _fmt(_safe_get(pl, "total_latency_ms", "p95", default=0.0)),
            "unit": "ms",
        })
        rows.append({
            "component": "pipeline",
            "device": str(_safe_get(pl, "device")),
            "metric": "sla_150ms_pct",
            "value": _fmt(_safe_get(pl, "sla_150ms_compliance_pct", default=0.0)),
            "unit": "%",
        })
        rows.append({
            "component": "pipeline",
            "device": str(_safe_get(pl, "device")),
            "metric": "sla_50ms_pct",
            "value": _fmt(_safe_get(pl, "sla_50ms_compliance_pct", default=0.0)),
            "unit": "%",
        })
        for stage_name, stage_stats in _safe_get(pl, "per_stage_latency_ms", default={}).items():
            if not isinstance(stage_stats, dict):
                continue
            rows.append({
                "component": f"pipeline_stage_{stage_name}",
                "device": str(_safe_get(pl, "device")),
                "metric": "p95_ms",
                "value": _fmt(stage_stats.get("p95", 0.0)),
                "unit": "ms",
            })

    return rows


def _build_markdown(
    results: dict[str, Any],
    rows: list[dict[str, str]],
    ts: str,
) -> str:
    """Build a markdown report string from runtime-computed results.

    All numeric values in the table are pulled from the results dict;
    no numbers are hardcoded here.
    """
    lines: list[str] = []
    lines.append(f"# Benchmark Report — {ts}")
    lines.append("")
    lines.append("Generated by `benchmarks/report.py`. All values are measured at runtime.")
    lines.append("")

    # --- High-level summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append("| Component | Metric | Value | Unit |")
    lines.append("|---|---|---|---|")

    det = results.get("detector", {})
    if det and "error" not in det:
        device = _safe_get(det, "device", default="?")
        lines.append(f"| Detector ({device}) | Cold start | {_fmt(_safe_get(det, 'cold_start_ms', default=0.0))} | ms |")
        lines.append(f"| Detector ({device}) | Latency p50 | {_fmt(_safe_get(det, 'latency', 'p50', default=0.0))} | ms |")
        lines.append(f"| Detector ({device}) | Latency p95 | {_fmt(_safe_get(det, 'latency', 'p95', default=0.0))} | ms |")
        lines.append(f"| Detector ({device}) | Throughput | {_fmt(_safe_get(det, 'throughput_fps', default=0.0))} | fps |")

    reid = results.get("reid", {})
    if reid and "error" not in reid:
        device = _safe_get(reid, "device", default="?")
        # Show best single-crop and batch=8 metrics.
        b1 = _safe_get(reid, "batch_results", "1", default={})
        b8 = _safe_get(reid, "batch_results", "8", default={})
        if isinstance(b1, dict):
            lines.append(f"| Re-ID encoder ({device}) | Batch=1 p95 | {_fmt(_safe_get(b1, 'latency', 'p95', default=0.0))} | ms |")
        if isinstance(b8, dict):
            lines.append(f"| Re-ID encoder ({device}) | Batch=8 p95 | {_fmt(_safe_get(b8, 'latency', 'p95', default=0.0))} | ms |")
            lines.append(f"| Re-ID encoder ({device}) | Batch=8 per-crop p95 | {_fmt(_safe_get(b8, 'per_crop_ms', 'p95', default=0.0))} | ms |")

    fb = results.get("feature_bank", {})
    if fb and "error" not in fb:
        for gs in ("100", "500", "1000"):
            gdata = _safe_get(fb, "gallery_results", gs, default={})
            if isinstance(gdata, dict):
                lines.append(f"| FeatureBank (g={gs}) | Query p95 | {_fmt(_safe_get(gdata, 'query_latency_us', 'p95', default=0.0))} | us |")
                lines.append(f"| FeatureBank (g={gs}) | Memory est. | {_fmt(_safe_get(gdata, 'estimated_memory_kb', default=0.0))} | KB |")

    db = results.get("database", {})
    if db and "error" not in db:
        wal_cmp = _safe_get(db, "wal_vs_no_wal", default={})
        if isinstance(wal_cmp, dict):
            lines.append(f"| Database | WAL throughput | {_fmt(_safe_get(wal_cmp, 'wal_throughput_rows_per_sec', default=0.0))} | rows/s |")
            lines.append(f"| Database | No-WAL throughput | {_fmt(_safe_get(wal_cmp, 'no_wal_throughput_rows_per_sec', default=0.0))} | rows/s |")
            lines.append(f"| Database | WAL speedup | {_fmt(_safe_get(wal_cmp, 'wal_speedup_factor', default=0.0))} | x |")

    pl = results.get("pipeline", {})
    if pl and "error" not in pl:
        device = _safe_get(pl, "device", default="?")
        lines.append(f"| Pipeline ({device}) | Throughput | {_fmt(_safe_get(pl, 'throughput_fps', default=0.0))} | fps |")
        lines.append(f"| Pipeline ({device}) | Total p95 | {_fmt(_safe_get(pl, 'total_latency_ms', 'p95', default=0.0))} | ms |")
        lines.append(f"| Pipeline ({device}) | SLA ≤150ms | {_fmt(_safe_get(pl, 'sla_150ms_compliance_pct', default=0.0))} | % |")
        lines.append(f"| Pipeline ({device}) | SLA ≤50ms | {_fmt(_safe_get(pl, 'sla_50ms_compliance_pct', default=0.0))} | % |")

    lines.append("")

    # --- Detector latency distribution ---
    if det and "error" not in det:
        device = _safe_get(det, "device", default="?")
        lat = _safe_get(det, "latency", default={})
        if isinstance(lat, dict):
            lines.append("## Detector Latency Distribution")
            lines.append("")
            lines.append(f"Device: `{device}` | Frame skip: `{_safe_get(det, 'frame_skip', default=1)}`")
            lines.append("")
            lines.append("| Percentile | Latency (ms) |")
            lines.append("|---|---|")
            for pct in ("p50", "p90", "p95", "p99", "mean", "std"):
                lines.append(f"| {pct} | {_fmt(lat.get(pct, 0.0))} |")
            lines.append("")

    # --- Re-ID batch sweep ---
    if reid and "error" not in reid:
        device = _safe_get(reid, "device", default="?")
        batch_results = _safe_get(reid, "batch_results", default={})
        if isinstance(batch_results, dict) and batch_results:
            lines.append("## Re-ID Encoder Batch Sweep")
            lines.append("")
            lines.append(f"Device: `{device}`")
            lines.append("")
            lines.append("| Batch Size | p95 (ms) | Per-Crop p95 (ms) | Throughput (crops/s) | Norm OK |")
            lines.append("|---|---|---|---|---|")
            for bs in sorted(batch_results.keys(), key=lambda x: int(x)):
                bdata = batch_results[bs]
                if not isinstance(bdata, dict):
                    continue
                lines.append(
                    f"| {bs} "
                    f"| {_fmt(_safe_get(bdata, 'latency', 'p95', default=0.0))} "
                    f"| {_fmt(_safe_get(bdata, 'per_crop_ms', 'p95', default=0.0))} "
                    f"| {_fmt(_safe_get(bdata, 'throughput_crops_per_sec', default=0.0))} "
                    f"| {_safe_get(bdata, 'norm_check_passed', default='?')} |"
                )
            lines.append("")

    # --- Feature bank gallery sweep ---
    if fb and "error" not in fb:
        gallery_results = _safe_get(fb, "gallery_results", default={})
        if isinstance(gallery_results, dict) and gallery_results:
            embed_dim = _safe_get(fb, "embed_dim", default="?")
            lines.append("## FeatureBank Gallery Sweep")
            lines.append("")
            lines.append(f"Embed dim: `{embed_dim}`")
            lines.append("")
            lines.append("| Gallery Size | Update p95 (us) | Query p95 (us) | Memory (KB) |")
            lines.append("|---|---|---|---|")
            for gs in sorted(gallery_results.keys(), key=lambda x: int(x)):
                gdata = gallery_results[gs]
                if not isinstance(gdata, dict):
                    continue
                lines.append(
                    f"| {gs} "
                    f"| {_fmt(_safe_get(gdata, 'update_latency_us', 'p95', default=0.0))} "
                    f"| {_fmt(_safe_get(gdata, 'query_latency_us', 'p95', default=0.0))} "
                    f"| {_fmt(_safe_get(gdata, 'estimated_memory_kb', default=0.0))} |"
                )
            lines.append("")

    # --- Database flush interval sweep ---
    if db and "error" not in db:
        flush_results = _safe_get(db, "flush_interval_results", default=[])
        if isinstance(flush_results, list) and flush_results:
            lines.append("## Database Flush Interval Sweep")
            lines.append("")
            lines.append("| Flush Interval | WAL | Throughput (rows/s) | Total Time (s) |")
            lines.append("|---|---|---|---|")
            for r in flush_results:
                if not isinstance(r, dict) or "error" in r:
                    continue
                lines.append(
                    f"| {r.get('flush_interval', '?')} "
                    f"| {r.get('wal', '?')} "
                    f"| {_fmt(r.get('throughput_rows_per_sec', 0.0))} "
                    f"| {_fmt(r.get('total_time_s', 0.0), decimals=4)} |"
                )
            lines.append("")

    # --- Pipeline stage breakdown ---
    if pl and "error" not in pl:
        device = _safe_get(pl, "device", default="?")
        per_stage = _safe_get(pl, "per_stage_latency_ms", default={})
        if isinstance(per_stage, dict) and per_stage:
            lines.append("## Pipeline Per-Stage Latency")
            lines.append("")
            lines.append(f"Device: `{device}` | Frame skip: `{_safe_get(pl, 'frame_skip', default=1)}` | Frames: `{_safe_get(pl, 'n_frames', default=0)}`")
            lines.append("")
            lines.append("| Stage | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |")
            lines.append("|---|---|---|---|---|")
            stage_order = ["detection", "crop_extract", "reid_encode", "faiss_query", "db_write"]
            for stage in stage_order:
                sdata = per_stage.get(stage, {})
                if not isinstance(sdata, dict):
                    continue
                lines.append(
                    f"| {stage} "
                    f"| {_fmt(sdata.get('p50', 0.0))} "
                    f"| {_fmt(sdata.get('p95', 0.0))} "
                    f"| {_fmt(sdata.get('p99', 0.0))} "
                    f"| {_fmt(sdata.get('mean', 0.0))} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated: {ts}*")
    lines.append("")

    return "\n".join(lines)
