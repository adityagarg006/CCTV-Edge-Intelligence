#!/usr/bin/env bash
# run_benchmarks.sh — run the full CCTV-Edge-Intelligence benchmark suite.
#
# Usage:
#   bash scripts/run_benchmarks.sh [--device cpu|cuda] [--output-dir DIR]
#                                   [--component all|detector|reid|...]
#                                   [--frame-skip N] [--skip-video-gen]
#
# All arguments are forwarded verbatim to benchmarks/run.py.
# The script exits with a non-zero status if any benchmark fails.
#
# Examples:
#   bash scripts/run_benchmarks.sh
#   bash scripts/run_benchmarks.sh --device cuda
#   bash scripts/run_benchmarks.sh --component detector --frame-skip 2
#   bash scripts/run_benchmarks.sh --skip-video-gen --output-dir /tmp/bench

set -euo pipefail

# Resolve the project root (one level above the scripts/ directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo "  CCTV-Edge-Intelligence Benchmark Suite"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Python:       $(python --version 2>&1)"
echo "  Date:         $(date)"
echo "============================================================"

# Change to project root so relative imports (src/, config/) work.
cd "${PROJECT_ROOT}"

# Default output directory.
OUTPUT_DIR="${PROJECT_ROOT}/reports"

# Forward all CLI arguments to the Python runner.
python benchmarks/run.py \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

EXIT_CODE=$?

echo "============================================================"
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "  Benchmark suite completed successfully."
else
    echo "  Benchmark suite exited with code ${EXIT_CODE}."
fi
echo "  Reports written to: ${OUTPUT_DIR}"
echo "============================================================"

exit "${EXIT_CODE}"
