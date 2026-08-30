#!/usr/bin/env bash
# Single-GPU fallback. Prefer scripts/run_a6000_2gpu.sh when both cards are free.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra gpu --extra dev
uv run committrap probe \
  --model Qwen/Qwen3-8B \
  --n-samples 8 \
  --n-items 80 \
  --seed 0 \
  --temperature 0.7 \
  --out runs/s1
echo "verdict: runs/s1/verdict.md"
