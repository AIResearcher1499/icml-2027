#!/usr/bin/env bash
# Full frozen probe on one A6000 (48 GB). Do not retune flags after seeing numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra gpu --extra dev
# Stable --out so a crash can resume: re-run this script, it skips finished keys.
uv run accesstrap probe \
  --model Qwen/Qwen3-8B \
  --n-samples 32 \
  --n-math 80 \
  --n-qa 80 \
  --seed 0 \
  --temperature 0.7 \
  --max-new-tokens 512 \
  --out runs/full
