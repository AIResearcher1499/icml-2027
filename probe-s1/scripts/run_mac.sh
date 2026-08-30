#!/usr/bin/env bash
# Mac: dummy, then one 8B sample with the SAME generate flags as A6000 (thinking, 2048, probes).
# That bench never locks. Full 80x8 paper run on Mac: omit --bench (days, may OOM).
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra gpu --extra dev

echo "dummy — DUMMY_SKIP"
uv run committrap probe --dummy --out runs/s1-dummy
uv run pytest -q

echo "8B bench — 1 item x 1 sample, Qwen/Qwen3-8B, same temp/max tokens as A6000; SMOKE_SKIP"
uv run committrap probe --bench --model Qwen/Qwen3-8B --out runs/s1-bench
echo "tok/s is in the generate lines above. Full paper run (not recommended on Mac):"
echo "  uv run committrap probe --model Qwen/Qwen3-8B --n-samples 8 --n-items 80 --out runs/s1"
