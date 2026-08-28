#!/usr/bin/env bash
# Preferred full probe: one process per A6000, then merge verdict.
# Re-run this script to resume both shards.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra gpu --extra dev

COMMON=(
  --model Qwen/Qwen3-8B
  --n-samples 32
  --seed 0
  --temperature 0.7
  --max-new-tokens 512
)

CUDA_VISIBLE_DEVICES=0 uv run accesstrap probe "${COMMON[@]}" \
  --n-math 80 --n-qa 0 --out runs/full-math &
PID_MATH=$!
CUDA_VISIBLE_DEVICES=1 uv run accesstrap probe "${COMMON[@]}" \
  --n-math 0 --n-qa 80 --out runs/full-qa &
PID_QA=$!

fail=0
wait "$PID_MATH" || fail=1
wait "$PID_QA" || fail=1
if [ "$fail" -ne 0 ]; then
  echo "a shard failed; re-run this script to resume" >&2
  exit 1
fi

uv run accesstrap merge runs/full-math runs/full-qa --out runs/full --n-samples 32
echo "verdict: runs/full/verdict.md"
