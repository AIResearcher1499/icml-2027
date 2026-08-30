#!/usr/bin/env bash
# Preferred S1 full probe: freeze 80 GSM8K items, one process per A6000, merge.
# Re-run this script to resume both shards. Do not --fresh unless you mean it.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra gpu --extra dev

uv run committrap init-items --out runs/s1 --n-items 80 --seed 0

COMMON=(
  --model Qwen/Qwen3-8B
  --n-samples 8
  --n-items 80
  --seed 0
  --temperature 0.7
  --items runs/s1/items.json
)

CUDA_VISIBLE_DEVICES=0 uv run committrap probe "${COMMON[@]}" \
  --shard 0/2 --out runs/s1-s0 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 uv run committrap probe "${COMMON[@]}" \
  --shard 1/2 --out runs/s1-s1 &
PID1=$!

fail=0
wait "$PID0" || fail=1
wait "$PID1" || fail=1
if [ "$fail" -ne 0 ]; then
  echo "a shard failed; re-run this script to resume" >&2
  exit 1
fi

uv run committrap merge runs/s1-s0 runs/s1-s1 --out runs/s1 --n-samples 8 --n-items 80
echo "verdict: runs/s1/verdict.md"
