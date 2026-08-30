#!/usr/bin/env bash
# Preferred P2 full probe: freeze 80 GSM8K items, one process per A6000 (item split), merge.
# Re-run this script to resume both shards. Do not --fresh unless you mean it.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra gpu --extra dev

uv run weighttraces init-items --out runs/p2 --n-items 80 --seed 0

COMMON=(
  --model Qwen/Qwen3-8B
  --n-samples 8
  --seed 0
  --temperature 0.7
  --arm both
  --items runs/p2/items.json
)

CUDA_VISIBLE_DEVICES=0 uv run weighttraces probe "${COMMON[@]}" \
  --shard 0/2 --out runs/p2-s0 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 uv run weighttraces probe "${COMMON[@]}" \
  --shard 1/2 --out runs/p2-s1 &
PID1=$!

fail=0
wait "$PID0" || fail=1
wait "$PID1" || fail=1
if [ "$fail" -ne 0 ]; then
  echo "a shard failed; re-run this script to resume" >&2
  exit 1
fi

uv run weighttraces merge runs/p2-s0 runs/p2-s1 --out runs/p2 --n-samples 8 --n-items 80
echo "verdict: runs/p2/verdict.md"
