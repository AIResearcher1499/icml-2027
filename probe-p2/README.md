# P2 72h probe (weights over traces)

Executable spec: [`../probe/KILL-p2.md`](../probe/KILL-p2.md).
Novelty: [`../docs/07-p2-novelty.md`](../docs/07-p2-novelty.md).

This package does **not** modify P1 `accesstrap`. Dummy and smoke never lock P2.
LIVE/KILL is only the merged 80-item, both-arm pool. A shard writes `SHARD_SKIP`.

P1 math is finished; the 2×A6000 pair may run this job.

## Install

```bash
cd literature_review/icml-2027/probe-p2
uv sync --extra gpu --extra dev
```

Mac / CI (no model):

```bash
uv sync --extra dev
uv run weighttraces probe --dummy
uv run pytest
```

## Full run (2×A6000)

From `probe-p2/` on the GPU box:

```bash
chmod +x scripts/run_a6000_2gpu.sh
./scripts/run_a6000_2gpu.sh
```

That freezes `runs/p2/items.json` (80 GSM8K, seed 0), runs shard `0/2` on GPU0 and `1/2` on GPU1 (both arms), then merges.

Resume: re-run the same script. It skips finished `(item, arm, condition, sample_idx)`. Do not pass `--fresh`.

One GPU:

```bash
./scripts/run_a6000.sh
```

Progress:

```bash
cat runs/p2-s0/progress.json
cat runs/p2-s1/progress.json
```

Paper verdict (only this file):

```bash
cat runs/p2/verdict.md
```

Do not retune `0.05` / `0.10` / `0.08` after seeing a number. Dummy/smoke never lock.
