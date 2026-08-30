# S1 72h probe (commitment trap)

Executable spec: [`../probe/KILL-s1.md`](../probe/KILL-s1.md).
Novelty: [`../docs/11-s1-novelty.md`](../docs/11-s1-novelty.md).

Dummy and smoke never lock. LIVE/KILL is only the merged 80-item pool (`gsm-016`…`gsm-079` live). A shard writes `SHARD_SKIP`.

Does **not** modify P1 `accesstrap` or P2 `weighttraces`.

## Mac

```bash
cd literature_review/icml-2027/probe-s1
./scripts/run_mac.sh
```

Dummy + pytest + **8B bench** (`--bench`: 1 item × 1 sample, **same model and generate flags as A6000**). Prints `tok/s`. Always `DUMMY_SKIP` / `SMOKE_SKIP`.

Full 80×8 on Mac (same CLI as one A6000; days, may OOM):

```bash
uv run committrap probe --model Qwen/Qwen3-8B --n-samples 8 --n-items 80 --out runs/s1
```

The 72h probe is **inference only**. No GRPO / SFT.

## 2×A6000

```bash
chmod +x scripts/run_a6000_2gpu.sh
./scripts/run_a6000_2gpu.sh
```

Resume: re-run the same script. Do not `--fresh`.

```bash
cat runs/s1-s0/progress.json
cat runs/s1-s1/progress.json
# after both shards complete:
# merge is at the end of the 2gpu script
cat runs/s1/verdict.md
```

Do not retune `40` / `0.50` / `0.05` / `0.10` / `0.05` after seeing a number.
