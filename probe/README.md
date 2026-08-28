# Access Trap 72h probe

Executable spec: [`KILL.md`](KILL.md). Do not retune thresholds after seeing numbers.

This is **not** a paper repo. Dummy and 0.6B smoke runs never lock P1.

## Install

```bash
cd literature_review/icml-2027/probe
uv sync --extra dev
```

GPU full run also needs:

```bash
uv sync --extra dev --extra gpu
```

## Commands

```bash
# Pipeline only (no model). Used in CI.
uv run accesstrap probe --dummy

# Tiny real model on CPU/MPS/GPU (not a paper decision).
uv run accesstrap probe --smoke

# Frozen full probe (one A6000). Re-run the same command to resume.
uv run accesstrap probe \
  --model Qwen/Qwen3-8B \
  --n-samples 32 --n-math 80 --n-qa 80 \
  --out runs/full

# Wipe the sample log and start over (keeps items.json).
uv run accesstrap probe --out runs/full --fresh ...

uv run accesstrap verdict runs/<stamp>/summary.json
uv run pytest
```

Full probe writes `runs/full/` (stable path). Re-run `./scripts/run_a6000.sh` to resume.

Each sample is appended to `samples.jsonl` and fsynced. A crash loses at most the in-flight generation.

`--out` omitted still creates `runs/<timestamp>-{dummy,smoke,full}/`.

## Outputs

| File | Contents |
|---|---|
| `summary.json` | pass@k, Venn, entropy, verdict |
| `verdict.md` | human-readable LIVE/KILL |
| `per_item.json` | per-item c/n without raw logit dumps |
| `samples.jsonl` | generated text (large) |
| `items.json` | frozen item payloads |

## Live / kill

See `KILL.md`. All three flags must be true for LIVE:

1. Coverage Venn: `|A∖B| >= 2|B∖A|` and excess ≥ 5
2. Gold fork entropy strictly below no-access
3. Distractor entropy ≥ no-access
