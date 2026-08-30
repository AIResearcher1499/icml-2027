# P3 72h probe (serial-depth prosthesis)

Executable spec: [`../probe/KILL-p3.md`](../probe/KILL-p3.md).
Novelty: [`../docs/08-p3-novelty.md`](../docs/08-p3-novelty.md).

This package does **not** modify P1 `accesstrap` or P2 `weighttraces`.
Dummy never locks P3. Full train is not implemented here (no 20M–300M, no 0.6B, no GPU).

Do not run a GPU job on the 2×A6000 pair while P2 is live.

## Install

```bash
cd literature_review/icml-2027/probe-p3
uv sync --extra dev
```

Mac / CI (no train):

```bash
uv run scratchdepth probe --dummy
uv run pytest
```

Without `--dummy` the CLI exits 2 and does not train.

## Live / kill

See `KILL-p3.md`. All three flags must be true for LIVE, on **L=2** only:

1. Serial no-CoT collapse: `drop_h(serial, L=2, direct) >= 0.40`
2. CoT restores that cell: `restore(serial, L=2, h=16) >= 0.30`
3. Parallel control does not collapse: `drop_h(parallel, L=2, direct) <= 0.10`

Dummy always writes `DUMMY_SKIP`. Do not retune `0.40` / `0.30` / `0.10` after seeing a number.
