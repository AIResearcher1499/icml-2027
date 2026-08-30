# P2 probe result — KILL

Date: 2026-08-30.
Spec: frozen `probe/KILL-p2.md` (2026-08-29; runner + post-merge analysis amendments 2026-08-30).
Run: merged pool `probe-p2/runs/p2/` after
`weighttraces merge runs/p2-s0 runs/p2-s1 --out runs/p2 --n-samples 8 --n-items 80`.
Source of the numbers below: `runs/p2/verdict.md` on the A6000 box (transcribed 2026-08-30). Canonical if a copy of `runs/p2/summary.json` is later checked in.

**Do not retune 0.05 / 0.10 / 0.08. Do not swap Distill-8B. Do not raise max_new_tokens. Do not drop items.**

## Decision

**KILL.** Capability failed. Empty-trace gap and not-JET passed and do not save the stem.

| Flag | Frozen test | Result |
|---|---|---|
| capability_live | `acc(think,F) - acc(base,F) >= 0.05` | False (`0.633 - 0.853 = -0.220`) |
| empty_gap_live | `drop_N(base) - drop_N(think) >= 0.10` | True (`0.845 - 0.613 = 0.233`) |
| not_jet_live | `drop_N(think) - drop_P(think) <= 0.08` | True (`0.613 - 0.556 = 0.056`) |
| jet_replica | `drop_P(think) < 0.05` and `drop_N(think) >= drop_N(base) - 0.05` | False |

Kill reason (from `verdict.md`):

- think is not better on full CoT by >= 0.05 (capability)

Matched kill line in `KILL-p2.md`: *(1) fails: think is not actually better on full CoT.*

n_items=80, n_samples=8. This is the frozen pool, not a shard.

## Numbers (merged)

Mean F CoT tokens (log only, not a gate): base `285.0`, think `1511.7`.

| arm | acc F | acc P | acc N | drop_P | drop_N |
|---|---|---|---|---|---|
| base | 0.853 | 0.052 | 0.008 | 0.802 | 0.845 |
| think | 0.633 | 0.077 | 0.020 | 0.556 | 0.613 |

Empty CoT (N) is near floor on **both** arms (base 0.8%, think 2.0%). The empty-gap flag passes mostly because think F is already 22 pp worse than base F, so there is less room to drop — not because think answers correctly without a trace.

P also collapses on both arms (base 5.2%, think 7.7%). This is not a JET replica (prefix-robust + empty-collapse). Prefix 50% does not preserve answers.

## What this kills

Sharpened P2: after RL, empty-trace accuracy holds while the SFT/base checkpoint still needs the trace, and that is not prefix-only early-exit.

The 72h proxy was same-checkpoint Qwen3-8B think vs base. On this protocol think is **worse** on full CoT than base, and neither arm survives empty-trace. The capability control exists exactly so a lower think F cannot be read as “compute moved into weights.”

## Explicitly not next

- Do not raise F `max_new_tokens` above 2048 after seeing mean think length 1512.
- Do not swap in Distill-8B / another checkpoint.
- Do not drop GSM8K items or cut a solvability band.
- Do not treat empty_gap / not_jet passing as a stay-of-execution.
- No GRPO, no paper repo, no P2-v2 on the same claim.

A new claim needs a new prereg in a new repo. This stem is closed.

## Next stem

P3 (sharpened, backup) remains open pending `probe/KILL-p3.md`. Dummy never locks. Not locked in `docs/05-decision.md`.
