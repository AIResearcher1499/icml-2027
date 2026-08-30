# P1 probe result — KILL

Date: 2026-08-30.
Spec: frozen `probe/KILL.md` (2026-08-28; merge amendment 2026-08-29).
Run: merged pool `runs/full/` after `accesstrap merge runs/full-math runs/full-qa --out runs/full --n-samples 32`.
Source of the numbers below: `runs/full/verdict.md` on the A6000 box (transcribed 2026-08-30). Canonical if a copy of `runs/full/summary.json` is later checked in.

**Do not retune thresholds. Do not drop Hotpot. Do not re-score at a different k.**

## Decision

**KILL.** Two of three LIVE flags failed. The third (distractor dissociation) passed and does not save the stem.

| Flag | Frozen test | Result |
|---|---|---|
| coverage_live | `\|A∖B\| >= 2\|B∖A\|` and excess `>= 5` | False (`0`, `0`, excess `0`) |
| gold_entropy_drop | `mean_ent(B) < mean_ent(A)` | False (`0.382 > 0.370`) |
| distractor_dissociation | `mean_ent(C) >= mean_ent(A)` | True (`0.430 >= 0.370`) |

Kill reasons (from `verdict.md`):

- coverage Venn fails (`|A-B|>=2|B-A|` and excess`>=5`); possible Tool-Overuse replica or access expands coverage
- no gold fork-entropy drop (`B < A` failed)

Matched kill line in `KILL.md`: *Venn has B ⊇ A or excess < 5, and no gold entropy drop → access expands coverage; P1 dies.*

This is **not** a Tool-Overuse replica: gold access **raises** mean pass@k on the primary band, it does not drop it. Closest extra pattern: distractors hurt accuracy and raise entropy (NoisyBench-shaped), while gold helps accuracy and does not degrade forks.

## Numbers (merged primary band)

- entropy rule: `lexical` (no fallback)
- primary-band items: **111 / 160** (band is large enough; excess `>= 5` was achievable)
- Venn A–B: `A_minus_B=0`, `B_minus_A=0`, `A_and_B=111`, `neither=0`
- entropy (nats): A `0.36996`, B `0.38160`, C `0.42977`
- mean pass primary:

| k | A (no-access) | B (gold) | C (distractor) |
|---|---|---|---|
| 1 | 0.870 | 0.924 | 0.742 |
| 8 | 0.987 | 0.990 | 0.877 |
| 32 | 1.000 | 1.000 | 0.910 |

At k=n=32 the primary-band solved sets of A and B are identical (perfect overlap). Gold access raises pass@1 by ~5.4 pp. Distractors lower pass@1 by ~12.8 pp and leave pass@32 at 0.91.

QA-only `runs/full-qa/` was KILL with a different entropy pattern. That shard is **not** the paper verdict (`KILL.md` 2026-08-29). It is not used to salvage or retune this result.

## What this kills

Sharpened P1: gold external access is a Flexibility Trap (solved-set shrinkage + fork-entropy drop under gold, dissociated from distractors).

On this frozen protocol the gold channel does not shrink the solved set and does not drop connective entropy. The assumption that access is false flexibility fails in the same direction as “access expands coverage.”

## Explicitly not next

- Do not recompute Venn at k=1 or k=8 after seeing k=32 saturate.
- Do not drop Hotpot and keep GSM8K.
- Do not change the connective list, primary-band 0.5, n=32, or entropy inequalities.
- No GRPO, no paper repo, no P4 (P4 was gated on P1 living).

A new claim needs a new prereg in a new repo. This stem is closed.

## Next stem

P2 (sharpened, hedge) remains open pending `probe/KILL-p2.md`. Not locked.
