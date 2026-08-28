# Decision log

Last updated: 2026-08-28.

**State: DIRECTION CHOSEN, CLAIM SHARPENED, NOT EXPERIMENT-LOCKED.**
P1 selected 2026-08-28. Novelty sweep finished: `docs/06-p1-novelty.md`.
Original P1 (mean accuracy / process quality across channels) is **closed** as a paper.
Sharpened P1 (solved-set Venn + fork-entropy degradation under **gold** access, dissociated from distractor entropy) is **open pending the 72h probe**.
Probe code: `probe/` (frozen `probe/KILL.md`). Dummy pipeline does not lock. Full run is `probe/scripts/run_a6000.sh`.
No GRPO, no paper repo until the A6000 probe is live.

## Options on the table

| ID | Name | 72h probe | Default if probe lives |
|---|---|---|---|
| P1 | Access Trap | yes | lock candidate #1 |
| P2 | Weights over traces | yes (no train) | lock candidate #2 |
| P3 | Language-channel scaling law | 1 week | backup |
| P4 | 80/20 for tool/memory tokens | no | only after P1 |

## Probe outcomes

| Date | Stem | Result (live / kill) | Evidence (path or one-sentence) |
|---|---|---|---|
| | | | |

## Lock

- Locked stem: P1 (direction only; experiment lock after probe)
- Date: 2026-08-28
- One-sentence claim (sharpened): External gold access is a Flexibility Trap — it shrinks the pass@k solved-item set by degrading entropy at internal forks; distractors move entropy the other way.
- Kill criterion already passed: original “access hurts accuracy” claim (TIM / Tool-Overuse / MemTrap / NoisyBench)
- New repo path (only after lock): —

## Anti-patterns

- Do not lock two stems.
- Do not retune a killed probe into the same claim.
- Do not open a paper repo until this file has a lock line.
- A killed stem gets a dated note here, or moves to `failed_ideas/`.
