# Decision log

Last updated: 2026-08-30.

**State: P1 PROBE KILL. NO PAPER LOCK. HEDGE P2 STILL OPEN.**
P1 selected 2026-08-28. Novelty sweep: `docs/06-p1-novelty.md`.
Original P1 (mean accuracy / process quality across channels) was already **closed** as a paper.
Sharpened P1 (solved-set Venn + fork-entropy degradation under **gold** access) is **killed** by the merged A6000 probe. Result: `docs/p1-probe-result-2026-08-30.md`. Do not retune.

P2 novelty sweep finished 2026-08-29 (`docs/07-p2-novelty.md`). Original P2 is closed as a paper. Sharpened P2 is the remaining hedge, **not locked**. Frozen spec: `probe/KILL-p2.md`. GPU runner: `probe-p2/` (`scripts/run_a6000_2gpu.sh`). Dummy never locks. Do not retune P2 after a NO-GO.

No GRPO, no paper repo until this file records a lock on a live stem.

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
| 2026-08-30 | P1 Access Trap (sharpened) | **KILL** | `docs/p1-probe-result-2026-08-30.md` — merged `runs/full/`: Venn A−B=B−A=0, `mean_ent(B)>mean_ent(A)`, C dissociation passed. Access expands coverage; P1 dies. |

## Lock

- Locked stem: **none** (P1 direction killed 2026-08-30; P2 hedge not locked)
- Date: —
- One-sentence claim (sharpened P1, **dead**): External gold access is a Flexibility Trap — it shrinks the pass@k solved-item set by degrading entropy at internal forks; distractors move entropy the other way.
- Kill criterion already passed: original “access hurts accuracy”; **and** sharpened coverage+entropy probe (`docs/p1-probe-result-2026-08-30.md`)
- New repo path (only after lock): —

## Anti-patterns

- Do not lock two stems.
- Do not retune a killed probe into the same claim.
- Do not open a paper repo until this file has a lock line.
- A killed stem gets a dated note here, or moves to `failed_ideas/`.
