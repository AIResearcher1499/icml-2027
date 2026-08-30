# Decision log

Last updated: 2026-08-30.

**State: P1 KILL. P2 KILL. P3 PARKED. NO PAPER LOCK. RESCAN 2026-08-30.**
P1 selected 2026-08-28. Novelty sweep: `docs/06-p1-novelty.md`.
Original P1 was already **closed** as a paper. Sharpened P1 is **killed** by the merged A6000 probe (`docs/p1-probe-result-2026-08-30.md`). Do not retune.

P2 novelty sweep finished 2026-08-29 (`docs/07-p2-novelty.md`). Original P2 is closed as a paper. Sharpened P2 is **killed** by the merged A6000 probe (`docs/p2-probe-result-2026-08-30.md`). Capability failed (`acc(think,F) < acc(base,F)`). Do not retune 0.05 / 0.10 / 0.08.

P3 novelty sweep finished 2026-08-30 (`docs/08-p3-novelty.md`). Original P3 is closed. Sharpened P3 is **parked** (not locked, not the next GPU job): tiny GPT-2 is a weak NLP fit, and the pretrained transfer plot is taken by arXiv:2608.09942. Dummy `probe-p3/` never locks. Do not run P3 GPU.

Rescan: `docs/09-rescan-2026-08-30.md` (R1–R3, poster-shaped). ICML-level: `docs/10-icml-level-stems-2026-08-30.md`. **S1 novelty sweep 2026-08-30:** `docs/11-s1-novelty.md`. Sharpened S1 open pending `probe/KILL-s1.md`. Not locked. Dummy never locks. Do not retune S1 after a NO-GO. S2 not started.

No GRPO, no paper repo until this file records a lock on a live stem. Do not lock P3 here from dummy.

## Options on the table

| ID | Name | 72h probe | Default if probe lives |
|---|---|---|---|
| P1 | Access Trap | yes | lock candidate #1 |
| P2 | Weights over traces | yes (no train) | lock candidate #2 |
| P3 | Language-channel necessity | yes (dummy; tiny GPT-2 later) | backup if P2 dies |
| P4 | 80/20 for tool/memory tokens | no | only after P1 |

## Probe outcomes

| Date | Stem | Result (live / kill) | Evidence (path or one-sentence) |
|---|---|---|---|
| 2026-08-30 | P1 Access Trap (sharpened) | **KILL** | `docs/p1-probe-result-2026-08-30.md` — merged `runs/full/`: Venn A−B=B−A=0, `mean_ent(B)>mean_ent(A)`, C dissociation passed. Access expands coverage; P1 dies. |
| 2026-08-30 | P2 Weights over traces (sharpened) | **KILL** | `docs/p2-probe-result-2026-08-30.md` — merged `runs/p2/`: think F 0.633 < base F 0.853 (capability fail). Empty-gap and not-JET passed; N near floor on both arms. |

## Lock

- Locked stem: **none** (P1/P2 killed 2026-08-30; P3 parked; R1–R3 not locked)
- Date: —
- One-sentence claim (sharpened P1, **dead**): External gold access is a Flexibility Trap — it shrinks the pass@k solved-item set by degrading entropy at internal forks; distractors move entropy the other way.
- One-sentence claim (sharpened P2, **dead**): After RL, empty-trace accuracy holds while base still needs the trace, not as prefix-only early-exit.
- Kill criterion already passed: original P1; sharpened P1 probe; original P2; sharpened P2 probe (`docs/p2-probe-result-2026-08-30.md`)
- New repo path (only after lock): —

## Anti-patterns

- Do not lock two stems.
- Do not retune a killed probe into the same claim.
- Do not open a paper repo until this file has a lock line.
- A killed stem gets a dated note here, or moves to `failed_ideas/`.
