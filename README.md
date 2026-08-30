# ICML 2027 topic scan

**Status:** P1 **KILL**, P2 **KILL**, P3 **PARKED** (2026-08-30). No paper lock. Rescan: `docs/09-rescan-2026-08-30.md`.

This directory is the working notebook for a *new* ICML 2027 paper.
It is not a paper repo. Do not train, preregister, or open a stem repo
until `docs/05-decision.md` records a lock.

Scan date: 2026-08-28. P1 novelty sweep: 2026-08-28 (`docs/06-p1-novelty.md`).
P2 novelty sweep: 2026-08-29 (`docs/07-p2-novelty.md`). Hedge only; not locked.
P3 novelty sweep: 2026-08-30 (`docs/08-p3-novelty.md`). Backup only; not locked.

Constraints: NLP background, 2× NVIDIA A6000 (48 GB each), target **ICML 2027**.
ACL 2027 is fallback, not the design target.

**P1 is closed.** Merged probe (`docs/p1-probe-result-2026-08-30.md`): gold access
did not shrink the solved set and did not drop fork entropy. Do not retune.

**P2 is closed.** Merged probe (`docs/p2-probe-result-2026-08-30.md`): think
was worse than base on full CoT; empty-trace collapsed on both arms. Do not retune.

**P3 is parked.** Tiny GPT-2 is a weak NLP fit; pretrained serial-depth CoT is
taken by arXiv:2608.09942. Dummy `probe-p3/` never locks. Do not GPU P3.

**Rescan (not locked):** Check Trap / Choice Trap / strategy coverage.
See `docs/09-rescan-2026-08-30.md`.

## Read in this order

| File | What it is |
|---|---|
| [docs/00-constraints.md](docs/00-constraints.md) | Compute, background, what “good” means |
| [docs/01-venue-and-calendar.md](docs/01-venue-and-calendar.md) | Why ICML, dates, dual-submission |
| [docs/02-icml-template.md](docs/02-icml-template.md) | What ICML 2026 actually rewarded |
| [docs/03-closed-claims.md](docs/03-closed-claims.md) | Claims already taken — do not redo |
| [docs/04-proposals.md](docs/04-proposals.md) | Four candidate papers (P1–P4) |
| [docs/05-decision.md](docs/05-decision.md) | Decision log |
| [docs/06-p1-novelty.md](docs/06-p1-novelty.md) | P1 related work, sharpened claim, 72h probe |
| [docs/07-p2-novelty.md](docs/07-p2-novelty.md) | P2 related work, sharpened claim, 72h probe (hedge) |
| [docs/08-p3-novelty.md](docs/08-p3-novelty.md) | P3 related work, sharpened claim, 72h probe (backup) |
| [probe/](probe/) | Frozen P1 (`KILL.md`). P2: `KILL-p2.md`. P3: `KILL-p3.md`. S1: `KILL-s1.md` |
| [probe-p2/](probe-p2/) | P2 dummy probe (Mac). Does not modify P1 `accesstrap` |
| [probe-p3/](probe-p3/) | P3 dummy probe (Mac). Does not modify P1/P2 packages |
| [probe-s1/](probe-s1/) | S1 commitment-trap probe. Dummy Mac; 2×A6000 merge |

## Shortlist

1. **P1 Access Trap — KILL.** Result: `docs/p1-probe-result-2026-08-30.md`. Do not retune. P4 stays dead with P1.
2. **P2 Weights over traces — KILL.** Result: `docs/p2-probe-result-2026-08-30.md`. Do not retune.
3. **P3 Language-channel necessity — PARKED.** `docs/08-p3-novelty.md`. Do not GPU.
4. **Rescan** — `docs/09-rescan-2026-08-30.md` (R1–R3, not ICML-bar).
5. **ICML-level candidates** — `docs/10-icml-level-stems-2026-08-30.md`. S1 / S2. Not locked.

Next action: S1 probe in `probe-s1/` (`KILL-s1.md`). Mac dummy: `./scripts/run_mac.sh`. A6000: `./scripts/run_a6000_2gpu.sh`. Dummy never locks. Do not open a paper repo until `docs/05-decision.md` records a lock.

## Rules for this folder

- English in all files here.
- Do not copy numbers or verdicts into the portfolio-root `Claude.md`.
- A `docs/prereg-*.md` is frozen once a data file exists. There is no prereg yet.
- Do not retune after a NO-GO. A killed probe becomes a dated note in this folder, or the stem is buried under `failed_ideas/`.
