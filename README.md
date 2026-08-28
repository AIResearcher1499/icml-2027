# ICML 2027 topic scan

**Status:** P1 chosen; claim sharpened after novelty sweep. No experiments yet.

This directory is the working notebook for a *new* ICML 2027 paper.
It is not a paper repo. Do not train, preregister, or open a stem repo
until `docs/05-decision.md` records a lock.

Scan date: 2026-08-28. P1 novelty sweep: 2026-08-28 (`docs/06-p1-novelty.md`).

Constraints: NLP background, 2× NVIDIA A6000 (48 GB each), target **ICML 2027**.
ACL 2027 is fallback, not the design target.

**Current claim (sharpened P1):** gold external access is a Flexibility Trap
(solved-set coverage + fork-entropy degradation), dissociated from the
distractor/confusion regime. Original “access hurts accuracy” is closed.

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
| [probe/](probe/) | Frozen 72h probe code (`KILL.md`) |

## Shortlist

1. **P1 Access Trap (active)** — gold access as Flexibility Trap: solved-set Venn + fork-entropy drop, vs distractor entropy rise. See `docs/06-p1-novelty.md`.
2. **P2 Weights over traces** — parked unless P1 probe dies.
3. **P3 / P4** — not started.

Next action: run the frozen probe in `probe/` (`KILL.md`). Dummy CI: `uv run accesstrap probe --dummy`. Full: `probe/scripts/run_a6000.sh`.

## Rules for this folder

- English in all files here.
- Do not copy numbers or verdicts into the portfolio-root `Claude.md`.
- A `docs/prereg-*.md` is frozen once a data file exists. There is no prereg yet.
- Do not retune after a NO-GO. A killed probe becomes a dated note in this folder, or the stem is buried under `failed_ideas/`.
