# ICML 2027 topic scan

**Status:** P1 sharpened probe **KILL** (2026-08-30). No paper lock. P2 hedge remains open.

This directory is the working notebook for a *new* ICML 2027 paper.
It is not a paper repo. Do not train, preregister, or open a stem repo
until `docs/05-decision.md` records a lock.

Scan date: 2026-08-28. P1 novelty sweep: 2026-08-28 (`docs/06-p1-novelty.md`).
P2 novelty sweep: 2026-08-29 (`docs/07-p2-novelty.md`). Hedge only; not locked.

Constraints: NLP background, 2× NVIDIA A6000 (48 GB each), target **ICML 2027**.
ACL 2027 is fallback, not the design target.

**P1 is closed.** Merged probe (`docs/p1-probe-result-2026-08-30.md`): gold access
did not shrink the solved set and did not drop fork entropy. Do not retune.

**Current hedge (sharpened P2, not locked):** on verbal CoT, a matched SFT→GRPO
path relocates computation (empty-trace robustness after RL, not prefix-only
early-exit). See `docs/07-p2-novelty.md`.

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
| [probe/](probe/) | Frozen P1 72h probe (`KILL.md`). P2 spec: `KILL-p2.md` |
| [probe-p2/](probe-p2/) | P2 dummy probe (Mac). Does not modify P1 `accesstrap` |

## Shortlist

1. **P1 Access Trap — KILL.** Result: `docs/p1-probe-result-2026-08-30.md`. Do not retune. P4 was gated on P1 living; it is not next.
2. **P2 Weights over traces (hedge, not locked)** — original wording closed; sharpened claim open pending `probe/KILL-p2.md`. Dummy: `uv run weighttraces probe --dummy` in `probe-p2/`.
3. **P3** — backup if P2 also dies. Not started.

Next action: P2 frozen GPU probe in `probe-p2/` (`KILL-p2.md`). Dummy never locks. Do not open a paper repo until `docs/05-decision.md` records a lock.

## Rules for this folder

- English in all files here.
- Do not copy numbers or verdicts into the portfolio-root `Claude.md`.
- A `docs/prereg-*.md` is frozen once a data file exists. There is no prereg yet.
- Do not retune after a NO-GO. A killed probe becomes a dated note in this folder, or the stem is buried under `failed_ideas/`.
