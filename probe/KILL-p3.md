# Frozen 72h probe criteria (P3)

Frozen: 2026-08-30. Do not edit thresholds after seeing a number.
Amendments go below as dated notes, never as silent edits.

Source: `docs/08-p3-novelty.md`. This file is the executable spec.
Dummy code: `literature_review/icml-2027/probe-p3/` (does not modify P1 `accesstrap` or P2 `weighttraces`).
Dummy never locks P3. Do not lock P3 in `docs/05-decision.md` from this probe.
Do not train 20M–300M. Do not eval 0.6B. Do not run this job on the 2×A6000 pair while P2 is live.

This file does not change P1 `KILL.md` or P2 `KILL-p2.md` (0.05 / 0.10 / 0.08 stay).

## Protocol (frozen)

- Architecture: GPT-2-rotary, `d_model=256`, 4 heads, MLP 4×. Width frozen so **depth** is the axis.
- Depths *L* ∈ {2, 4, 8}
- Train from scratch. Token vocab = `{0..16, *, +, =, SEP, ANS, EOS}` plus scratchpad digits. No pretrained weights.
- Serial task (live): iterated modular **product** `a_1 * a_2 * … * a_h (mod 17)`, each `a_i` uniform in `1..16`
- Parallel task (live): iterated modular **sum** `a_1 + a_2 + … + a_h (mod 17)`, same *h* and *p*
- Horizons *h* ∈ {4, 8, 16}
- Copy-with-offset: log-only if run; it does **not** enter the live test
- Conditions per (task, *L*, *h*):
  - `direct`: train and eval to emit `ANS <gold>` with no intermediate tokens
  - `cot`: train and eval a forced scratchpad of the **running state** (partial product or partial sum after each operand), then `ANS <gold>`
- Train budget (not a threshold): 30k steps, batch 128, seq ≤ 96, AdamW 3e-4, seed 0. Do not train longer to grok a KILL into a LIVE.
- Eval: greedy decode, exact match on the residue after `ANS`. One generation per item.
- Item pool (full): 256 eval items per (task, *h*), seed 0. Dummy: 4 items per (task, *h*).
- Primary metric pool: all eval items of that (task, *h*). Do not drop “easy/hard” items after seeing a number.

## Metrics (frozen)

Let `acc(task, L, h, cond)` be exact-match accuracy on the eval pool.

- `drop_h(task, L, cond) = acc(task, L, 4, cond) - acc(task, L, 16, cond)`
- `restore(task, L, h) = acc(task, L, h, cot) - acc(task, L, h, direct)`

Log (not live gates): acc at *h*=8; acc at *L* ∈ {4, 8}; mean scratchpad length; copy-with-offset if run.

Do not compute P1 Venn, P1 connective entropy, or P2 `drop_N` / `drop_P` in this probe.

## Live (all three required)

Live numbers are computed on **L=2** only (the one-pass depth budget). *L* ∈ {4, 8} is a log.

1. Serial collapse: `drop_h(serial, L=2, direct) >= 0.40`
2. CoT prosthesis: `restore(serial, L=2, h=16) >= 0.30`
3. Not-length: `drop_h(parallel, L=2, direct) <= 0.10`

## Kill (any one)

- (1) fails: no-CoT never breaks as *h* grows on the serial arm → cannot reproduce Li/Feng; P3 dies
- (2) fails: CoT does not restore the serial cell → not a language-channel finding
- (3) fails: parallel collapses too (`drop_h(parallel, L=2, direct) > 0.10`) → length / Faith-and-Fate / RASP-L replica
- Only *L*=8 no-CoT recovers *h*=16 while (1)–(3) fail → Physics of LM depth replica, not a verbal-channel paper
- Dummy never decides live/kill for the paper

## GPU (not this Mac; not the P2 pair while P2 is running)

- Tiny GPT-2, bf16, one 48 GB GPU is more than enough. Do not tensor-parallel. Do not load Qwen.
- Grid: 3 depths × 2 conditions × 2 tasks = 12 trains. Copy-with-offset is extra and log-only.
- Budget sketch (not a threshold): ~15–40 min/job on an A6000 → **~0.5–1 GPU-day** with retries. Not a 20M–300M sweep.
- Dummy on Mac first. Dummy never trains.

## Amendments

- 2026-08-30: Dummy runner lives in `probe-p3/` (`uv run scratchdepth probe --dummy`). Dummy still never locks. No 0.6B smoke. Not a threshold change.
