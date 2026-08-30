# Frozen 72h probe criteria (P2)

Frozen: 2026-08-29. Do not edit thresholds after seeing a number.
Amendments go below as dated notes, never as silent edits.

Source: `docs/07-p2-novelty.md`. This file is the executable spec.
Dummy code: `literature_review/icml-2027/probe-p2/` (does not modify P1 `accesstrap`).
Dummy/smoke never lock P2. Do not lock P2 in `docs/05-decision.md` from this probe.

P1 `runs/full/` is merged (KILL 2026-08-30). The 2×A6000 pair may run this job.
No tensor-parallel. One process per GPU (item split 0/2 | 1/2), then `weighttraces merge`.

This file does not change P1 `KILL.md`, the P1 connective list, the P1 primary band, or P1 Venn thresholds.

## Protocol (frozen)

- Models (full): `Qwen/Qwen3-8B`
  - `think`: `enable_thinking=True`
  - `base`: `enable_thinking=False`, user suffix: `Solve step by step. Put the final numeric answer after ####.`
- Smoke model: `Qwen/Qwen3-0.6B`, same template flags
- Temperature 0.7, top_p 1.0, seed 0
- F `max_new_tokens` 2048; P/N answer `max_new_tokens` 64
- Full: n=8 samples/item; smoke: n=2; dummy: n=4
- Candidate pool: 80 GSM8K test items (fixed shuffle, seed 0)
- Primary metric pool: all 80 items. Do not retune a solvability band after seeing a number.
- MATH-500 (or harder) is log-only if run; it does not enter the live test.
- For each model, sample n completions under F (full CoT).
- For each F sample, prefill interventions (same sample, not a new independent draw):
  - P: keep the first 50% of **that sample’s** CoT tokens (integer `floor(0.5 * n_cot_tokens)`; if n_cot_tokens=0, P equals N), close the think/step block, generate the answer
  - N: empty CoT (Qwen3: assistant prefix `<think>\n</think>\n`; base: no steps, go to `####`), generate the answer
- Scoring: GSM8K numeric match after `####` (or last number in the answer span). Same matcher for all conditions.
- CoT span: `think` = tokens inside `<think>…</think>`; `base` = tokens before `####`. Length is this span, not the answer.

## Metrics (frozen)

- `acc(model, cond)` = mean sample correctness over the 80 items × n samples (avg@n)
- `drop_P(model) = acc(model, F) - acc(model, P)`
- `drop_N(model) = acc(model, F) - acc(model, N)`
- Unbiased pass@k for k in {1, 8} ∩ {k: k ≤ n}, logged **only** as an Invisible Leash control. Not a live gate.
- Mean CoT token length under F, per model, logged (length confound). Not a live gate.

Do not compute P1 Venn, P1 connective entropy, or CIR’s per-token JS average in this probe.

## Live (all three required)

Let `dN_b = drop_N(base)`, `dN_t = drop_N(think)`, `dP_t = drop_P(think)`.

1. Capability: `acc(think, F) - acc(base, F) >= 0.05`
2. Empty-trace gap: `dN_b - dN_t >= 0.10`
3. Not-JET: `dN_t - dP_t <= 0.08`

## Kill (any one)

- (3) fails and (2) is driven by prefix-only: `dP_t < 0.05` and `dN_t >= dN_b - 0.05` → JET replica
- (2) fails: think is not more empty-trace-robust than base → original “moved into weights” is weak
- (1) fails: think is not actually better on full CoT
- Only pass@1 up / pass@8 down, (1)–(3) fail → Limit-of-RLVR / Invisible Leash replica
- Smoke (0.6B) / dummy never decides live/kill for the paper

## GPU (not this Mac, not the P1 pair)

- One 48 GB GPU is enough (8B bf16 + 2k generate). Do not tensor-parallel.
- Two models sequential, or two GPUs **on another machine** (one model each).
- Budget sketch (not a threshold): 80 items × 8 F samples × 2 models, then 2 prefills each. Think CoTs are long; plan ~0.5–1 GPU-day. Dummy on Mac first.

## Amendments

- 2026-08-30: Full runner lives in `probe-p2/` (`uv run weighttraces probe`). Dummy still never locks. Two-GPU split is items 0/2 | 1/2 then merge; verdict only on the merged 80-item both-arm pool. P1 A6000s may be used (P1 merged). Not a threshold change.
- 2026-08-30 (post-merge analysis freeze; written **before** seeing GPU numbers). Not a threshold change. Live/Kill numbers above stay `0.05` / `0.10` / `0.08`. n=8, 80 GSM8K, think vs base proxy stay. Do not retune after a number.
  - **Verdict lock surface.** LIVE/KILL is only `runs/p2/` after `weighttraces merge` of a complete pool: 80 items × 2 arms (`think`, `base`) × 3 conditions (F/P/N) × n=8. A shard writes `SHARD_SKIP`. Dummy writes `DUMMY_SKIP`. Smoke writes `SMOKE_SKIP`. Incomplete pool writes `INCOMPLETE`. None of those lock. Do not peek shard `samples.jsonl` or per-item dumps to pre-decide.
  - **No checkpoint swap after seeing a number.** Do not replace think/base with Distill-8B, a different Qwen3 checkpoint, or a GRPO checkpoint after seeing `drop_N` / `drop_P`. Distill-8B remains a logged third arm only if it was already in the frozen protocol (it is not). The paper, if LIVE, trains a matched SFT→GRPO curve later; that is not a probe retune.
  - **No pool shrink after seeing a number.** Primary metric pool is all 80 GSM8K items (fixed shuffle, seed 0). Do not drop easy/hard items, do not cut a solvability band, do not reweight, after seeing `drop_N` or `acc(F)`. MATH-500 stays log-only and does not enter the live test.
  - **pass@k cannot rescue KILL.** Unbiased pass@k (k ∈ {1,8} ∩ {k≤n}) is an Invisible Leash / Limit-of-RLVR **control**. A merge KILL stays KILL if pass@1 rises or pass@8 falls. Coverage shrink is not a live gate and is not a stay-of-execution.
  - **Mean CoT length cannot rescue KILL.** Mean F CoT token length per arm is a length-confound **log**. A merge KILL stays KILL if think CoTs are longer, shorter, or 50% of think is still many tokens. Length is not a live gate.
  - **If merge KILL:** one line in `docs/05-decision.md`. No P2-v2 on the same claim (empty-trace robustness of think vs base on verbal CoT; prefix vs empty JET split). Do not retune 0.05 / 0.10 / 0.08. A new stem needs a new prereg in a new repo, or the direction is buried under `failed_ideas/`.
  - **If merge LIVE:** only then inspect per-item tables, length confound, and pass@k. Those are paper-section material, not a second live test. Do not lock P2 in `docs/05-decision.md` from dummy/smoke.
- 2026-08-30: Merged full probe recorded **KILL**. Numbers: `docs/p2-probe-result-2026-08-30.md`. Not a threshold change. Do not re-score.
