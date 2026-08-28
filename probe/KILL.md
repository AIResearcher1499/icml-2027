# Frozen 72h probe criteria

Frozen: 2026-08-28. Do not edit thresholds after seeing a number.
Amendments go below as dated notes, never as silent edits.

Source: `docs/06-p1-novelty.md`. This file is the executable spec.

## Protocol (frozen)

- Model (full): `Qwen/Qwen3-8B`, `enable_thinking=False`
- Smoke model: `Qwen/Qwen3-0.6B`, same template flags
- Temperature 0.7, top_p 1.0, max_new_tokens 512, seed 0
- Full: n=32 samples/item; smoke: n=4
- Candidate pool: 80 GSM8K items with ≥2 `<<expr=val>>` annotations; 80 HotpotQA distractor-validation items
- Primary band: items with no-access pass@min(8,n) ≥ 0.5. Do not retune 0.5.
- Conditions per item: A no-access, B gold-access, C distractor-access
- Gold math: all calculator identities except the last (no final-answer leak)
- Gold QA: gold supporting sentences only
- Distractor math: numeric values in those identities perturbed
- Distractor QA: non-supporting paragraphs from the *same* Hotpot context

## Metrics (frozen)

- Unbiased pass@k for k in {1, 8, 32} ∩ {k: k ≤ n}
- Solved set at k=n: item in the set iff at least one of n samples is correct
- Venn on the primary band: |A∖B|, |B∖A|, |A∩B|
- Lexical-connective entropy: mean next-token entropy (nats) at positions whose decoded token matches the frozen list, measured on the *internal* span (not quoted gold/tool lines)
- Fallback (pre-registered): if a condition has <20 lexical hits globally, use mean entropy of the per-sequence top-20% highest-entropy tokens instead. Record which rule fired. Do not mix rules across conditions in the live test — if any condition falls back, all three use the fallback.

Frozen connective list (lowercase, stripped):

therefore, thus, hence, since, because, so, wait, however, instead,
alternatively, actually, but, first, then, let's, hmm

## Live (all three required)

1. Coverage: `|A∖B| >= 2 * |B∖A|` and `|A∖B| - |B∖A| >= 5`
2. Gold entropy drop: `mean_ent(B) < mean_ent(A)` (strict)
3. Distractor dissociation: `mean_ent(C) >= mean_ent(A)`

## Kill (any one)

- Only mean pass@k / avg@k drops on B, Venn fails (1) → Tool-Overuse replica
- Only C hurts and entropy rises, (1) or (2) fail → NoisyBench replica
- Venn has B ⊇ A or excess < 5, and no gold entropy drop → access expands coverage; P1 dies
- Smoke (0.6B / dummy) never decides live/kill for the paper

## Amendments

- 2026-08-28: Hub id is `Qwen/Qwen3-8B` (post-trained; Qwen3 dropped the `-Instruct` suffix). Same checkpoint, thinking off. Not a threshold change.
