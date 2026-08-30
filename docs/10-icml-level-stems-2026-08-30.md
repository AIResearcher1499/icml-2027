# ICML-level stems (not Flexibility Trap clones)

Date: 2026-08-30.
R1–R3 in `docs/09-rescan-2026-08-30.md` are **not** this list. They reuse P1’s Venn protocol on a new channel. That is poster-shaped, not Outstanding-shaped.

Bar for this file: an assumption the **ML** field is using in 2026; a quantity whose **sign** can flip; an intervention on a frozen 8B (2×A6000); NLP is the object, not the audience. Method only after the finding. Deep-sweep before GPU. Not locked.

## Why not another Venn

Flexibility Trap took *order*. P1 failed *access*. P2 failed *empty-trace*. A 2027 paper that is “Venn of condition A vs B on Qwen3-8B GSM8K” reads as the same measurement with a new label. ICML already rewarded the template once.

## S1 — Compute after commitment has the wrong sign

**Assumption the field uses.** Extra test-time tokens help, and they help *most* when the model is still uncertain. Early-exit papers (DEER, Dynasor, CUSUM) treat the low-entropy **Confidence Region** as “done, and probably correct.”

**Claim.** CoT has two phases (Xu et al., arXiv:2606.02020). After the change-point, sequential continuation is a **commitment**, not exploration. On traces that are already wrong at the change-point, matched-budget compute spent *continuing* does not recover (or locks in); the same budget spent *re-forking at the change-point* does. The Confidence Region is reliable **conditional on being correct**; when wrong, it is a trap.

**Quantity (sign flip).** Let `t*` be a frozen change-point detector (CUSUM on token entropy, pre-registered). On items wrong at `t*`:

`Δ_continue = acc(full CoT) − acc(truncate at t*)`  
`Δ_refork = acc(resample k prefixes from t*) − acc(truncate at t*)`

Prediction: `Δ_refork > Δ_continue` at matched extra tokens. Optionally ECE rises with tokens after `t*` on the wrong set (calibration drift), while it falls before `t*`.

**Intervention.** Not a new architecture. Truncate vs continue vs resample-from-`t*` on the **same** F traces.

**Why this is ICML-shaped.** Test-time compute is the ML object. The finding is *where* the extra token has positive vs negative return, not “check your work” or “MCQ vs generate.” NLP is how the phase is written (the CoT).

**Not a method paper in week 1.** CUSUM-weighted voting and early exit are Xu et al. We do not ship a better halt rule until the sign flip exists.

**Kill-threats (must full-text before GPU).**

| Paper | Why it might already be this |
|---|---|
| Xu et al., 2606.02020 | Two-phase entropy; CUSUM early-exit **assuming** confidence ≈ correct. If they already resample at `t*`, we are them. |
| CDUR, 2606.11211 | Longer CoT *budget* → overconfidence (Hypothesis Lock-In). 47 trap items. If their B-sweep is the same as phase-continue, we are a detector swap. |
| von Recum et al., 2602.07470 | CoT interventions at **fixed timesteps**. Early hurts more. If time ≈ phase, incremental. |
| JET, 2509.23392 | Prefix-50% enough. Different: they keep the prefix; we **branch** at `t*`. |
| CES / ETR | Train entropy shaping. We are inference, frozen policy. |

**72h probe:** frozen in `probe/KILL-s1.md` after the 2026-08-30 sweep (`docs/11-s1-novelty.md`). Sharpened: wrong-at-`t*` lock-in + refork gap **beats** JET 50% cut. Dummy never locks.

**Compute.** Inference, 2×A6000, similar to P2 (think CoTs). No GRPO in the probe.

## S2 — High-entropy tokens are a mixture: forks then overthink

**Assumption the field uses.** 80/20 (NeurIPS 2025): the top ~20% **highest-entropy** CoT tokens carry the RLVR gradient. Implementations treat that set as *one* population of forks.

**Claim.** High-entropy tokens are two populations on the same trace: **early = forks** (ablating them flips the answer), **late = overthink triggers** (ablating them *recovers* or shortens without flipping). The 80/20 recipe on the full trace trains the wrong tail.

**Quantity (sign flip).** On the same frozen traces, causal Δ of masking the top-20% entropy tokens in the **prefix** (before first answer-span peak / before `t*`) vs in the **suffix**. Prediction: prefix mask hurts acc; suffix mask does not (or helps). Opposite signs on the same statistic.

**Intervention.** Token mask / replace-with-`...` on a frozen 8B. No train in the probe.

**Why this is ICML-shaped.** Directly amends a NeurIPS 2025 + ICML 2026 Outstanding *programme* (entropy forks), instead of cloning its Venn. NLP object: the tokens. ML object: what RL should train.

**Not GRPO in week 1.** Optional later: GRPO only on prefix-forks. That is the method consequence.

**Kill-threats.**

| Paper | Why it might already be this |
|---|---|
| 80/20, 2506.01939 | If they already split early vs late high-H, we are them. |
| Lotfi et al., 2606.00206 | Overthinking markers at high-H positions — **quantized** models. Do **not** salvage `commit-qat`. S2 is BF16, mask split, not INT4 commit_lag. |
| CES, 2605.19358 | Penalize high-H on *correct* paths, reward on *incorrect*. Training, not prefix/suffix causal split. |
| ETR, 2604.05355 | Entropy *trend* reward. Trajectory, not token population. |
| Feng et al., 2509.19284 | Failed-step fraction > length. Different axis. |

**72h probe.** Qwen3-8B thinking, 80 GSM8K, n=8. Log token entropy (no P1 connective list — that list is frozen for P1 only). Prefix = first 50% of CoT **or** pre-`t*` if S1’s detector is frozen; pre-register **one**. LIVE: `acc(suffix-mask) − acc(prefix-mask) ≥ 0.08` and prefix-mask < full. KILL: same sign, or only length changes.

## Explicitly not ICML-level (for this lab, now)

- R1 Check Trap / R2 Choice Trap / R3 strategy coverage (`docs/09-rescan`)
- P3 GPT-2 / 2608.09942 rerun
- Semantic calibration broken by CoT (Nakkiran et al., 2511.04869 — taken)
- Weak-model prefixes for RLVR exploration (2608.27420, 27 Aug 2026 — taken as method)
- Native vs English CoT (Layer Swap, Lingua Franca, cot-lang failed)
- Another verifier ROC / BoN paper (ROC-n-reroll, LLM-as-a-Verifier)

## Probe order

1. Full-text S1 kill-threats (Xu, CDUR, von Recum, JET). If S1 is a detector-swap on CDUR, **drop S1**.
2. If S1 dies, S2 sweep (80/20, Lotfi, CES, ETR).
3. GPU only after a `KILL-s*.md` freeze. Dummy never locks.

Do not lock in `docs/05-decision.md` from this file.
