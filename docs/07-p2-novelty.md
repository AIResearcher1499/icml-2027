# P2 novelty sweep (Weights over traces after RL)

Sweep date: 2026-08-29.
**Probe KILL 2026-08-30** — `docs/p2-probe-result-2026-08-30.md`. This file is the design note, not the result. Do not retune the live/kill lines below.

Hedge stem. **Original P2 is too close to 2026 prior work.** The sharpened claim was open pending the 72h probe and is now **closed by that probe**.

This sweep is full-text on the kill-threats listed in `docs/04-proposals.md` plus 2025–2026 neighbours (internalization, no-think, prefix truncation, CoT necessity). It does not copy numbers or verdicts from other repos or session memory.

## Verdict

| Version | Status |
|---|---|
| Original P2 (“after RLVR, CoT is no longer where the model computes; 50% truncate / paraphrase / mid-trace noise hurts SFT more than GRPO”) | **Closed as a paper.** Incremental over CIR-SR (prefix-truncation CIR drops under outcome RL) + JET (LRMs keep most accuracy after 50% prefix truncate) + Weight of Silence (weights-over-scratchpad after RL, chess/latent). |
| Sharpened P2 (below) | **Closed by the 72h probe** (`docs/p2-probe-result-2026-08-30.md`). Capability failed. |

### Sharpened one-sentence claim

On **verbal** CoT, a matched SFT → GRPO path **relocates** computation: empty-trace (no-CoT) accuracy holds up after RL while the SFT checkpoint still needs the trace, and that empty-trace robustness is **not** explained by prefix-only early-exit (JET) nor by pass@k coverage shrink (Invisible Leash / Limit-of-RLVR).

That is the only version worth taking to ICML. The contribution is a **three-regime dissociation along the RL path**, not “pass@k shrinks” and not “you can cut the last 50%.”

---

## Adjacent papers (must-cite; do not redo)

### A. Weight of Silence — Kshirsagar, arXiv:2607.20952

Assumption: latent thoughts are an inference-time scratchpad the model consults.
Object: Coconut-style latent vectors on **chess**, staged latent curriculum then GRPO.
Finding: legality 48% → 61%; checkmate confabulation eliminated. A six-condition causal battery (substitute, matched noise, ablate, length-matched ablate, exact-zero) on the **same model before and after RL**: content substitutions/noise leave play unchanged; only exact-zero collapses; post-RL retains more legality under total signal loss (1% vs 9%). RL adds **robustness to disruption**, not reliance on thought content. Weight-space: RL Δ concentrated in MLP (gate_proj 56.5%), mean effective rank 2.5 / 32.
They also run an explicit-CoT GRPO checkpoint (confabulation also hits 0) but the causal battery is on **latent** thoughts.
**Steal:** pre/post-RL causal battery; “robustness gap is the finding.”
**Not us:** chess + continuous thoughts, not verbal CoT on language tasks. No JET-style prefix vs empty-trace split.

### B. CIR-SR — Yu, Tartaglini, Hase, Guestrin, Potts, arXiv:2604.22074 (ICML-styled)

Closest kill-threat to original P2.
CIR = mean JS between the answer distribution at full CoT vs every prefix truncation (Lanham-style early-answer). SR = verifier recovers the answer from the trace alone.
Qwen2.5 Instruct 1.5B/3B/7B + Llama-3.2-3B, 40 ReasoningGym tasks, outcome RLVR.
Finding: (1) accuracy can rise while CIR/SR **fall** (19/40 CIR down, 17/40 SR down). Low final CIR means the answer is decided **before** `<think>`. (2) On low-CIR/SR tasks, RLVR **without any reasoning** matches accuracy. (3) CIR/SR rise only when Δacc > 50 pp. (4) Small SFT on expert traces **before** RL raises CIR/SR; they treat low CIR as a **bug to fix** (SFT or auxiliary CIR/SR rewards).
They already have a **start-vs-end** CIR comparison and prefix truncation as the instrument.
**Difference we still have:** matched **SFT-CoT vs GRPO of the same stem** on verbal math CoT; empty-trace vs prefix-50% vs full (three regimes); entropy-fork ablation; language tasks where CoT is necessary (Emmons), not ReasoningGym items solvable with no CoT. CIR-SR does not dissociate JET (front-loaded trace) from weights.

### C. JET — Han et al., ICLR 2026, arXiv:2509.23392

“LRMs accumulate sufficient information **early**.” Restricting to the first 75% of a reasoning chain keeps >90% of originally-correct solutions; first **50%** still keeps ~75%. They train Just-Enough Thinking (trajectory truncation in the GRPO rollout + length reward) so the model **stops**.
**Exact 50% prefix intervention on thinking models.** Interpretation is overthinking / evidence accumulation, not “compute moved into weights.” A 50%-truncate-only probe is a JET replica.

### D. Lanham et al., 2023, arXiv:2307.13702

Early answering, adding mistakes, paraphrasing CoT. Prompted CoT, not RL. Faithfulness varies by task; **larger models are often less faithful**. Performance boost is not just extra test-time compute. Snapshot, no SFT→RL curve.

### E. 3TF — Wu et al., arXiv:2511.03408

Thought-Training and Thought-Free inference: hybrid Think/No-Think model, train on CoT, infer with empty think. Qwen3-8B native: Thinking GSM8K 94.2 vs NoThinking 89.9; AIME 80.0 vs 50.0. 3TF recovers most of Thinking accuracy at a fraction of the tokens **by training for internalization**.
**Not a measurement of incidental RL.** If the 72h no-CoT condition only recovers Qwen3’s trained no-think mode, we are 3TF/Qwen3-fusion, not P2.

### F. Implicit / internalized CoT (cite, do not become)

| Paper | What it did | Why not us |
|---|---|---|
| Deng et al., Implicit CoT via KD, arXiv:2311.01460 | Distill teacher hidden states; reason “vertically” | Method to *install* implicit CoT |
| Deng, Choi, Shieber, Explicit→Implicit CoT, arXiv:2405.14838 | Curriculum that **removes** CoT steps during SFT | SFT internalization, not RL localization |
| Coconut, Hao et al., arXiv:2412.06769 | Continuous thought recurrence | Latent architecture |
| CODI / SIM-CoT 2025 | Distill explicit CoT into latents | Method papers |
| CoUT, arXiv:2505.19756 | Prompt “unconscious” internalization | Prompt, not RL path |

P2 is not another internalization **method**. It is a **measurement**: does ordinary outcome GRPO already move verbal compute into the weights, and when.

### G. CoT necessity / faithfulness 2025–2026 (cite, do not become)

| Paper | Why nearby | Why not us |
|---|---|---|
| Emmons et al., *When CoT is Necessary…*, arXiv:2507.05246 | CoT-as-rationalization vs CoT-as-computation; hard tasks force CoT use (Lanham inconsistency **vanishes** as math gets harder) | Safety/monitorability. Predicts P2 should **fail** on hard items if CoT is still the serial scratchpad. Use as a **condition**, not a detector. |
| Chen et al. (Anthropic), *Reasoning Models Don’t Always Say What They Think*, arXiv:2505.05410 | Hint verbalization <20%; outcome RL improves faithfulness then **plateaus**; CoT not necessary in their hint setting | Hint-faithfulness, not empty-trace localization along RL |
| BonaFide, Gur-Arieh et al., arXiv:2605.25052 | Faithfulness **metrics** near chance | Do not start a metric paper |
| *Two Regimes…*, arXiv:2607.23458 | 69% of labeled unfaithfulness is on **wrong** answers; detectors at chance there | Same: no new detector |
| Turpin et al. 2023 | Biasing features flip answers; CoT omits them | Prompted CoT |
| ProFIL, arXiv:2605.11467 | “Reasoning theater”; probe-filtered GRPO | Method to cut theater |
| *Not Just the Destination*, arXiv:2603.12397 | Training on traces (even without answers) changes no-think behaviour | Training-time causal, not inference ablation × checkpoint |
| Feng et al., arXiv:2509.19284 | Failed-step fraction > length; faithfulness **out of scope** | Process quality, not localization |
| Switch, arXiv:2606.13106 | Zeroing Switch latents costs 66.7 pp (they **are** causal) | Latent; opposite finding to Weight of Silence |

### H. RLVR pass@k coverage shrink (backdrop, not the contribution)

| Paper | Finding | Role for P2 |
|---|---|---|
| Yue et al., Limit-of-RLVR, arXiv:2504.13837 | pass@1 up, large-k down; RL paths already in the base | Control. Log pass@k; **do not gate** on coverage shrink. |
| Wu et al., *The Invisible Leash*, arXiv:2507.14843 | Empirical-support shrinkage outweighs expansion | Same control |
| Wang et al., 80/20, NeurIPS 2025, arXiv:2506.01939 | Top ~20% entropy tokens carry the RLVR gradient | Steal entropy-fork ablation for the **full** paper, not the 72h dummy |

If the probe only shows thinking models have higher pass@1 and lower pass@32, we have rediscovered Limit-of-RLVR.

---

## What is actually still open

Three properties **together**, not any one of them:

1. **Empty-trace robustness gap along a training path**, not a single faithfulness scalar.
   Same stem, SFT-CoT checkpoint vs GRPO checkpoint (72h proxy: off-the-shelf no-think vs think/distill).
   Quantity: `drop_N = acc(F) − acc(N)` where N = forced empty CoT / immediate answer (CIR at k=0).
   Prediction: `drop_N(SFT) − drop_N(RL)` is large and positive.

2. **Dissociation from JET (prefix-only).**
   P = prefix 50% of **that sample’s** CoT, then force the answer.
   JET: `drop_P` small, `drop_N` still large.
   Weights: `drop_N` ≈ `drop_P` (both small) on the RL/think checkpoint.
   If 50% truncate is robust and empty-trace still collapses, we are JET.

3. **Not coverage shrink, not “think is just better.”**
   Think/RL must actually be better on full CoT (`acc_F(think) > acc_F(base)`), and pass@k at large k is logged as a **control** (Invisible Leash), not a live criterion.
   Optional ICML-useful: high-entropy fork ablation vs low-entropy body (80/20 DNA) on the same checkpoints — **after** the 72h probe lives. Not in week 1.

A method (if any) only after the finding: e.g. train with an empty-trace auxiliary so RL cannot hide compute in weights, or the reverse. Do not invent this in week 1.

**Why this is not Weight of Silence.** Verbal CoT, language/math, prefix vs empty split, off-the-shelf then a **curve over GRPO steps** (the paper), not chess latents.

**Why this is not CIR-SR.** CIR-SR already shows outcome RL need not raise CIR, and often lowers it, on ReasoningGym Instruct models. They want to **restore** CIR. We need (a) matched SFT-CoT vs GRPO, (b) empty vs prefix dissociation, (c) a necessity band (Emmons) where no-CoT is actually costly for SFT. If the probe only shows CIR-like “RL less sensitive to prefixes,” we agree it is CIR-SR — and we kill it.

---

## Reviewer attack lines (pre-answer)

| Attack | Answer |
|---|---|
| “This is Weight of Silence.” | Chess + latent vectors. We intervene on verbal CoT and split prefix-50% vs empty-trace. |
| “CIR-SR already truncated CoT over RL.” | They report a CIR scalar start→end on Instruct+ReasoningGym and treat low CIR as a training bug. We need empty vs prefix dissociation on a matched SFT-CoT/GRPO stem, on items where SFT still needs the trace. |
| “JET already truncated 50%.” | JET is prefix-only on LRMs (overthinking). If empty-trace still collapses, we are JET and we kill. |
| “Lanham already.” | Prompted CoT, one checkpoint. Path over RL, plus the JET split. |
| “3TF / iCoT already internalized CoT.” | Those **train** internalization. We measure whether ordinary outcome GRPO does it incidentally. |
| “Emmons says hard tasks need CoT.” | That is the **condition**. If empty-trace robustness only appears on GSM8K-easy and dies on a harder slice, we report the boundary; we do not retune the claim into “CoT still matters on AIME.” |
| “Limit-of-RLVR / Invisible Leash.” | Coverage shrink is the backdrop. We do not gate on pass@k Venn. |
| “Qwen3 think vs no-think is the same weights.” | The 72h run is a **proxy**. Same-checkpoint mode switch is conservative (think is *trained* to use traces). If think is still more empty-trace-robust, that is surprising. The paper, if the probe lives, trains a real SFT→GRPO curve. Distill-8B is SFT-on-traces, not GRPO — log it as a third arm, do not treat it as the RL checkpoint. |
| “Length confound: 50% of a 2k-token think is still 1k tokens.” | Gate on **empty-trace** (N), not on 50%. Log mean CoT length. P is only for the JET split. |

---

## 72h probe — only this design

Executable spec: `probe/KILL-p2.md`. Dummy (Mac, no model): `literature_review/icml-2027/probe-p2/` (`uv run weighttraces probe --dummy`). Dummy/smoke never lock. P1 merged 2026-08-30; the 2×A6000 pair may run the P2 GPU job (`probe-p2/scripts/run_a6000_2gpu.sh`).

Do **not** probe “does 50% truncate hurt thinking less than base.” That result is already JET + a length confound.

**Stack (full):** `Qwen/Qwen3-8B` (official Hub id; no `-Instruct` suffix).
- `think`: `enable_thinking=True`
- `base`: `enable_thinking=False`, with a step-by-step user suffix so F still emits verbal steps (not the trained empty-think shortcut alone)

Smoke: `Qwen/Qwen3-0.6B`, same flags. Dummy: no weights.

**Items:** 80 GSM8K (primary). Optional log-only: 20 MATH-500-level items if they fit the GPU budget; they do **not** enter the live test.

**Protocol (prefill, Lanham/CIR-style):** for each model, sample n CoT completions (F). For each F sample, prefill
- P: first 50% of **that sample’s** CoT tokens, then force the answer
- N: empty CoT / empty `<think></think>`, then force the answer

**Must-log:** avg@n and unbiased pass@k (k∈{1,8} ∩ {k: k≤n}); `drop_P`, `drop_N`; mean CoT token length per model; N never uses P1 Venn or P1 connective entropy.

**Live / kill:** frozen in `probe/KILL-p2.md`. All three live flags required. Dummy never decides.

No GRPO in the 72h probe. No second family until live. No tensor-parallel; two processes, one GPU each.

---

## Reading list (full-text for this sweep)

Priority 1 (read): 2607.20952, 2604.22074, 2307.13702, 2509.23392, 2511.03408, 2507.05246, 2505.05410, 2504.13837, 2507.14843, 2405.14838, 2505.09388 (Qwen3 think fusion).

Priority 2 before camera-ready, not before probe: 2605.25052, 2607.23458, 2605.11467, 2603.12397, 2606.13106, 2412.06769, 2506.01939, 2509.19284.
