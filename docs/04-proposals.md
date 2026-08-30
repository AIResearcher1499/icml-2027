# Candidate papers (unlocked)

Last updated: 2026-08-30.

Four stems. Rank is a probe order, not a lock. Run kill probes **before** GRPO. Lock at most one. ICML thin-slicing policy: do not submit two close variants.

Default stack: Qwen3-8B (base and/or instruct), vLLM, Unsloth GRPO if needed, 2×A6000.

---

## P1 — The Access Trap

**Probe order:** 1 (preferred).

**Assumption.** Adding an external language channel (tool, memory, retrieved document) *enlarges* the solution set.

**Claim.** The same way arbitrary order is a false flexibility, **access** is a false flexibility. The model uses trusted external text to skip high-entropy internal forks. pass@k coverage **shrinks** relative to no-access; mean entropy of internal (non-tool) tokens drops; the no-access unique-solve set is not recovered by access (Flexibility Trap’s 21.3% vs 0.6% pattern).

**Why it is not TIM.** TIM: accuracy vs pairwise process quality, code interpreter + math only. P1: coverage + fork entropy, **three** channels, causal paraphrase/noise of the external text.

**Why it is not Flexibility Trap.** Object is *access*, not generation order.

**Quantities.** pass@1, pass@k (k ∈ {32, 256}); Jaccard of solved-item sets; mean token entropy on internal tokens; Δ under paraphrase / noise / gold-vs-distractor evidence.

**Fit.** Inference-first. Optional short GRPO only after the finding. Strongest ICML analogue for an NLP author.

**72-hour kill probe.** 100 items (multi-step math + multi-hop QA). Three conditions: (a) no extra context, (b) gold evidence/tool, (c) gold + one authoritative-looking distractor. If pass@32 does **not** shrink under (b) or (c), and internal fork entropy does not drop → **kill P1**. If it shrinks and entropy drops → P1 lives.

**ICML reviewer questions to pre-answer.** “How is this not TIM?” Coverage + entropy + three channels. “How is this not Flexibility Trap?” Different flexibility. “When does access *help*?” Need a negative/condition section (short-horizon lookup vs long-horizon inference).

---

## P2 — Weights over traces after RL

**Sharpened claim (2026-08-29 novelty sweep):** see `docs/07-p2-novelty.md`. Original wording is closed as a paper. Hedge only; not locked. Frozen 72h spec: `probe/KILL-p2.md`. Dummy: `probe-p2/`.

**Probe order:** 2.

**Assumption.** After RLVR, the chain of thought is where the model computes. Truncating or noising it must collapse accuracy.

**Claim.** On **language** tasks (not chess): the same intervention suite (truncate CoT at 50%, paraphrase keeping the answer, noise mid-trace embeddings) hurts the SFT checkpoint more than the GRPO checkpoint. RL **moves** computation into the weights; the trace becomes epiphenomenal. Known pass@k shrinkage is the backdrop, not the contribution. The contribution is **causal localization along the RL path**.

**Why it is not *Weight of Silence*.** That paper is chess + latent vectors. This is verbal CoT, SFT vs RL checkpoints, entropy-aware interventions.

**Why it is not Lanham 2023 / CIR-SR.** Those are snapshots. P2 needs a **curve over RL steps**.

**Quantities.** Accuracy Δ under each intervention × checkpoint; pass@1 vs pass@256 (control); fraction of high-entropy tokens whose ablation changes the answer.

**Fit.** Needs a short GRPO run on Qwen3-8B LoRA (feasible). The 72-hour probe uses *off-the-shelf* distill/thinking vs base — no training.

**72-hour kill probe (no train).** Qwen3-8B-Base vs a thinking/distill 8B (e.g. DeepSeek-R1-Distill-Qwen-8B or Qwen3-8B thinking). Truncate CoT at 50%. If the thinking model drops **as much as** the base, the “moved into weights” story is weak → **lean away from P2**. If thinking is more truncate-robust *and* has higher pass@1 → P2 lives, then train the curve.

**ICML reviewer questions.** “Lanham already.” Answer: path over RL, not a single faithfulness number. “CIR/SR already.” Answer: intervention × checkpoint, entropy forks, not only causal-importance scalars.

---

## P3 — When is a language channel necessary?

**Sharpened claim (2026-08-30 novelty sweep):** see `docs/08-p3-novelty.md`. Original wording is closed as a paper. Backup only; not locked. Frozen 72h spec: `probe/KILL-p3.md`. Dummy: `probe-p3/`.

**Probe order:** 3 (backup if P2 dies). Do not start GPU while P2 is on the A6000s.

**Assumption.** CoT always helps; longer is better; overthinking is a length-control problem on easy items.

**Claim.** A verbal channel is a **serial-depth prosthesis**, not a general thinking bonus. At matched sequence length, no-CoT accuracy collapses when *serial* horizon *h* exceeds the one-pass depth budget; CoT restores that cell; a length-matched **parallel** control does not collapse. Transfer (paper, not 72h): that frozen (*h*, *L*) predictor ranks 0.6B–8B CoT-uplift better than difficulty.

**Why it is not *Physics of Language Models*.** They train *with* CoT and show depth is still needed for *planning* on iGSM. We forbid iGSM. Trivial-plan serial product + parallel control.

**Why it is not Li 2024.** They already have CoT vs no-CoT on serial vs mod-add. The 72h probe is a GO/NO-GO that our stack can produce the dissociation. If that plot is all we have, we do not lock.

**Fit.** Tiny GPT-2 on 2×A6000 after P2 frees the pair. Less NLP-object, more ML-science. Weaker comparative advantage, cleaner ICML voice.

**72-hour kill probe (no 20M–300M, no 0.6B).** Iterated mod-product (serial) vs mod-sum (parallel), *h* ∈ {4,8,16}, *L* ∈ {2,4,8}, forced scratchpad vs direct. Live/kill frozen in `probe/KILL-p3.md`. Dummy: `uv run scratchdepth probe --dummy`. If no-CoT never breaks as *h* grows on the serial arm, or the parallel arm collapses too → **kill P3**.

---

## P4 — 80/20 for tool and memory tokens

**Probe order:** 4. Do not start unless P1 lives.

**Assumption.** Every token in an agent trace (thought, tool argument, observation) deserves RL.

**Claim.** High-entropy forks sit at **channel boundaries** (call tool / write memory / trust observation), not in CoT body. GRPO on the top 20% entropy of *tool-decision* tokens ≥ full GRPO; RL on observation tokens hurts.

**Why it is not 80/20.** 80/20 is CoT math. P4 is compound systems.

**Fit.** Needs P1’s measurement stack plus GRPO in a tool sandbox. Heavier. Method risk (looks like “GRPO trick”).

**No 72-hour probe this week.**

---

## Five-month sketch (only after a lock)

| Window | P1 | P2 |
|---|---|---|
| Week 1 | Kill probe | Kill probe (no train) |
| Weeks 2–3 | 3-channel protocol, pass@k, entropy | GRPO 8B, checkpoint every ~50 steps |
| Month 2 | Paraphrase/noise interventions | Intervention suite × checkpoint |
| Month 3 | More tasks; optional GRPO that forbids fork-skipping | Localization curve |
| Month 4 | Ablations + “when access helps” | Second family + seeds |
| Month 5 | Write; ICML abstract | Write; ICML abstract |

P3 is the backup stem. P4 is a follow-up chapter, not a first paper.
