# P3 novelty sweep (When is a language channel necessary?)

Sweep date: 2026-08-30.
Backup stem if P2 dies. **Original P3 is too close to 2023–2025 prior work.** A sharpened claim remains open. Do not run a GPU train against the original wording. Do not lock P3 in `docs/05-decision.md`. Dummy never locks.

This sweep is full-text on the kill-threats listed in `docs/04-proposals.md` plus neighbours (hidden reasoning, CoT expressivity, grokking, length generalization, scratchpads, overthinking). It does not copy numbers or verdicts from other repos or session memory. It does not retune P1 or P2.

## Verdict

| Version | Status |
|---|---|
| Original P3 (“on synthetic tasks with horizon *h*, models without a verbal channel fail when depth ≲ *c·h*; fit `P(success \| no-CoT) = f(depth / h)` on 20M–300M, then test whether 0.6B–8B overthinking items are those where weights already suffice”) | **Closed as a paper.** Incremental over Physics of LM 2.1 (depth vs reasoning length, even *with* CoT) + Li et al. 2024 (CoT vs no-CoT on serial vs parallel synthetic) + Feng et al. 2023 (bounded-depth transformers cannot do arithmetic without CoT) + Merrill & Sabharwal (no-CoT ⊆ TC⁰; poly CoT = P; empirical log-depth vs *n*) + Emmons 2025 / Sprague 2024 / Chen overthinking (pretrained CoT-necessity ≈ difficulty). |
| Sharpened P3 (below) | **Open, contingent on the 72h probe.** Kill if the probe only rediscovers Li’s serial-vs-modadd plot, Dziri/Zhou length collapse, or “easy items do not need CoT.” |

### Sharpened one-sentence claim

A verbal channel is a **serial-depth prosthesis**, not a general thinking bonus: at matched sequence length, no-CoT accuracy collapses when *serial* horizon *h* exceeds the one-pass depth budget, CoT restores that cell, and a length-matched **parallel** control does not collapse. The remaining paper, if the probe lives, is whether that frozen (*h*, *L*) predictor ranks pretrained 0.6B–8B CoT-uplift **better than difficulty** — not another “small transformers need scratchpads” plot.

That is the only version worth taking to ICML. The 72h probe is a GO/NO-GO that our stack can produce the dissociation, not a lock.

---

## Adjacent papers (must-cite; do not redo)

### A. Physics of Language Models, Part 2.1 — Ye, Xu, Li, Allen-Zhu, arXiv:2407.20311 (ICLR 2025)

Closest kill-threat to original P3’s *depth* story.
Controlled iGSM (synthetic GSM-like DAGs, arithmetic mod 23). GPT-2-rotary trained from scratch. Hidden (mental) reasoning via V-probing: the model plans `nece(A)` before generating; it also learns all-pair `dep(A,B)` even for unused parameters (“level-2”).
**Result 7+8 (the original-P3 kill):** depth, not size, is the reasoning resource. A 16-layer 576-dim model solves harder problems (reasoning length `op`) than a 4-layer 1920-dim model twice as large. **This holds even when CoT is used.** Deeper layers predict `nece(A)` at larger distance *t* from the query. They explicitly say CoT does *not* remove the depth need, because deciding the *first* CoT step still requires multi-step mental planning. Adding a “backward thinking” CoT would reduce the depth requirement; standard textbook CoT does not include that plan.
**Steal:** controlled synthetic with a numeric horizon; depth vs width at matched parameter count; probing as a later paper tool.
**Not us:** they train *with* CoT and show depth still matters for *planning*. Original P3 (“no-CoT fails iff depth ≲ *c·h*”) is the *execution* story, and they already own “depth vs reasoning length on iGSM.” Do not reuse iGSM. Do not claim a transferable *f*(depth/*h*) curve they did not plot — they plotted accuracy vs `op` for several depths, which is the same object.

Part 1 (arXiv:2305.13673) is CFG/Dyck and hidden DP in attention. Cite; it is not CoT-necessity.

### B. Li, Liu, Zhou, Ma — *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems*, arXiv:2402.12875 (ICLR 2024)

Closest kill-threat to the **72h probe**.
Theory: constant-precision constant-depth transformers without CoT ⊆ AC⁰; *T* CoT steps can simulate SIZE[*T*]. Poly CoT = P/poly. Log CoT does not leave AC⁰.
Empirics on tiny decoder-only transformers: **modular addition** (TC⁰, parallelizable — CoT helps little; depth 1 already does parity) vs **permutation composition S₅**, **iterated squaring**, **circuit value** (serial — CoT helps a lot, especially at low depth). Hint (intermediate values in the prompt, not generated) is weaker than CoT.
**If the 72h probe only reproduces “CoT helps S₅ / iterated ops more than mod-add at low depth,” we are a Li replica and we do not lock a paper.** The probe uses that dissociation as a GO/NO-GO, then (if LIVE) the paper has to *transfer* a frozen predictor, not republish the plot.

### C. Feng, Zhang, Gu, Ye, He, Wang — *Towards Revealing the Mystery behind Chain of Thought*, arXiv:2305.15408 (NeurIPS 2023 oral)

Bounded-depth transformers cannot directly output answers for arithmetic / linear equations unless size is super-polynomial in input length (parallel complexity, not serialized cost). Autoregressive transformers of *constant* size can solve both by generating CoT in a standard math format. Also DP.
Empirics: transformers fail at direct answers and succeed with CoT demonstrations.
**Not us:** existence/expressivity. They do not fit *f*(depth/*h*), do not have a parallel control at matched length as a live gate, and do not transfer to pretrained overthinking.

### D. Merrill & Sabharwal — CoT expressivity and log-depth

- *The Expressive Power of Transformers with Chain of Thought*, arXiv:2310.07923: no-CoT decoder ≈ TC⁰; linear CoT recognizes regular languages; poly CoT = P (projected pre-norm).
- *A Little Depth Goes a Long Way*, arXiv:2503.03961: Θ(log *n*) depth (even looped/universal) expresses regular-language recognition and graph connectivity; width must be superpolynomial to leave TC⁰; **O(log *n*) CoT steps stay in TC⁰**. Empirical fit for regular-language recognition: depth *d* ≈ 4.8 log₂ *n* − 15.8 (*r*² = 0.93). Depth scaling beats width and beats CoT for these problems.
**Kill-threat to “CoT is the only prosthesis.”** For some serial problems, **depth** is a cheaper prosthesis than CoT. Sharpened P3 must not claim “the language channel is necessary whenever the problem is serial.” It is necessary when *one-pass depth is too small and we are not going to grow L*. The probe freezes *L* ∈ {2, 4, 8} and asks whether CoT substitutes at small *L* on a serial task whose *plan* is trivial (running product), which is the Feng/Li execution regime, not the Physics-of-LM planning regime.

### E. Emmons et al. — *When Chain of Thought is Necessary…*, arXiv:2507.05246

CoT-as-rationalization vs CoT-as-computation. Necessity = the model cannot succeed without CoT. Replicating Turpin/Chen/Lanham: simple hints stay unfaithful; **complex hints** (multi-step arithmetic in the hint) force the hint into the CoT. Lanham-style inconsistency **vanishes as math gets harder** (magnitude sweep on *ax=b*). Safety/monitorability paper on frontier models.
**Steal:** necessity as the object; difficulty as a *condition*.
**Not us:** no synthetic depth axis, no parallel control, no small pretrains. Predicts that a sloppy “overthinking items are easy items” transfer will just rediscover this paper. Use as the **difficulty confound** the probe must beat.

### F. Scratchpads, length generalization, grokking, compositionality (cite, do not become)

| Paper | What it did | Why not us |
|---|---|---|
| Nye et al., scratchpads, arXiv:2112.00114 | Train to emit intermediate steps; addition → program execution | Method: scratchpads help multi-step. No depth/*h* law, no parallel control as a live gate |
| Wei / Kojima CoT prompting | Prompted CoT on pretrained LMs | Prompt, not small-train necessity |
| Zhou et al., RASP-L length gen, arXiv:2310.16028 | Transformers length-generalize iff a short RASP-L program exists; scratchpad *formats* unlock parity/addition | Length generalization / algorithm class, not CoT vs depth. **Kill if our serial collapse is only “longer *n* fails”** |
| Dziri et al., Faith and Fate, arXiv:2305.18654 | Computation-graph depth *and* width; accuracy decays with both; subgraph matching | Graph complexity, not verbal-channel vs weights. Width is their other axis; our parallel control is the width/parallel analogue |
| Lee et al., Teaching Arithmetic to Small Transformers, arXiv:2307.03381 | Reverse / simplified / detailed scratchpad formats for addition | Data format, not *h* vs *L* |
| Power et al. 2022 grokking; Nanda et al. modular-addition circuits | Delayed generalization of modular arithmetic *in the weights* | Internalization without a verbal channel. Backdrop: weights *can* suffice. Not a CoT-necessity curve |
| Olsson et al. induction heads; Jelassi et al. copy vs SSMs | Copy is a parallel/induction-head skill | **Why copy-with-offset is log-only**, not the live serial arm |
| Pfau, Merrill, Bowman, *Let’s Think Dot by Dot*, arXiv:2404.15758 | Filler tokens (`. . .`) can replace informative CoT on *parallelizable* hidden compute; hard to learn without dense supervision | Extra tokens ≠ verbal content. If filler matches CoT on our serial arm, the “language” story dies — log filler as a control in the *paper*, not in the 72h dummy |
| Goyal et al. pause tokens, ICLR 2024 | Pretrain with pause tokens | Method to install extra serial slots |
| Prystawski, Li, Goodman, arXiv:2304.03843 | CoT helps when training data has *local* dependency clusters | Data-locality, not depth budget |
| Bhattamishra / Hahn / Yao on Dyck | Transformers and Dyck / counters | Language structure, not CoT vs *L* |

### G. Pretrained CoT-uplift / overthinking (the naive transfer is closed)

| Paper | Finding | Role for P3 |
|---|---|---|
| Sprague et al., *To CoT or not to CoT*, arXiv:2409.12183 | CoT helps mainly math/symbolic; on MMLU, no-CoT ≈ CoT unless an equals sign appears | Task-*type*, not item-level serial horizon |
| Chen et al., *Do NOT Think That Much for 2+3=?*, arXiv:2412.21187 | o1-like models overthink easy questions (1953% more tokens on 2+3) | Efficiency catalog. “Easy ⇒ overthinking” is the **difficulty confound** |
| OptimalThinkingBench, arXiv:2508.13141 | Overthinking vs underthinking; no model is optimal | Benchmark, not a law |
| Feng et al., *What Characterizes Effective Reasoning?*, arXiv:2509.19284 | Failed-step fraction > length; faithfulness out of scope | Process quality (already in P2’s neighbour list) |
| 3TF / iCoT / Coconut / Deng implicit CoT | *Train* internalization | Method papers. P3 is not another internalization method |

If the transfer chapter only shows “GSM8K items the 8B already solves without CoT don’t need CoT,” we have rediscovered Sprague + Chen. The transfer must beat **no-CoT accuracy** (and length) as a predictor.

### H. Hidden reasoning in residual streams (cite, do not become)

Physics of LM probing; *Reading Between the Dots* (arXiv:2607.03502) decodes filler-token hidden compute; Switch latents are causal (arXiv:2606.13106). P3’s 72h probe does not probe hidden states. A later paper section may; week 1 does not.

---

## What is actually still open

Three properties **together**, not any one of them:

1. **Serial collapse at small depth, no-CoT.**
   Frozen task: iterated modular *product* of length *h* (serial). Direct-answer (no scratchpad) accuracy at *L*=2 must fall as *h* grows from 4 to 16.
   If this fails, we cannot even reproduce Li/Feng — kill.

2. **CoT as a serial-depth prosthesis on that cell.**
   Forced scratchpad = running partial product, then the answer. It must restore *L*=2, *h*=16. A “hint” (values in the prompt, not generated) is log-only; Li already showed hint < CoT.

3. **Dissociation from length / difficulty / parallel width.**
   Length-matched **parallel** control: iterated modular *sum* of the same *h* and same *p*. No-CoT must *not* collapse as *h* grows. This is the live gate that Zhou (length gen), Dziri (graph size), and “longer is harder” cannot pass.
   Copy-with-offset is **log-only**. Copy is an induction-head / RASP-short program; using it as the serial arm would be Jelassi/Zhou, not serial depth.

A method (if any) only after the finding: e.g. a cheap *ĥ* estimator that skips CoT when *ĥ* / *L* is small. Do not invent this in week 1.

**Transfer (paper, not 72h).** Freeze the (*h*, *L*) predictor on the synthetic grid. On 0.6B–8B, bin items by an *ĥ* that is *not* no-CoT accuracy (gold solution DAG depth, or a serial/parallel synthetic rewrite of the same numbers). Predict CoT-uplift. Beat difficulty. Empty-trace (P2-style N) as a secondary instrument only if P2 is already dead — do not dual-lock.

**Why this is not Physics of LM.** They own depth vs `op` *with* CoT on iGSM planning. We forbid iGSM. We use a trivial-plan serial chain so CoT can actually substitute for depth (Feng’s construction), and we require a parallel control.

**Why this is not Li 2024.** They already have CoT vs no-CoT on serial vs mod-add. The 72h probe *reproduces* that as a stack check. The paper, if LIVE, is the **frozen-predictor transfer** that they did not do. If we stop at the synthetic plot, we agree it is Li — and we do not lock.

**Why this is not Emmons / Sprague / overthinking.** Those are pretrained, difficulty- or task-typed. Transfer must lose if the only predictor that works is “easy.”

---

## Reviewer attack lines (pre-answer)

| Attack | Answer |
|---|---|
| “This is Physics of LM 2.1.” | They train *with* CoT and show depth is still needed for *planning* on iGSM. We use a trivial-plan serial product and a no-CoT vs CoT × parallel control. If we rerun iGSM, we kill. |
| “Li already did CoT on S₅ vs mod-add.” | Yes. The 72h probe is a GO/NO-GO that we can produce the dissociation. If that is all we have, we do not lock. The paper is a frozen (*h*, *L*) predictor that beats difficulty on pretrained items. |
| “Feng already proved no-CoT arithmetic needs super-poly size.” | Expressivity. We need a *learnable* collapse at tiny *L* plus a parallel control that does not collapse. |
| “Merrill 2025 already fitted *d* ~ 4.8 log *n*.” | Regular-language recognition, and they argue *depth beats CoT* for that class. Our serial arm is iterated multiply (NC¹-ish serial), not DFA tracking. If our parallel control is secretly a DFA, we are Merrill and we kill. |
| “Emmons already: hard ⇒ CoT is necessary.” | Difficulty. We need serial vs parallel at **matched no-CoT accuracy / matched length**. |
| “Sprague / Chen: easy items don’t need CoT.” | Same confound. Transfer must beat no-CoT accuracy as a predictor. |
| “This is Faith and Fate (graph depth).” | They collapse with *both* depth and width. Our parallel arm is the width/parallel analogue and must *not* collapse. |
| “This is length generalization / RASP-L.” | Then the parallel control fails (3). If serial and parallel both die with *h*, we are Zhou/Dziri. |
| “Copy-with-offset is your serial task.” | No. Copy is log-only. Live serial = iterated mod-product. |
| “Filler tokens already buy serial slots.” | Pfau: filler helps *parallelizable* hidden compute and needs dense supervision. Log filler in the paper; not a 72h live gate. |
| “Grokking: weights eventually suffice.” | That is the internalization endgame. The probe asks whether, at the *frozen* train budget, no-CoT has already internalized the serial chain. Do not train until grokking to rescue a KILL. |
| “P2 already does empty-trace.” | Different object (RL relocation vs depth prosthesis). Do not lock two stems. If P2 LIVE, P3 stays buried. If P2 KILL, P3 may use empty-trace as an instrument *after* the synthetic probe lives. |

---

## 72h probe — only this design

Executable spec: `probe/KILL-p3.md`. Dummy (Mac, no train, no 0.6B): `literature_review/icml-2027/probe-p3/` (`uv run scratchdepth probe --dummy`). Dummy never locks. Do not train 20M–300M for this probe. Do not eval Qwen3-0.6B. Do not touch P1 `accesstrap` or P2 `weighttraces`. Do not use the 2×A6000 pair while P2 is running.

Do **not** probe “does CoT help more as problems get longer.” That result is already Nye + Li + Dziri.

**Stack (full, when A6000s are free):** GPT-2-rotary, `d_model=256`, 4 heads, depths *L* ∈ {2, 4, 8} (width frozen so depth is the axis). Train from scratch. No 0.6B, no 8B, no GRPO.

**Tasks:**
- Serial (live): iterated modular **product** *a₁···aₕ* mod 17, *h* ∈ {4, 8, 16}
- Parallel (live): iterated modular **sum** of the same *h* and *p*
- Copy-with-offset: log-only; does not enter the live test

**Conditions:** `direct` (no-CoT, answer only) vs `cot` (forced scratchpad of the running state, then answer).

**Live / kill:** frozen in `probe/KILL-p3.md`. All three live flags required. Dummy never decides.

No paper repo until `docs/05-decision.md` records a lock. No P3-v2 on the same claim after a NO-GO.

---

## Reading list (full-text for this sweep)

Priority 1 (read): 2407.20311, 2402.12875, 2305.15408, 2310.07923, 2503.03961, 2507.05246, 2112.00114, 2310.16028, 2305.18654, 2409.12183, 2412.21187, 2404.15758, 2304.03843.

Priority 2 before camera-ready, not before probe: 2305.13673 (Physics of LM Part 1), 2307.03381, Power 2022 grokking, 2508.13141, pause-token / filler follow-ups, 2607.03502.
