# S1 novelty sweep (compute after commitment)

Sweep date: 2026-08-30.
**Open, contingent on the 72h probe.** Do not lock in `docs/05-decision.md`. Dummy never locks.

This sweep is full-text on the kill-threats in `docs/10-icml-level-stems-2026-08-30.md`. It does not retune P1/P2.

## Verdict

| Version | Status |
|---|---|
| Original S1 (“after Xu’s change-point, sequential continue has the wrong sign; refork recovers”) | **Too close to Xu + JET unless a control is frozen.** Xu already: Confidence Region is reliable + redundant → early exit and CUSUM-weighted voting. JET already: % truncation of *wrong* traces can correct them (ECR). |
| Sharpened S1 (below) | **Open.** The remaining claim is a **sign dissociation at Xu’s `t*` that is not a JET length-percentile replica.** |

### Sharpened one-sentence claim

Xu’s Confidence Region is reliable **only on traces that are already correct at the change-point**. On traces whose probed answer at `t*` is **wrong**, matched-budget compute spent *continuing* does not recover (or locks in); the same budget spent *re-forking at `t*`* does — and that gap is **larger than** the same contrast at a length-matched 50% cut (JET’s instrument).

That is the only version worth taking to ICML. Early-exit (Xu) and train-to-stop (JET) are the method papers on the *correct* / *average* population. We need the **wrong-at-`t*`** cell and the **entropy-vs-length** control.

---

## Adjacent papers (must-cite; do not redo)

### A. Xu et al., arXiv:2606.02020 — two-phase entropy + CUSUM

Object: **answer-probe** entropy `H_i` (force an intermediate answer after each CoT step `T_i`), not next-token entropy of the CoT.
Finding: Uncertainty Region (acc <20%) → abrupt Confidence Region (acc >60% plateau). Confidence has High Reliability and High Redundancy (>30% extra tokens after the answer appears).
Method: CUSUM on `H_i`; early **exit** at `τ_h`; **weight full trajectories** by `S_final` for self-consistency.
Models: Qwen3-4B-Thinking-2507, Qwen3-14B, R1-Distill-Qwen-7B. AIME24/25, GPQA.
**They never resample from `t*`.** They never stratify continue vs truncate on **wrong-at-`t*`**. Their story is: once in Confidence, reasoning is largely done and extra tokens are waste — for the **aggregate**, which is dominated by traces that became correct.
**Steal:** `H_i` definition, CUSUM, two-phase structure. Use the same detector so we are not a “different entropy” paper.
**Kill if:** on wrong-at-`t*`, continue ≈ refork, or the wrong-at-`t*` set is too small to test (High Reliability leaves no cell).

### B. Hiremath & Hiremath, arXiv:2606.11211 — CDUR

Calibration Drift Under Reasoning: ECE as a function of **prompted budget** (none / light / medium / heavy). Hypothesis Lock-In: early `h0` commits the chain; verbalized confidence rises with length. CABStop: halt when verbalized `p̂ − α̂ > δ`.
Evidence: Llama-3.1-8B, **47 trap questions**, 42% valid parses, 70B incomplete. Authors themselves call it directional, not a law.
**Not us:** verbalized confidence, prompt-budget, no entropy `t*`, no refork. Cite as lock-in intuition. Do not become a bigger CDUR.

### C. von Recum et al., arXiv:2602.07470 — CoT interventions

Perturb the model’s **own correct** CoT at fractional timesteps `{0.1,0.3,0.5,0.7,0.9}`. Seven interventions. Finding: RLLMs **recover**; worse when early; paraphrasing suppresses doubt and hurts; recovery inflates length.
**Opposite cell:** they start from traces that already solve the item. We start from traces **wrong at `t*`**. Time index ≠ entropy change-point. Cite for “early interventions matter” and doubt-as-recovery. Not a scoop.

### D. Han et al., JET, arXiv:2509.23392 — prefix truncation + RL

Pilot on MATH500, Distill-Qwen-7B: first 50% of CoT keeps ~75% of originally-correct answers (ARR); truncating **originally-incorrect** traces can **correct** them (ECR rises as T falls). Then they **train** (DAPO + length reward) to stop.
**Closest kill on the wrong-trace cell.** Difference we still need: (1) cut is entropy `t*`, not `floor(T·L)`; (2) matched-budget **refork** vs continue, not only truncate-and-answer; (3) no train in the 72h probe; (4) live gate that `t*` gap **beats** the 50% gap, else we are JET without RL.

### E. Other nearby (cite, do not become)

| Paper | Why nearby | Why not us |
|---|---|---|
| DEER / Dynasor / EAT / HALT-CoT | Heuristic early exit | Xu already beats them; we are not an exit method |
| Local Branch Routing, 2606.25354 | Token-level lookahead tree | Method; every-token not `t*` |
| 80/20, CES, ETR | Entropy for **training** | S2, not S1 |
| Snell test-time compute / Brown monkeys | More samples help | We need **where** the extra sample is spent |

---

## What is actually still open

Three properties **together**:

1. **Wrong-at-`t*` cell exists.** Enough traces have a probed answer at Xu’s `t*` that is wrong (otherwise Xu’s High Reliability kills the design).
2. **Sign of sequential compute is non-positive on that cell.** `acc(continue +B) − acc(truncate at t*) ≤ ε` (ε frozen). Extra tokens after commitment do not recover.
3. **Refork at `t*` beats continue, and beats JET’s 50% cut.** Matched extra tokens. If 50% shows the same gap, we are JET.

A method (pause/refork decoder) only after LIVE.

---

## Reviewer attack lines

| Attack | Answer |
|---|---|
| “This is Xu.” | They exit and reweight **full** traces. We condition on wrong-at-`t*` and **branch**. If that cell is empty or continue=refork, we agree we are Xu — and we kill. |
| “This is JET.” | They cut at length percentile and then train. We freeze `t*` vs 50% as a live **control**. Equal gaps → KILL. |
| “This is CDUR.” | Verbalized conf × prompt budget, n=47. Different quantity. |
| “This is von Recum.” | They perturb **correct** traces. We do not inject; we spend compute two ways on **wrong-at-`t*`**. |
| “This is just beam search from a prefix.” | Then 50% should match `t*`. The paper is that the **entropy phase** is the prefix that flips the sign, not an arbitrary cut. |
| “Xu’s H_i is expensive.” | We use their detector (pre-registered probe stride). Do not silently switch to next-token entropy. |

---

## 72h probe

Executable spec: `probe/KILL-s1.md`. Runner: `probe-s1/` (`uv run committrap probe --dummy` on Mac; `scripts/run_a6000_2gpu.sh` on the A6000 box). Dummy never locks. Do not retune after a number.

No GRPO. No second family until live.
