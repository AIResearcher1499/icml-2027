# P1 novelty sweep (Access Trap)

Sweep date: 2026-08-28.
Direction chosen. **Original P1 is too close to 2026 prior work.** A sharpened claim remains open. Do not run the 72h probe against the original wording.

## Verdict

| Version | Status |
|---|---|
| Original P1 (“access shrinks accuracy / reasoning across tool, memory, retrieval”) | **Closed as a paper.** Incremental over TIM + Tool-Overuse + MemTrap + NoisyBench + DeR². |
| Sharpened P1 (below) | **Open, contingent on the 72h probe.** Kill if the probe only rediscovers mean-accuracy drops. |

### Sharpened one-sentence claim

External language access is a *Flexibility Trap*: even **gold** (helpful) access lets the model skip high-entropy internal forks, so the **pass@k solved-item set** of no-access is not a subset of access; fork entropy of internal connectives **drops** under gold access and **rises** under distractors (two regimes, one mechanism family).

That is the only version worth taking to ICML.

---

## Adjacent papers (must-cite; do not redo)

### A. Flexibility Trap — Ni et al., ICML 2026 Outstanding, arXiv:2601.15165

Assumption: arbitrary *order* enlarges the reasoning solution space.
Finding: pass@k curves flatten; at k=1024, AR uniquely solves 21.3% of HumanEval that arbitrary-order misses (reverse 0.6%). Mechanism: confidence-based unmasking **bypasses** high-entropy logical connectives; later context **degrades fork entropy**.
Object: decoding order in dLLMs. **Not** tools / memory / RAG.
**Steal:** pass@k *set* overlap + fork-entropy at connectives. That is the DNA.

### B. TIM — Bayat et al., ACL 2026, arXiv:2511.10899

Code interpreter + competition math (PyMath, n=1679, code helpful but not sufficient).
Finding: accuracy **+19.3 pp**; non-tool wins pairwise *process* quality up to 41.5%; more tool calls → shallower reasoning; errors shift arithmetic → logic/assumption. TIM ≈ substitute enumeration for proof.
Quantities: pass@1, pairwise process, miss-rate of derivation steps. **No pass@k coverage Venn. No fork entropy.**
Single channel. Process-quality paper (ACL-shaped).

### C. Tool-Overuse Illusion — Zeng et al., arXiv:2604.19749 (ICML-styled; prior work for 2027)

Tools on GSM8K/AIME. avg@8 on *simple* items (avg@8≥0.5 without tools) **drops 3.3–14.5%** when tools are enabled. Qwen3-8B still averages **2.2 tool calls** in the avg@1024>0.8 bin. Mechanisms: knowledge-epistemic illusion; outcome-only RLVR increases tool turns (e.g. 2.2→6.8).
They already use **avg@1024 as a coverage proxy for internal knowledge**, then show tools *hurt that bin*.
**Closest kill-threat to original P1.** Difference we still have: they report mean avg@k, not solved-set Venn; they do not measure connective fork entropy; tools+math only; they do not contrast gold vs distractor entropy.

### D. MemTrapBench — Wang et al., arXiv:2608.20202 (20 Aug 2026)

Memory (even faithful, relevant) induces Reasoning Fixation / Belief Distortion. All five memory frameworks **underperform no-memory by >10 pp** (Gemini-3-Flash 85.16→71.17). Designed trap seeds, 1050 dialogues. Mitigation: AdaptiveMem prompt.
**Pass@1, trap-constructed.** Not coverage sets, not fork entropy, not gold-helpful access as a flexibility.

### E. NoisyBench — Lee et al., arXiv:2601.07226

11 datasets, RAG/reasoning/alignment/tool. Distractors (random doc, chat history, hard negative) drop SOTA up to **80%**. Agents amplify by over-trusting noisy tools. Inverse test-time scaling. **Output entropy grows with noise** (confusion). Attention piles on distractor tokens. RARE (rationale-aware RL) helps.
This is the *confusion* regime. Our gold-access claim predicts the **opposite** entropy movement at forks (degradation / skip, not inflation). If the probe cannot show that dissociation, we are NoisyBench.

### F. DeR² — Ying et al., arXiv:2601.21937

Four regimes: Instruction-only / Concepts / Related-only / Full-set. **Mode-switch fragility:** Instruction-only > Full-set on some frontier models (Gemini-3-Pro 64.2 vs 53.7). Related-only can beat Full-set (noise-induced loss). Process errors: abandon parametric path, fail to execute named concepts.
Accuracy by regime, not pass@k sets or fork entropy. Retrieval channel only. Useful as a *condition* (when extra docs hurt), not as our mechanism.

### G. Other nearby (cite, do not become)

| Paper | Why nearby | Why not us |
|---|---|---|
| Acting Less is Reasoning More / OTC-PO, arXiv:2504.14870 | Cognitive offloading; too many tool calls | Efficiency / tool-count, not coverage |
| SMART (ACL 2025 Findings) | Tool-overuse mitigation | Agent policy, not forks |
| Tool-Light / entropy-after-tool, arXiv:2509.23285 | Entropy of *subsequent* tokens after a tool return | Training signal for TIR quality, not coverage collapse |
| Entropy-reduction as tool-quality, arXiv:2602.02050 | Good tool calls lower later entropy | Correlates quality with ΔH, does not claim access shrinks the solvable set |
| PASS@(k,T), arXiv:2604.14877 | Tool-use **RL** expands pass@(k,T) vs base on compositional gathering | Training, not inference-time access on a frozen policy; can *contradict* a sloppy P1 |
| Why Do Reasoning Models Lose Coverage, arXiv:2605.17026 | Coverage shrink from SFT forks in data | Post-training data, not external text |
| 80/20 Rule, NeurIPS 2025, arXiv:2506.01939 | High-entropy CoT tokens carry RLVR | No external channel |
| Peak-then-collapse KG tools, arXiv:2605.26037 | RL quote-and-stop | Training collapse, not inference access |

---

## What is actually still open

Three properties **together**, not any one of them:

1. **Solved-set coverage** (Flexibility Trap Fig. 4), not mean accuracy / avg@k.
   Items uniquely solved at pass@k by no-access vs gold-access vs distractor-access.
   Prediction: gold-access unique-solves few; no-access unique-solves many (access set ⊄ no-access set, and often a strict subset at large k).
2. **Fork-entropy degradation on internal connectives** (Flexibility Trap Fig. 7) under **gold** access.
   Prediction: mean entropy of Therefore/Thus/Since (or model-specific high-entropy forks) **drops** when gold evidence/tool output is in context, even if pass@1 rises.
3. **Dissociation from the noise regime** (NoisyBench).
   Same items, distractor access: fork entropy **rises** (confusion) or attention mass moves to the distractor; gold access: fork entropy **falls** (skip). One paper, two arrows.

Optional but ICML-useful: the same pair of quantities on **two channels** (tool and retrieved gold passage). Memory as a third channel is a plus, not required for the first paper — MemTrap already owns trap-memory pass@1.

A method (if any) only after the finding: constrain the model to emit the fork *before* consuming the external span (Access-JustGRPO analogue). Do not invent this in week 1.

---

## Reviewer attack lines (pre-answer)

| Attack | Answer |
|---|---|
| “This is TIM.” | TIM is process pairwise + pass@1 on code-math. We report solved-set Venn + connective entropy. |
| “This is Tool-Overuse.” | They show avg@8 drop on simple bins. We show coverage *sets* and entropy *degradation at forks*, including gold retrieval not just tools. |
| “This is NoisyBench.” | They study distractors; entropy goes up. We need gold access to move entropy the other way. |
| “This is MemTrap.” | Traps + pass@1. We are not constructing traps; gold-helpful access is the interesting case. |
| “This is Flexibility Trap with a rename.” | Same *quantities*, different *flexibility* (external text vs generation order). If the probe cannot show entropy degradation under gold access, we agree it is a rename — and we kill it. |
| “PASS@(k,T) says tools expand coverage.” | That paper is **RL training** of a tool policy vs base. We measure a **frozen** policy with vs without access at inference. Both can be true. |

---

## 72h probe — only this design

Code: `literature_review/icml-2027/probe/` (`KILL.md` is the frozen executable spec). Dummy: `uv run accesstrap probe --dummy`. A6000: `probe/scripts/run_a6000.sh`.

Do **not** probe “does accuracy drop with context.” That result is already published.

Stack: `Qwen/Qwen3-8B` with `enable_thinking=False` (official Hub id; Qwen3 has no `-Instruct` suffix). One GPU is enough.

**Items (≈100, mixed):**
- 50 GSM8K/MATH items in the *internally solvable* band (pilot: no-access pass@8 ≥ 0.5). Channel: optional Python interpreter (gold-helpful).
- 50 multi-hop QA items with a **gold supporting paragraph** available. Channel: retrieved gold span, not a trap.

**Conditions (within item):**
- A: no extra access
- B: gold access (tool result or gold paragraph)
- C: distractor access (wrong-but-fluent tool output, or hard-negative paragraph)

**Must-log:**
- pass@1, unbiased pass@k (k∈{1,8,32})
- solved-set Venn at k=32: |A∖B|, |B∖A|, |A∩B|
- mean token entropy of a fixed connective list in the *internal* (non-tool, non-quote) span
- optional: whether the first high-entropy fork is emitted before or after the first access span

**Live (continue):**
- |A∖B| ≫ |B∖A| at k=32 on gold condition B, **and**
- connective entropy B < A, **and**
- connective entropy C ≥ A (or attention/confusion signature distinct from B)

**Kill:**
- only mean avg@k drops on B (Tool-Overuse replica), or
- only C hurts and entropy rises (NoisyBench replica), or
- Venn is symmetric / B superset of A with no entropy drop (access actually expands coverage; P1 dies)

No GRPO. No second model family until live.

---

## Reading list (full-text before the probe write-up)

Priority 1 (already read for this sweep): 2601.15165, 2511.10899, 2604.19749, 2608.20202, 2601.07226, 2601.21937.

Priority 2 before camera-ready, not before probe: 2504.14870, 2509.23285, 2602.02050, 2604.14877, 2506.01939, SMART ACL 2025.
