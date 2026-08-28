# Constraints

Last updated: 2026-08-28.

## Author / lab

- Background: NLP (evaluation, analysis, language as object). Not systems, not theory-first.
- Compute: **2× NVIDIA A6000**, 48 GB each (~96 GB total). Typical split: one GPU train, one GPU rollout, or TP=2.
- Time: ~5 months to an ICML 2027 deadline (CFP not posted; plan late January 2027).
- This scan did **not** reuse prior repo docs or session memory. Portfolio collisions are listed at the bottom so we do not open a second stem on a locked line.

## What fits this envelope

Doable, and enough for ICML if the *claim type* is right:

- Full SFT 7B; LoRA 14B–32B; 70B 4-bit inference
- GRPO / RLVR on Qwen3-8B / Qwen2.5-7B (Unsloth / verl)
- Pretrain from scratch ~4M–1B with a real hyperparameter sweep
- Fine-tune existing 7B dLLMs (LLaDA / Dream) — **not recommended** as a new stem (see collisions)
- Heavy eval + local 8B–32B judges

Not doable as an ICML SOTA chase:

- Pretrain 7B+ from scratch
- Long-CoT RLVR on 32B+ against lab recipes
- “New architecture beats Llama-3 70B”
- Heavy multimodal / video / world models

**Claim type that wins with this budget:** isolate a mechanism. Not chase a leaderboard.

## What “ICML-shaped” means for an NLP author

The object can be language (CoT, tool arguments, memory notes, retrieved passages). The *argument* cannot be an ACL analysis catalog. It needs:

1. A default assumption the field is using
2. The right quantity (coverage, causal Δ, fork entropy — not accuracy alone)
3. An intervention (mask / noise / forbid a channel)
4. A method only if it is the *consequence* of the finding, and it is minimal

## Portfolio collisions (do not open a second stem)

These already have repos in the portfolio. Out of scope for this scan:

- KV-cache compression / allocation
- RAG entity attribution / deceptive grounding
- Diffusion-LM speed / fertility outside English
- Bits-per-parameter memorization (MDM vs AR)

A new ICML paper may *cite* those literatures. It must not retune a NO-GO from them, and must not be a thinly sliced twin of a locked stem.

## Explicitly dropped for this venue

- Generic RAG / HyDE-style retrieval
- Agent frameworks and multi-agent debate
- LoRA / PEFT / quantization variants
- GRPO-on-GSM8K/AIME for +2 points
- Jailbreak catalogs, hallucination detectors
- dLLM parallel-decoding systems (ICML 2026 Outstanding already took the assumption)
- ACL-native judge-bias catalogs without a training loop
