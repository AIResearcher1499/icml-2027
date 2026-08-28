# Closed claims — do not redo

Last updated: 2026-08-28.

If a probe rediscovers one of these and nothing else, that is a NO-GO, not a paper.

| Assumption already shot | Paper | Takeaway |
|---|---|---|
| Arbitrary token order helps reasoning | Flexibility Trap, ICML 2026 Outstanding, arXiv:2601.15165 | Coverage shrinks; high-entropy tokens get skipped |
| RLVR should train every token | Beyond the 80/20 Rule, NeurIPS 2025, arXiv:2506.01939 | Top ~20% entropy tokens carry the gradient |
| Latent scratchpad is read at inference (chess) | *The Weight of Silence*, arXiv:2607.20952 | Noise on latent thoughts does not change play; gain is in the weights |
| RLVR expands pass@k / empirical support | Limit-of-RLVR; *The Invisible Leash*, arXiv:2507.14843 | pass@1 up, large-k coverage down |
| Outcome reward makes traces causally important | CIR/SR, arXiv:2604.22074 | Correct answers with unused or insufficient traces |
| Faithfulness *metrics* measure faithfulness | BonaFide, arXiv:2605.25052 | Near chance; no transfer step ↔ CoT |
| Unfaithfulness is one regime | *Two Regimes…*, arXiv:2607.23458 | 69% of labeled unfaithfulness is on **wrong** answers; detectors at chance there |
| Tool use that raises accuracy improved reasoning | Tool-Induced Myopia (TIM), ACL 2026, arXiv:2511.10899 | Accuracy +19.3, process quality down. **ACL-shaped**; they did not measure pass@k coverage or fork entropy |
| Longer CoT / more review is better | Feng et al., *What Characterizes Effective Reasoning?*, arXiv:2509.19284 | Failed-step fraction beats length; they **left faithfulness out of scope** |
| Extra memory / docs always help | MemTrapBench (Aug 2026); consolidation-from-GT failures; DeR² mode-switch fragility | Catalogs exist; no unified coverage+entropy law across channels |

## Open next to these (the actual gaps)

- TIM did not measure **solution-set coverage** or **internal fork entropy** when a tool is attached.
- Flexibility Trap did not treat **external text** (tools, memory, retrieved passages) as a flexibility.
- *Weight of Silence* is chess + latent vectors, not verbal CoT across SFT → RL checkpoints.
- 80/20 is CoT-on-math. Tool-decision tokens and memory-write tokens are untested.
- BonaFide says metrics fail. The remaining move is **intervention**, not a new detector.

## Do not start

- Another dLLM RL that preserves arbitrary order
- Another unfaithful-CoT classifier
- Another agent-memory architecture leaderboard
- Another GRPO math recipe
