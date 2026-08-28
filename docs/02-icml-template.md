# What ICML rewarded in 2026

Last updated: 2026-08-28.

ICML 2026 Outstanding Paper (one of two, from ~24k submissions):

> Ni et al., *The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models* (arXiv:2601.15165).

The field believed arbitrary-order generation **strictly enlarges** the reasoning solution space. They measured **pass@k coverage**, not pass@1. Arbitrary order **narrowed** coverage: models skipped high-uncertainty tokens that are the actual forks. On HumanEval at k=1024, left-to-right uniquely solved 21.3% of problems that arbitrary order missed; the reverse was 0.6%. The method (JustGRPO) is a *consequence*: constrain order **during RL**, keep parallel decoding at inference.

Same lab, NeurIPS 2025: Wang et al., *Beyond the 80/20 Rule* (arXiv:2506.01939). ~20% high-entropy CoT tokens carry nearly all of the RLVR gradient. Training the low-entropy 80% hurts.

## The joint research programme

**Computation lives at high-entropy forks. Any “flexibility” that lets the model skip forks collapses reasoning.**

That is the DNA to copy. Not “another GRPO.” Not “another dLLM decoder.”

## Reviewer criteria (ICML 2026 CFP, paraphrased)

Original, rigorous, of interest to the *machine learning* community. Claims supported by reproducible experiments and/or theory. Situated against prior work. Thin slices of the same theme across concurrent submissions are treated as prior work.

Award shortlist criteria (ICML blog, 2026-07-05): strong-accept quality, nontrivial longevity, interest **beyond a niche subcommunity**, topic diversity.

An NLP object is allowed. An NLP-only *audience* is a reject risk. The paper must change how ML people think about generation, RL, or test-time compute — not only how *ACL people annotate traces.

## Shape of a viable paper from this lab

```
assumption the field is using
        ↓
quantity that can go the other way (coverage, causal Δ, fork entropy)
        ↓
intervention that isolates the channel (mask / noise / forbid)
        ↓
optional minimalist method that follows the finding
```

Honor the template. Do not add a method until the finding exists.
