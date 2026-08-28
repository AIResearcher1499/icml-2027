"""Fork entropy from per-token records."""

from __future__ import annotations

import math

from accesstrap.connectives import (
    MIN_LEXICAL_HITS,
    TOP_ENTROPY_FRACTION,
    is_connective_token,
)


def entropy_from_logprobs(logprobs: list[float]) -> float:
    """Natural-entropy from a list of log-probabilities (may be a top-k truncated dist).

    Remaining mass is treated as one leftover bucket so truncated top-k is not
    treated as a full distribution.
    """
    if not logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in logprobs]
    s = sum(probs)
    leftover = max(0.0, 1.0 - s)
    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * math.log(p)
    if leftover > 0:
        ent -= leftover * math.log(leftover)
    return float(ent)


def entropy_from_logits(logits: list[float]) -> float:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    probs = [e / z for e in exps]
    return float(-sum(p * math.log(p) for p in probs if p > 0))


def is_internal_token(token: str, access_block: str | None) -> bool:
    """Drop tokens that are copying the access block verbatim."""
    if not access_block:
        return True
    t = token.strip()
    if len(t) < 4:
        return True
    return t not in access_block


def lexical_entropies(
    tokens: list[str],
    entropies: list[float],
    access_block: str | None,
) -> list[float]:
    out: list[float] = []
    for tok, ent in zip(tokens, entropies, strict=True):
        if is_connective_token(tok) and is_internal_token(tok, access_block):
            out.append(ent)
    return out


def topk_entropies(entropies: list[float], fraction: float = TOP_ENTROPY_FRACTION) -> list[float]:
    if not entropies:
        return []
    k = max(1, int(math.ceil(len(entropies) * fraction)))
    ranked = sorted(entropies, reverse=True)
    return ranked[:k]


def choose_entropy_rule(lexical_hits_by_cond: dict[str, int]) -> str:
    if any(h < MIN_LEXICAL_HITS for h in lexical_hits_by_cond.values()):
        return "top20"
    return "lexical"
