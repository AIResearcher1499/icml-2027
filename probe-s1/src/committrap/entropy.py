"""Answer-probe entropy and t* (sharpest drop). Thresholds frozen in KILL-s1.md."""

from __future__ import annotations

import math

MIN_DROP = 0.05
PROBE_STRIDE = 256


def entropy_from_logits(logits: list[float]) -> float:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    ent = 0.0
    for e in exps:
        p = e / s
        if p > 0.0:
            ent -= p * math.log(p)
    return ent


def probe_cuts(n_cot_tokens: int) -> list[int]:
    if n_cot_tokens <= 0:
        return []
    cuts = list(range(PROBE_STRIDE, n_cot_tokens, PROBE_STRIDE))
    if n_cot_tokens not in cuts:
        cuts.append(n_cot_tokens)
    return cuts


def tstar_index(hs: list[float]) -> int | None:
    """Index i of H_i in the sharpest drop H_{i-1}-H_i with drop >= MIN_DROP.

    i is 0-based into hs; needs at least two probes. None if every drop < 0.05.
    """
    if len(hs) < 2:
        return None
    best_i: int | None = None
    best_drop = MIN_DROP
    for i in range(1, len(hs)):
        drop = hs[i - 1] - hs[i]
        if drop >= MIN_DROP and (best_i is None or drop > best_drop):
            best_drop = drop
            best_i = i
    return best_i
