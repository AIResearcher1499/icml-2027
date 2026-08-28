"""pass@k and coverage-set helpers. Pure functions."""

from __future__ import annotations

import math
from typing import Iterable


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. 2021 unbiased pass@k estimator."""
    if n <= 0 or k <= 0:
        return 0.0
    if k > n:
        k = n
    if c < 0:
        c = 0
    if c > n:
        c = n
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def solved(c: int) -> bool:
    return c >= 1


def venn_counts(in_a: Iterable[bool], in_b: Iterable[bool]) -> dict[str, int]:
    a_only = b_only = both = neither = 0
    for xa, xb in zip(in_a, in_b, strict=True):
        if xa and xb:
            both += 1
        elif xa:
            a_only += 1
        elif xb:
            b_only += 1
        else:
            neither += 1
    return {
        "A_minus_B": a_only,
        "B_minus_A": b_only,
        "A_and_B": both,
        "neither": neither,
    }


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
