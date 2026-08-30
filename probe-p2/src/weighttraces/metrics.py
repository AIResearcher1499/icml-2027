"""avg@n and unbiased pass@k. No P1 Venn."""

from __future__ import annotations

import math


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    if k > n:
        k = n
    c = min(max(c, 0), n)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_acc(correct: list[bool]) -> float:
    if not correct:
        return 0.0
    return sum(1 for x in correct if x) / len(correct)
