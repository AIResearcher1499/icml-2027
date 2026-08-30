"""Cell accuracies. No P1 Venn, no P2 drop_N."""

from __future__ import annotations


def mean_acc(correct: list[bool]) -> float:
    if not correct:
        return 0.0
    return sum(1 for x in correct if x) / len(correct)
