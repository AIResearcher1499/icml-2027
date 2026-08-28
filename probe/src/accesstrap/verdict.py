"""Live/kill rules. Thresholds frozen in KILL.md."""

from __future__ import annotations

from typing import Any

MIN_EXCESS = 5
COVERAGE_RATIO = 2.0


def coverage_live(a_minus_b: int, b_minus_a: int) -> bool:
    excess = a_minus_b - b_minus_a
    return a_minus_b >= COVERAGE_RATIO * b_minus_a and excess >= MIN_EXCESS


def entropy_live(mean_a: float | None, mean_b: float | None, mean_c: float | None) -> tuple[bool, bool]:
    """Return (gold_drop_live, distractor_dissoc_live)."""
    if mean_a is None or mean_b is None or mean_c is None:
        return False, False
    gold_drop = mean_b < mean_a
    dist_dissoc = mean_c >= mean_a
    return gold_drop, dist_dissoc


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    venn = summary["venn_AB"]
    a_minus_b = venn["A_minus_B"]
    b_minus_a = venn["B_minus_A"]
    cov = coverage_live(a_minus_b, b_minus_a)
    gold_drop, dist_dissoc = entropy_live(
        summary["entropy"]["A"],
        summary["entropy"]["B"],
        summary["entropy"]["C"],
    )
    live = cov and gold_drop and dist_dissoc
    kill_reasons: list[str] = []
    if not cov:
        kill_reasons.append(
            "coverage Venn fails (|A-B|>=2|B-A| and excess>=5); "
            "possible Tool-Overuse replica or access expands coverage"
        )
    if not gold_drop:
        kill_reasons.append("no gold fork-entropy drop (B < A failed)")
    if not dist_dissoc:
        kill_reasons.append("distractor entropy below no-access (C >= A failed)")
    if live:
        decision = "LIVE"
    else:
        decision = "KILL"
    return {
        "decision": decision,
        "coverage_live": cov,
        "gold_entropy_drop": gold_drop,
        "distractor_dissociation": dist_dissoc,
        "kill_reasons": kill_reasons,
        "note": "Smoke/dummy runs never lock the paper.",
    }
