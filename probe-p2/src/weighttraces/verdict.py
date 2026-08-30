"""Live/kill rules. Thresholds frozen in probe/KILL-p2.md."""

from __future__ import annotations

from typing import Any

CAPABILITY_MIN = 0.05
EMPTY_GAP_MIN = 0.10
NOT_JET_MAX = 0.08
JET_PREFIX_MAX = 0.05
JET_N_SLACK = 0.05


def decide(summary: dict[str, Any], *, dummy: bool = False, smoke: bool = False) -> dict[str, Any]:
    acc = summary["acc"]
    drop_p = summary["drop_P"]
    drop_n = summary["drop_N"]

    cap = acc["think"]["F"] - acc["base"]["F"]
    dN_b = drop_n["base"]
    dN_t = drop_n["think"]
    dP_t = drop_p["think"]

    live_cap = cap >= CAPABILITY_MIN
    live_gap = (dN_b - dN_t) >= EMPTY_GAP_MIN
    live_not_jet = (dN_t - dP_t) <= NOT_JET_MAX

    jet_replica = (dP_t < JET_PREFIX_MAX) and (dN_t >= dN_b - JET_N_SLACK)

    kill_reasons: list[str] = []
    if not live_cap:
        kill_reasons.append("think is not better on full CoT by >= 0.05 (capability)")
    if not live_gap:
        kill_reasons.append("empty-trace gap drop_N(base)-drop_N(think) < 0.10")
    if not live_not_jet:
        kill_reasons.append("empty-trace drop exceeds prefix drop by > 0.08 (JET-shaped)")
    if jet_replica:
        kill_reasons.append("JET replica: prefix-robust but empty-trace not more robust than base")

    live = live_cap and live_gap and live_not_jet and not jet_replica
    if dummy or smoke:
        decision = "DUMMY_SKIP" if dummy else "SMOKE_SKIP"
        live = False
    elif live:
        decision = "LIVE"
    else:
        decision = "KILL"

    return {
        "decision": decision,
        "capability_live": live_cap,
        "empty_gap_live": live_gap,
        "not_jet_live": live_not_jet,
        "jet_replica": jet_replica,
        "capability_delta": cap,
        "empty_gap": dN_b - dN_t,
        "n_minus_p_think": dN_t - dP_t,
        "kill_reasons": kill_reasons,
        "note": "Smoke/dummy runs never lock the paper.",
    }
