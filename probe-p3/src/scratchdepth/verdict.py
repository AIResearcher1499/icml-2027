"""Live/kill rules. Thresholds frozen in probe/KILL-p3.md."""

from __future__ import annotations

from typing import Any

SERIAL_DROP_MIN = 0.40
RESTORE_MIN = 0.30
PARALLEL_DROP_MAX = 0.10
LIVE_DEPTH = 2
H_LO = 4
H_HI = 16


def _acc(summary: dict[str, Any], task: str, depth: int, horizon: int, cond: str) -> float:
    return float(summary["acc"][task][str(depth)][str(horizon)][cond])


def drop_h(summary: dict[str, Any], task: str, depth: int, cond: str) -> float:
    return _acc(summary, task, depth, H_LO, cond) - _acc(summary, task, depth, H_HI, cond)


def restore(summary: dict[str, Any], task: str, depth: int, horizon: int) -> float:
    return _acc(summary, task, depth, horizon, "cot") - _acc(summary, task, depth, horizon, "direct")


def decide(summary: dict[str, Any], *, dummy: bool = False) -> dict[str, Any]:
    d_serial = drop_h(summary, "serial", LIVE_DEPTH, "direct")
    d_par = drop_h(summary, "parallel", LIVE_DEPTH, "direct")
    rest = restore(summary, "serial", LIVE_DEPTH, H_HI)

    live_collapse = d_serial >= SERIAL_DROP_MIN
    live_restore = rest >= RESTORE_MIN
    live_not_len = d_par <= PARALLEL_DROP_MAX

    kill_reasons: list[str] = []
    if not live_collapse:
        kill_reasons.append("serial no-CoT drop_h at L=2 < 0.40 (never breaks as h grows)")
    if not live_restore:
        kill_reasons.append("CoT does not restore serial L=2 h=16 by >= 0.30")
    if not live_not_len:
        kill_reasons.append("parallel no-CoT drop_h at L=2 > 0.10 (length confound)")

    live = live_collapse and live_restore and live_not_len
    if dummy:
        decision = "DUMMY_SKIP"
        live = False
    elif live:
        decision = "LIVE"
    else:
        decision = "KILL"

    return {
        "decision": decision,
        "serial_collapse_live": live_collapse,
        "cot_restore_live": live_restore,
        "not_length_live": live_not_len,
        "drop_h_serial_L2_direct": d_serial,
        "restore_serial_L2_h16": rest,
        "drop_h_parallel_L2_direct": d_par,
        "kill_reasons": kill_reasons,
        "note": "Dummy runs never lock the paper.",
    }
