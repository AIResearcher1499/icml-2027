"""Live/kill rules. Thresholds frozen in probe/KILL-s1.md."""

from __future__ import annotations

from typing import Any

MIN_CELL = 40
N_LIVE_ITEMS = 64
MIN_DEFINED_FRAC = 0.50
MAX_RECOVERY = 0.05
MIN_GAP = 0.10
MIN_GAP_MINUS_50 = 0.05


def decide(summary: dict[str, Any], *, dummy: bool = False, smoke: bool = False) -> dict[str, Any]:
    n_samples = int(summary.get("n_samples") or 8)
    n_cell = int(summary["n_cell"])
    n_defined = int(summary["n_defined"])
    live_traces = N_LIVE_ITEMS * n_samples
    defined_frac = n_defined / max(live_traces, 1)
    recovery = float(summary["recovery_continue"])
    gap_t = float(summary["gap_tstar"])
    gap_50 = float(summary["gap_50"])

    live_cell = n_cell >= MIN_CELL
    live_phase = defined_frac >= MIN_DEFINED_FRAC
    live_lock = recovery <= MAX_RECOVERY
    live_gap = (gap_t >= MIN_GAP) and ((gap_t - gap_50) >= MIN_GAP_MINUS_50)

    kill_reasons: list[str] = []
    if not live_cell:
        kill_reasons.append("wrong-at-t* cell too small (n_cell < 40); Xu High Reliability")
    if not live_phase:
        kill_reasons.append("two-phase missing (defined t* fraction < 0.50)")
    if not live_lock:
        kill_reasons.append("original suffix recovers after t* (continue > 0.05)")
    if not live_gap:
        kill_reasons.append("refork does not beat continue by 0.10, or 50% cut matches (JET replica)")

    live = live_cell and live_phase and live_lock and live_gap
    if dummy or smoke:
        decision = "DUMMY_SKIP" if dummy else "SMOKE_SKIP"
        live = False
    elif live:
        decision = "LIVE"
    else:
        decision = "KILL"

    return {
        "decision": decision,
        "cell_live": live_cell,
        "phase_live": live_phase,
        "lockin_live": live_lock,
        "gap_live": live_gap,
        "n_cell": n_cell,
        "defined_frac": defined_frac,
        "recovery_continue": recovery,
        "gap_tstar": gap_t,
        "gap_50": gap_50,
        "kill_reasons": kill_reasons,
        "note": "Smoke/dummy runs never lock the paper.",
    }
