"""Frozen thresholds. Dummy never returns LIVE."""

from scratchdepth.verdict import decide


def _acc_table(serial_direct, serial_cot, parallel_direct):
    """Build acc[task][depth][h][cond] with L=2 live cells filled; others dummy-high."""

    def cell(direct, cot):
        return {"direct": direct, "cot": cot}

    def depths_for(direct4, direct16, cot16, parallel=False):
        # L=2 is the live row. Other depths exist so the schema matches KILL-p3.md.
        row2 = {
            "4": cell(direct4, 0.95),
            "8": cell((direct4 + direct16) / 2, 0.90),
            "16": cell(direct16, cot16),
        }
        if parallel:
            row2 = {
                "4": cell(direct4, 0.95),
                "8": cell(direct4, 0.95),
                "16": cell(direct16, 0.95),
            }
        filled = {"2": row2}
        for d in ("4", "8"):
            filled[d] = {
                "4": cell(0.95, 0.95),
                "8": cell(0.95, 0.95),
                "16": cell(0.95, 0.95),
            }
        return filled

    return {
        "serial": depths_for(serial_direct[0], serial_direct[1], serial_cot),
        "parallel": depths_for(parallel_direct[0], parallel_direct[1], 0.95, parallel=True),
    }


def _summary(serial_direct, serial_cot, parallel_direct):
    return {"acc": _acc_table(serial_direct, serial_cot, parallel_direct)}


def test_dummy_never_live():
    s = _summary(serial_direct=(0.90, 0.20), serial_cot=0.85, parallel_direct=(0.90, 0.88))
    v = decide(s, dummy=True)
    assert v["decision"] == "DUMMY_SKIP"
    assert v["serial_collapse_live"]
    assert v["cot_restore_live"]
    assert v["not_length_live"]


def test_live_shape_without_dummy_flag():
    s = _summary(serial_direct=(0.90, 0.20), serial_cot=0.85, parallel_direct=(0.90, 0.88))
    v = decide(s, dummy=False)
    assert v["decision"] == "LIVE"


def test_serial_no_collapse_kills():
    s = _summary(serial_direct=(0.90, 0.85), serial_cot=0.90, parallel_direct=(0.90, 0.88))
    v = decide(s, dummy=False)
    assert v["decision"] == "KILL"
    assert not v["serial_collapse_live"]


def test_cot_no_restore_kills():
    s = _summary(serial_direct=(0.90, 0.20), serial_cot=0.25, parallel_direct=(0.90, 0.88))
    v = decide(s, dummy=False)
    assert v["decision"] == "KILL"
    assert not v["cot_restore_live"]


def test_parallel_collapse_kills():
    s = _summary(serial_direct=(0.90, 0.20), serial_cot=0.85, parallel_direct=(0.90, 0.40))
    v = decide(s, dummy=False)
    assert v["decision"] == "KILL"
    assert not v["not_length_live"]
