"""Frozen thresholds. Dummy never returns LIVE."""

from weighttraces.verdict import decide


def _base_summary(acc, drop_p, drop_n):
    return {"acc": acc, "drop_P": drop_p, "drop_N": drop_n}


def test_dummy_never_live():
    s = _base_summary(
        acc={"think": {"F": 0.90, "P": 0.85, "N": 0.82}, "base": {"F": 0.70, "P": 0.50, "N": 0.40}},
        drop_p={"think": 0.05, "base": 0.20},
        drop_n={"think": 0.08, "base": 0.30},
    )
    v = decide(s, dummy=True)
    assert v["decision"] == "DUMMY_SKIP"
    assert v["capability_live"]
    assert v["empty_gap_live"]
    assert v["not_jet_live"]


def test_live_shape_without_dummy_flag():
    s = _base_summary(
        acc={"think": {"F": 0.90, "P": 0.85, "N": 0.82}, "base": {"F": 0.70, "P": 0.50, "N": 0.40}},
        drop_p={"think": 0.05, "base": 0.20},
        drop_n={"think": 0.08, "base": 0.30},
    )
    v = decide(s, dummy=False)
    assert v["decision"] == "LIVE"


def test_jet_replica_kills():
    s = _base_summary(
        acc={"think": {"F": 0.90, "P": 0.88, "N": 0.50}, "base": {"F": 0.70, "P": 0.55, "N": 0.40}},
        drop_p={"think": 0.02, "base": 0.15},
        drop_n={"think": 0.40, "base": 0.30},
    )
    v = decide(s, dummy=False)
    assert v["decision"] == "KILL"
    assert v["jet_replica"]


def test_equal_empty_drop_kills():
    s = _base_summary(
        acc={"think": {"F": 0.90, "P": 0.60, "N": 0.55}, "base": {"F": 0.70, "P": 0.45, "N": 0.35}},
        drop_p={"think": 0.30, "base": 0.25},
        drop_n={"think": 0.35, "base": 0.35},
    )
    v = decide(s, dummy=False)
    assert v["decision"] == "KILL"
    assert not v["empty_gap_live"]


def test_capability_fail_kills():
    s = _base_summary(
        acc={"think": {"F": 0.70, "P": 0.68, "N": 0.66}, "base": {"F": 0.72, "P": 0.40, "N": 0.30}},
        drop_p={"think": 0.02, "base": 0.32},
        drop_n={"think": 0.04, "base": 0.42},
    )
    v = decide(s, dummy=False)
    assert v["decision"] == "KILL"
    assert not v["capability_live"]
