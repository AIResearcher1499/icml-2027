from committrap.verdict import decide


def _s(**kw):
    base = {
        "n_samples": 8,
        "n_cell": 80,
        "n_defined": 320,
        "recovery_continue": 0.02,
        "gap_tstar": 0.20,
        "gap_50": 0.05,
    }
    base.update(kw)
    return base


def test_dummy_never_live():
    v = decide(_s(), dummy=True)
    assert v["decision"] == "DUMMY_SKIP"
    assert v["cell_live"]
    assert v["gap_live"]


def test_live_shape():
    v = decide(_s(), dummy=False)
    assert v["decision"] == "LIVE"


def test_jet_replica_kills():
    v = decide(_s(gap_tstar=0.20, gap_50=0.18), dummy=False)
    assert v["decision"] == "KILL"
    assert not v["gap_live"]


def test_recovery_kills():
    v = decide(_s(recovery_continue=0.20), dummy=False)
    assert v["decision"] == "KILL"
    assert not v["lockin_live"]


def test_small_cell_kills():
    v = decide(_s(n_cell=10), dummy=False)
    assert v["decision"] == "KILL"
    assert not v["cell_live"]
