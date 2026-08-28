from accesstrap.metrics import unbiased_pass_at_k, venn_counts
from accesstrap.verdict import coverage_live, decide, entropy_live


def test_pass_at_k_all_correct():
    assert unbiased_pass_at_k(32, 32, 1) == 1.0
    assert unbiased_pass_at_k(32, 0, 1) == 0.0


def test_pass_at_k_partial():
    # n=2, c=1, k=1 → 0.5
    assert abs(unbiased_pass_at_k(2, 1, 1) - 0.5) < 1e-9


def test_venn():
    v = venn_counts([True, True, False, False], [True, False, True, False])
    assert v == {"A_minus_B": 1, "B_minus_A": 1, "A_and_B": 1, "neither": 1}


def test_coverage_live_thresholds():
    assert coverage_live(10, 2) is True
    assert coverage_live(6, 2) is False  # excess 4 < 5
    assert coverage_live(5, 0) is True
    assert coverage_live(4, 0) is False


def test_entropy_live():
    assert entropy_live(1.0, 0.7, 1.1) == (True, True)
    assert entropy_live(1.0, 1.2, 1.1) == (False, True)
    assert entropy_live(1.0, 0.7, 0.9) == (True, False)
    assert entropy_live(None, 0.7, 1.1) == (False, False)


def test_decide_live():
    summary = {
        "venn_AB": {"A_minus_B": 12, "B_minus_A": 2, "A_and_B": 20, "neither": 0},
        "entropy": {"A": 1.0, "B": 0.6, "C": 1.2},
    }
    d = decide(summary)
    assert d["decision"] == "LIVE"


def test_decide_kill():
    summary = {
        "venn_AB": {"A_minus_B": 1, "B_minus_A": 10, "A_and_B": 20, "neither": 0},
        "entropy": {"A": 1.0, "B": 1.2, "C": 0.5},
    }
    d = decide(summary)
    assert d["decision"] == "KILL"
    assert d["kill_reasons"]
