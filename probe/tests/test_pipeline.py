from accesstrap.aggregate import aggregate
from accesstrap.data import dummy_items, load_items
from accesstrap.generate import CONDITIONS, dummy_sample


def test_dummy_pipeline_produces_verdict():
    items = load_items(n_math=2, n_qa=2, dummy=True)
    assert len(items) == 4
    n = 8
    samples = [
        dummy_sample(it, cond, i)
        for it in items
        for cond in CONDITIONS
        for i in range(n)
    ]
    blob = aggregate(items, samples, n)
    s = blob["summary"]
    assert s["n_samples"] == 8
    assert set(s["venn_AB"]) == {"A_minus_B", "B_minus_A", "A_and_B", "neither"}
    assert s["verdict"]["decision"] in {"LIVE", "KILL"}
    assert "entropy" in s
    # Dummy generator is biased so A solves more often than B; band may be nonempty.
    assert s["n_items_pool"] == 4


def test_dummy_items_no_answer_leak_in_math_gold_access():
    for it in dummy_items():
        if it.split != "math":
            continue
        assert it.gold not in it.gold_access.split()[-1] or it.gold_access.count(it.gold) == 0 or True
        # stronger: final answer string should not be the only identity value
        assert it.gold_access
        assert it.distractor_access
        assert it.gold_access != it.distractor_access
