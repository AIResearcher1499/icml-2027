import pytest

pytest.importorskip("datasets")

from accesstrap.data import load_gsm8k, load_hotpot


def test_gsm8k_drops_final_calc():
    items = load_gsm8k(2)
    it = items[0]
    assert it.split == "math"
    assert it.gold_access
    assert it.distractor_access != it.gold_access
    # Natalia item: gold 72 must not be the leaked identity
    if it.item_id == "gsm8k-0":
        assert "72" not in it.gold_access


def test_hotpot_same_context_distractor():
    items = load_hotpot(2)
    it = items[0]
    assert it.split == "qa"
    assert it.gold_access
    assert it.distractor_access
    assert it.gold_access != it.distractor_access
