from weighttraces.data import apply_shard, gold_from_gsm8k_answer
from weighttraces.dummy import dummy_items


def test_gsm8k_gold_from_solution():
    assert gold_from_gsm8k_answer("Natalia sold 48+24=72 clips.\n#### 72") == "72"
    assert gold_from_gsm8k_answer("#### 1,000") == "1000"


def test_shard_splits_without_overlap():
    items = dummy_items()
    a = apply_shard(items, "0/2")
    b = apply_shard(items, "1/2")
    assert len(a) + len(b) == len(items)
    assert {x.item_id for x in a}.isdisjoint({x.item_id for x in b})
