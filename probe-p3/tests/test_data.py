from scratchdepth.data import gold_of, make_item
from scratchdepth.dummy import dummy_items


def test_serial_gold_is_product_mod_17():
    assert gold_of("serial", [3, 5, 2, 7]) == (3 * 5 * 2 * 7) % 17
    assert gold_of("parallel", [3, 5, 2, 7]) == (3 + 5 + 2 + 7) % 17


def test_make_item_reproducible_and_nonzero_operands():
    a = make_item("serial", 8, seed=0, idx=3)
    b = make_item("serial", 8, seed=0, idx=3)
    assert a.operands == b.operands
    assert a.gold == gold_of("serial", a.operands)
    assert all(1 <= x <= 16 for x in a.operands)
    assert a.horizon == 8


def test_dummy_items_cover_live_grid():
    items = dummy_items()
    keys = {(it.task, it.horizon) for it in items}
    assert keys == {("serial", 4), ("serial", 8), ("serial", 16), ("parallel", 4), ("parallel", 8), ("parallel", 16)}
    assert len(items) == 24
