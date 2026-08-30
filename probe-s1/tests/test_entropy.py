from committrap.entropy import probe_cuts, tstar_index


def test_tstar_sharpest_drop():
    assert tstar_index([1.0]) is None
    assert tstar_index([1.0, 0.97]) is None  # drop 0.03 < 0.05
    assert tstar_index([1.2, 0.2]) == 1
    assert tstar_index([1.0, 0.9, 0.1]) == 2  # drops 0.1 and 0.8


def test_probe_cuts_include_end():
    assert probe_cuts(0) == []
    assert probe_cuts(256) == [256]
    assert probe_cuts(512) == [256, 512]
    assert probe_cuts(257) == [256, 257]
