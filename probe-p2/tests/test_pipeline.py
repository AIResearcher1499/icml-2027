from pathlib import Path

from weighttraces.cli import aggregate, main
from weighttraces.dummy import dummy_items, dummy_sample
from weighttraces.score import is_correct, normalize_num


def test_scorer_hash_and_last_number():
    assert is_correct("steps #### 72", "72")
    assert is_correct("#### 72.0", "72")
    assert not is_correct("#### 73", "72")
    assert normalize_num("72.0") == "72"


def test_dummy_pipeline_writes_dummy_skip(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["probe", "--dummy", "--out", str(tmp_path / "run")])
    assert rc == 0
    summary = (tmp_path / "run" / "summary.json").read_text()
    assert "DUMMY_SKIP" in summary
    assert "LIVE" not in summary.split("decision")[1][:40]


def test_aggregate_has_three_conditions():
    items = dummy_items()
    samples = [
        dummy_sample(it, m, c, i)
        for it in items
        for m in ("base", "think")
        for c in ("F", "P", "N")
        for i in range(4)
    ]
    summary = aggregate(items, samples, n_samples=4, dummy=True)
    assert summary["verdict"]["decision"] == "DUMMY_SKIP"
    assert summary["acc"]["think"]["F"] > summary["acc"]["base"]["F"]
    assert summary["drop_N"]["base"] > summary["drop_N"]["think"]
