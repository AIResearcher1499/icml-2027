from pathlib import Path

from scratchdepth.cli import aggregate, main
from scratchdepth.dummy import CONDS, DEPTHS, dummy_items, dummy_sample
from scratchdepth.score import is_correct


def test_scorer_ans_and_hash():
    assert is_correct("running 3*5=15 ANS 8", 8)
    assert is_correct("ANS 8", 8)
    assert is_correct("#### 8", 8)
    assert not is_correct("ANS 9", 8)


def test_dummy_pipeline_writes_dummy_skip(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["probe", "--dummy", "--out", str(tmp_path / "run")])
    assert rc == 0
    summary = (tmp_path / "run" / "summary.json").read_text()
    assert "DUMMY_SKIP" in summary
    assert "LIVE" not in summary.split("decision")[1][:40]
    verdict = (tmp_path / "run" / "verdict.md").read_text()
    assert "# P3 probe verdict: DUMMY_SKIP" in verdict
    assert "# P3 probe verdict: LIVE" not in verdict


def test_probe_without_dummy_exits_2(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["probe", "--out", str(tmp_path / "run")])
    assert rc == 2
    assert not (tmp_path / "run" / "summary.json").exists()


def test_aggregate_has_two_tasks_three_horizons():
    items = dummy_items()
    samples = [
        dummy_sample(it, d, c) for it in items for d in DEPTHS for c in CONDS
    ]
    summary = aggregate(items, samples, dummy=True)
    assert summary["verdict"]["decision"] == "DUMMY_SKIP"
    assert "serial" in summary["acc"] and "parallel" in summary["acc"]
    assert set(summary["acc"]["serial"]["2"]) == {"4", "8", "16"}
    # LIVE-shaped dummy: serial direct drops with h; parallel does not.
    s2 = summary["acc"]["serial"]["2"]
    p2 = summary["acc"]["parallel"]["2"]
    assert s2["4"]["direct"] > s2["16"]["direct"]
    assert s2["16"]["cot"] > s2["16"]["direct"]
    assert p2["4"]["direct"] - p2["16"]["direct"] < 0.40
