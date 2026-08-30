from pathlib import Path

from committrap.cli import main
from committrap.score import is_correct


def test_scorer():
    assert is_correct("steps #### 72", "72")
    assert is_correct("<think>48+36=84</think>\nThe answer is #### 72", "72")
    assert not is_correct("#### 73", "72")


def test_bench_plus_dummy_is_tiny(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["probe", "--dummy", "--bench", "--out", str(tmp_path / "bench")])
    assert rc == 0
    summary = (tmp_path / "bench" / "summary.json").read_text()
    assert "DUMMY_SKIP" in summary or "SMOKE_SKIP" in summary


def test_dummy_pipeline_dummy_skip(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["probe", "--dummy", "--out", str(tmp_path / "run")])
    assert rc == 0
    verdict = (tmp_path / "run" / "verdict.md").read_text()
    assert "DUMMY_SKIP" in verdict
    assert "# S1 probe verdict: LIVE" not in verdict
