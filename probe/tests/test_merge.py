from pathlib import Path

from accesstrap.cli import main


def test_merge_math_and_qa_shards(tmp_path: Path):
    math_dir = tmp_path / "math"
    qa_dir = tmp_path / "qa"
    out = tmp_path / "merged"
    assert main(["probe", "--dummy", "--n-math", "2", "--n-qa", "0", "--out", str(math_dir)]) == 0
    assert main(["probe", "--dummy", "--n-math", "0", "--n-qa", "2", "--out", str(qa_dir)]) == 0
    assert main(["merge", str(math_dir), str(qa_dir), "--out", str(out), "--n-samples", "8"]) == 0
    summary = (out / "summary.json").read_text()
    assert '"n_items_pool": 4' in summary
    assert (out / "verdict.md").exists()


def test_merge_incomplete_shard_fails(tmp_path: Path):
    math_dir = tmp_path / "math"
    qa_dir = tmp_path / "qa"
    assert main(["probe", "--dummy", "--n-math", "2", "--n-qa", "0", "--out", str(math_dir)]) == 0
    qa_dir.mkdir()
    (qa_dir / "items.json").write_text("{}\n")
    assert main(["merge", str(math_dir), str(qa_dir), "--out", str(tmp_path / "m")]) == 2
