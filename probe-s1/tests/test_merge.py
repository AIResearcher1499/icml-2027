from pathlib import Path

from committrap.cli import main


def test_dummy_shards_merge_stays_dummy_skip(tmp_path: Path):
    a = tmp_path / "s0"
    b = tmp_path / "s1"
    out = tmp_path / "merged"
    assert main(["probe", "--dummy", "--shard", "0/2", "--out", str(a)]) == 0
    assert main(["probe", "--dummy", "--shard", "1/2", "--out", str(b)]) == 0
    rc = main(
        ["merge", str(a), str(b), "--out", str(out), "--n-samples", "4", "--n-items", "4"]
    )
    assert rc == 0
    assert "DUMMY_SKIP" in (out / "verdict.md").read_text()


def test_merge_incomplete_exits_2(tmp_path: Path):
    a = tmp_path / "s0"
    out = tmp_path / "merged"
    assert main(["probe", "--dummy", "--shard", "0/2", "--out", str(a)]) == 0
    rc = main(["merge", str(a), "--out", str(out), "--n-samples", "4", "--n-items", "4"])
    assert rc == 2
