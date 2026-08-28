from pathlib import Path

from accesstrap.checkpoint import load_samples
from accesstrap.cli import main


def test_resume_skips_finished_keys(tmp_path: Path):
    out = tmp_path / "run"
    assert main(["probe", "--dummy", "--out", str(out)]) == 0
    samples_path = out / "samples.jsonl"
    lines = samples_path.read_text().splitlines()
    assert len(lines) == 96  # 4 items * 3 cond * 8 samples
    # Simulate crash: drop the last 20 complete rows.
    samples_path.write_text("\n".join(lines[:-20]) + "\n")
    # Torn line at the end should be ignored.
    with samples_path.open("a") as f:
        f.write('{"item_id": "truncated"')
    first, keys_before = load_samples(samples_path)
    assert len(keys_before) == 76

    assert main(["probe", "--dummy", "--out", str(out)]) == 0
    again, keys_after = load_samples(samples_path)
    assert len(keys_after) == 96
    assert (out / "summary.json").exists()
    assert (out / "progress.json").exists()
    assert len(again) == 96
    # Keys from the first 76 survive.
    assert keys_before <= keys_after


def test_fresh_renames_old_log(tmp_path: Path):
    out = tmp_path / "run"
    assert main(["probe", "--dummy", "--out", str(out)]) == 0
    assert main(["probe", "--dummy", "--out", str(out), "--fresh"]) == 0
    assert (out / "samples.jsonl.bak").exists()
    _, keys = load_samples(out / "samples.jsonl")
    assert len(keys) == 96
