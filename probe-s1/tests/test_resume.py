from pathlib import Path

from committrap.checkpoint import load_events
from committrap.cli import main


def test_resume_skips_finished_keys(tmp_path: Path):
    out = tmp_path / "run"
    assert main(["probe", "--dummy", "--out", str(out)]) == 0
    samples_path = out / "samples.jsonl"
    lines = samples_path.read_text().splitlines()
    assert len(lines) > 10
    samples_path.write_text("\n".join(lines[:-8]) + "\n")
    with samples_path.open("a") as f:
        f.write('{"kind": "truncated"')
    _first, keys_before = load_events(samples_path)

    assert main(["probe", "--dummy", "--out", str(out)]) == 0
    _again, keys_after = load_events(samples_path)
    assert keys_before <= keys_after
    assert (out / "summary.json").exists()
    assert (out / "progress.json").exists()


def test_fresh_renames_old_log(tmp_path: Path):
    out = tmp_path / "run"
    assert main(["probe", "--dummy", "--out", str(out)]) == 0
    assert main(["probe", "--dummy", "--out", str(out), "--fresh"]) == 0
    assert (out / "samples.jsonl.bak").exists()
