"""Append-only sample log so a crash can resume."""

from __future__ import annotations

import json
import os
from pathlib import Path

from weighttraces.dummy import Sample

SAMPLES_NAME = "samples.jsonl"
ITEMS_NAME = "items.json"
PROGRESS_NAME = "progress.json"


def sample_key(item_id: str, model: str, condition: str, sample_idx: int) -> tuple[str, str, str, int]:
    return (item_id, model, condition, int(sample_idx))


def truncate_torn_tail(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        last = text.rstrip("\n").rsplit("\n", 1)[-1]
        try:
            json.loads(last)
            return
        except json.JSONDecodeError:
            pass
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            print("dropping torn jsonl line", flush=True)
            continue
        kept.append(raw)
    path.write_text(("\n".join(kept) + ("\n" if kept else "")), encoding="utf-8")


def append_sample(path: Path, sample: Sample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_samples(path: Path) -> tuple[list[Sample], set[tuple[str, str, str, int]]]:
    samples: list[Sample] = []
    done: set[tuple[str, str, str, int]] = set()
    if not path.exists():
        return samples, done
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                s = Sample.from_dict(rec)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                skipped += 1
                print(f"skipping corrupt {path.name} line {line_no}", flush=True)
                continue
            key = sample_key(s.item_id, s.model, s.condition, s.sample_idx)
            if key in done:
                continue
            done.add(key)
            samples.append(s)
    if skipped:
        print(f"checkpoint: skipped {skipped} corrupt line(s)", flush=True)
    return samples, done


def write_progress(path: Path, *, done: int, total: int, last: str) -> None:
    payload = {"done": done, "total": total, "last": last}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
