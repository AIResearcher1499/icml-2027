"""Append-only event log so a crash can resume."""

from __future__ import annotations

import json
import os
from pathlib import Path

from committrap.dummy import Event, event_key

SAMPLES_NAME = "samples.jsonl"
ITEMS_NAME = "items.json"
PROGRESS_NAME = "progress.json"


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


def append_event(path: Path, event: Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_events(path: Path) -> tuple[list[Event], set[tuple]]:
    events: list[Event] = []
    done: set[tuple] = set()
    if not path.exists():
        return events, done
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                e = Event.from_dict(rec)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                skipped += 1
                print(f"skipping corrupt {path.name} line {line_no}", flush=True)
                continue
            key = event_key(e)
            if key in done:
                continue
            done.add(key)
            events.append(e)
    if skipped:
        print(f"checkpoint: skipped {skipped} corrupt line(s)", flush=True)
    return events, done


def write_progress(path: Path, *, done: int, total: int, last: str) -> None:
    payload = {"done": done, "total": total, "last": last}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
