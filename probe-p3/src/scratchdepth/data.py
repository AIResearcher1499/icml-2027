"""Synthetic serial product / parallel sum items. Seed frozen in KILL-p3.md."""

from __future__ import annotations

import random

from scratchdepth.dummy import (
    DUMMY_N_PER_CELL,
    HORIZONS,
    MOD,
    TASKS,
    ProbeItem,
    dummy_items,
)


def gold_of(task: str, operands: list[int], mod: int = MOD) -> int:
    if not operands:
        raise ValueError("empty operands")
    if task == "serial":
        acc = 1
        for a in operands:
            acc = (acc * a) % mod
        return acc
    if task == "parallel":
        return sum(operands) % mod
    raise ValueError(f"unknown task {task!r}")


def make_item(task: str, horizon: int, *, seed: int, idx: int, mod: int = MOD) -> ProbeItem:
    rng = random.Random(f"{seed}:{task}:{horizon}:{idx}")
    operands = [rng.randint(1, mod - 1) for _ in range(horizon)]
    gold = gold_of(task, operands, mod)
    return ProbeItem(
        item_id=f"{task}-h{horizon:02d}-{idx:03d}",
        task=task,
        horizon=horizon,
        operands=operands,
        gold=gold,
    )


def generate_items(*, n_per_cell: int, seed: int, dummy: bool) -> list[ProbeItem]:
    if dummy:
        return dummy_items()
    items: list[ProbeItem] = []
    for task in TASKS:
        for h in HORIZONS:
            for i in range(n_per_cell):
                items.append(make_item(task, h, seed=seed, idx=i))
    return items


def load_items(
    *,
    dummy: bool,
    n_per_cell: int = DUMMY_N_PER_CELL,
    seed: int = 0,
    items_path=None,
) -> list[ProbeItem]:
    if dummy:
        return dummy_items()
    if items_path is not None:
        import json
        from pathlib import Path

        raw = json.loads(Path(items_path).read_text())
        recs = list(raw.values()) if isinstance(raw, dict) else raw
        return [ProbeItem.from_dict(r) for r in recs]
    return generate_items(n_per_cell=n_per_cell, seed=seed, dummy=False)
