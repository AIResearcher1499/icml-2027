"""Deterministic fake direct/CoT traces. Dummy never locks."""

from __future__ import annotations

from dataclasses import dataclass, field

from scratchdepth.score import is_correct

MOD = 17
TASKS = ("serial", "parallel")
HORIZONS = (4, 8, 16)
DEPTHS = (2, 4, 8)
CONDS = ("direct", "cot")
DUMMY_N_PER_CELL = 4


@dataclass
class ProbeItem:
    item_id: str
    task: str
    horizon: int
    operands: list[int]
    gold: int

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "task": self.task,
            "horizon": self.horizon,
            "operands": list(self.operands),
            "gold": self.gold,
        }

    @classmethod
    def from_dict(cls, rec: dict) -> ProbeItem:
        return cls(
            item_id=rec["item_id"],
            task=str(rec["task"]),
            horizon=int(rec["horizon"]),
            operands=[int(x) for x in rec["operands"]],
            gold=int(rec["gold"]),
        )


@dataclass
class Sample:
    item_id: str
    depth: int
    condition: str
    text: str
    correct: bool | None = None
    n_scratch_tokens: int = 0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "depth": self.depth,
            "condition": self.condition,
            "text": self.text,
            "correct": self.correct,
            "n_scratch_tokens": self.n_scratch_tokens,
        }

    @classmethod
    def from_dict(cls, rec: dict) -> Sample:
        return cls(
            item_id=rec["item_id"],
            depth=int(rec["depth"]),
            condition=str(rec["condition"]),
            text=str(rec["text"]),
            correct=rec.get("correct"),
            n_scratch_tokens=int(rec.get("n_scratch_tokens") or 0),
        )


def dummy_items() -> list[ProbeItem]:
    from scratchdepth.data import make_item

    items: list[ProbeItem] = []
    for task in TASKS:
        for h in HORIZONS:
            for i in range(DUMMY_N_PER_CELL):
                items.append(make_item(task, h, seed=1000 + i, idx=i))
    return items


def _target_p(task: str, depth: int, horizon: int, cond: str) -> float:
    """LIVE-shaped dummy rates so the pipeline can be unit-tested.

    Dummy verdict is still DUMMY_SKIP.
    """
    if task == "serial" and cond == "direct":
        base = {4: 0.90, 8: 0.55, 16: 0.20}[horizon]
        if depth == 4:
            return min(0.95, base + 0.25)
        if depth == 8:
            return min(0.97, base + 0.45)
        return base
    if task == "serial" and cond == "cot":
        if depth == 2:
            return {4: 0.95, 8: 0.90, 16: 0.85}[horizon]
        return 0.95
    if task == "parallel" and cond == "direct":
        return 0.90 if depth == 2 else 0.95
    return 0.95


def _ok(task: str, depth: int, horizon: int, cond: str, item_id: str) -> bool:
    p = _target_p(task, depth, horizon, cond)
    idx = int(item_id.rsplit("-", 1)[-1])
    n_ok = int(round(p * DUMMY_N_PER_CELL))
    return idx < n_ok


def dummy_sample(item: ProbeItem, depth: int, cond: str) -> Sample:
    ok = _ok(item.task, depth, item.horizon, cond, item.item_id)
    ans = item.gold if ok else (item.gold + 1) % MOD
    if cond == "direct":
        text = f"ANS {ans}"
        n_scratch = 0
    else:
        text = f"running {item.operands[0]} ... ANS {ans}"
        n_scratch = max(item.horizon - 1, 0)
    return Sample(item.item_id, depth, cond, text, correct=ok, n_scratch_tokens=n_scratch)


def assert_scorer(item: ProbeItem, sample: Sample) -> bool:
    return is_correct(sample.text, item.gold)
