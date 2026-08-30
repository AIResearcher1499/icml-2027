"""Deterministic fake F/P/N traces. Dummy never locks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from weighttraces.score import is_correct

MODELS = ("base", "think")
CONDS = ("F", "P", "N")


@dataclass
class ProbeItem:
    item_id: str
    question: str
    gold: str

    def to_dict(self) -> dict:
        return {"item_id": self.item_id, "question": self.question, "gold": self.gold}

    @classmethod
    def from_dict(cls, rec: dict) -> ProbeItem:
        return cls(item_id=rec["item_id"], question=rec["question"], gold=str(rec["gold"]))


@dataclass
class Sample:
    item_id: str
    model: str
    condition: str
    sample_idx: int
    text: str
    n_cot_tokens: int
    cot_text: str = ""
    cot_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "model": self.model,
            "condition": self.condition,
            "sample_idx": self.sample_idx,
            "text": self.text,
            "n_cot_tokens": self.n_cot_tokens,
            "cot_text": self.cot_text,
            "cot_ids": self.cot_ids,
        }

    @classmethod
    def from_dict(cls, rec: dict) -> Sample:
        return cls(
            item_id=rec["item_id"],
            model=rec["model"],
            condition=rec["condition"],
            sample_idx=int(rec["sample_idx"]),
            text=rec["text"],
            n_cot_tokens=int(rec["n_cot_tokens"]),
            cot_text=str(rec.get("cot_text") or ""),
            cot_ids=[int(x) for x in (rec.get("cot_ids") or [])],
        )


def dummy_items() -> list[ProbeItem]:
    # Tiny GSM8K-shaped items. Golds are fixed; dummy traces decide correctness.
    return [
        ProbeItem("gsm-0", "Natalia sold clips to 48 friends.", "72"),
        ProbeItem("gsm-1", "Weng earns $12 an hour.", "10"),
        ProbeItem("gsm-2", "Betty is saving money.", "18"),
        ProbeItem("gsm-3", "Julie is reading a 120-page book.", "36"),
    ]


def _ok(model: str, cond: str, item_id: str, sample_idx: int) -> bool:
    """LIVE-shaped dummy rates so the pipeline can be unit-tested.

    think F ≫ base F; think N stays close to F; base N collapses.
    Dummy verdict is still DUMMY_SKIP.
    """
    h = int(hashlib.sha256(f"{model}:{cond}:{item_id}:{sample_idx}".encode()).hexdigest(), 16)
    r = h % 16
    if model == "think":
        if cond == "F":
            return r != 0
        if cond == "P":
            return r not in {0, 1}
        return r not in {0, 1, 2}
    if cond == "F":
        return r < 12
    if cond == "P":
        return r < 8
    return r < 6


def dummy_sample(item: ProbeItem, model: str, cond: str, sample_idx: int) -> Sample:
    ok = _ok(model, cond, item.item_id, sample_idx)
    ans = item.gold if ok else str(int(item.gold) + 1)
    n_cot = {"F": 40, "P": 20, "N": 0}[cond]
    if cond == "N":
        text = f"#### {ans}"
    elif cond == "P":
        text = f"<think>half the steps</think>\n#### {ans}"
    else:
        text = f"<think>full chain of thought with several steps</think>\n#### {ans}"
    if model == "base" and cond != "N":
        text = f"Let's add. Therefore #### {ans}"
    return Sample(item.item_id, model, cond, sample_idx, text, n_cot)


def assert_scorer(item: ProbeItem, sample: Sample) -> bool:
    return is_correct(sample.text, item.gold)
