"""Deterministic fake F / probes / reforks. Dummy never locks."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

PROBE_STRIDE = 256


@dataclass
class ProbeItem:
    item_id: str
    question: str
    gold: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, rec: dict) -> ProbeItem:
        return cls(item_id=rec["item_id"], question=rec["question"], gold=str(rec["gold"]))


@dataclass
class Event:
    kind: str
    item_id: str
    sample_idx: int
    text: str
    n_cot_tokens: int = 0
    cot_ids: list[int] = field(default_factory=list)
    cut: int | None = None
    H: float | None = None
    refork_idx: int | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "item_id": self.item_id,
            "sample_idx": self.sample_idx,
            "text": self.text,
            "n_cot_tokens": self.n_cot_tokens,
            "cot_ids": self.cot_ids,
            "cut": self.cut,
            "H": self.H,
            "refork_idx": self.refork_idx,
        }

    @classmethod
    def from_dict(cls, rec: dict) -> Event:
        return cls(
            kind=rec["kind"],
            item_id=rec["item_id"],
            sample_idx=int(rec["sample_idx"]),
            text=rec["text"],
            n_cot_tokens=int(rec.get("n_cot_tokens") or 0),
            cot_ids=[int(x) for x in (rec.get("cot_ids") or [])],
            cut=None if rec.get("cut") is None else int(rec["cut"]),
            H=None if rec.get("H") is None else float(rec["H"]),
            refork_idx=None if rec.get("refork_idx") is None else int(rec["refork_idx"]),
        )


def event_key(e: Event) -> tuple:
    if e.kind == "F":
        return ("F", e.item_id, e.sample_idx)
    if e.kind == "probe":
        return ("probe", e.item_id, e.sample_idx, int(e.cut or 0))
    if e.kind == "refork":
        return ("refork", e.item_id, e.sample_idx, int(e.cut or 0), int(e.refork_idx or 0))
    raise ValueError(e.kind)


def dummy_items() -> list[ProbeItem]:
    return [
        ProbeItem("gsm-000", "Natalia sold clips.", "72"),
        ProbeItem("gsm-001", "Weng earns $12 an hour.", "10"),
        ProbeItem("gsm-002", "Betty is saving money.", "18"),
        ProbeItem("gsm-003", "Julie is reading a 120-page book.", "36"),
    ]


def dummy_F(item: ProbeItem, sample_idx: int) -> Event:
    return Event(
        kind="F",
        item_id=item.item_id,
        sample_idx=sample_idx,
        text=f"<think>long chain</think>\nThe answer is #### {int(item.gold) + 1}",
        n_cot_tokens=512,
        cot_ids=[],
    )


def dummy_probe(item: ProbeItem, sample_idx: int, cut: int) -> Event:
    # First probe high H, second low H → t* at second cut; trunc wrong.
    h = 1.2 if cut <= PROBE_STRIDE else 0.2
    return Event(
        kind="probe",
        item_id=item.item_id,
        sample_idx=sample_idx,
        text=f"The answer is #### {int(item.gold) + 1}",
        n_cot_tokens=512,
        cut=cut,
        H=h,
    )


def dummy_refork(item: ProbeItem, sample_idx: int, cut: int, refork_idx: int) -> Event:
    h = int(hashlib.sha256(f"{item.item_id}:{sample_idx}:{cut}:{refork_idx}".encode()).hexdigest(), 16)
    ok = (h % 4) != 0
    ans = item.gold if ok else str(int(item.gold) + 1)
    return Event(
        kind="refork",
        item_id=item.item_id,
        sample_idx=sample_idx,
        text=f"<think>branch</think>\nThe answer is #### {ans}",
        cut=cut,
        refork_idx=refork_idx,
    )
