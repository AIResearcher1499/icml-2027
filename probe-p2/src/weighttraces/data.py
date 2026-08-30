"""GSM8K test items. Shuffle seed frozen in KILL-p2.md."""

from __future__ import annotations

from weighttraces.dummy import ProbeItem, dummy_items
from weighttraces.score import last_number, normalize_num


def gold_from_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        tail = answer.rsplit("####", 1)[-1]
        n = last_number(tail.replace(",", ""))
        if n is not None:
            return normalize_num(n)
    n = last_number(answer.replace(",", ""))
    if n is None:
        raise ValueError("gsm8k row has no numeric gold")
    return normalize_num(n)


def load_gsm8k(n_items: int, seed: int) -> list[ProbeItem]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed)
    items: list[ProbeItem] = []
    for i, row in enumerate(ds):
        if i >= n_items:
            break
        items.append(
            ProbeItem(
                item_id=f"gsm-{i:03d}",
                question=str(row["question"]).strip(),
                gold=gold_from_gsm8k_answer(str(row["answer"])),
            )
        )
    if len(items) < n_items:
        raise RuntimeError(f"gsm8k test shorter than n_items={n_items}")
    return items


def apply_shard(items: list[ProbeItem], shard: str | None) -> list[ProbeItem]:
    if not shard:
        return items
    if "/" not in shard:
        raise ValueError(f"shard must look like 0/2, got {shard!r}")
    raw_i, raw_n = shard.split("/", 1)
    i, n = int(raw_i), int(raw_n)
    if n <= 0 or not (0 <= i < n):
        raise ValueError(f"invalid shard {shard!r}")
    start = i * len(items) // n
    end = (i + 1) * len(items) // n
    return items[start:end]


def load_items(
    *,
    dummy: bool,
    n_items: int,
    seed: int,
    items_path=None,
) -> list[ProbeItem]:
    if dummy:
        return dummy_items()[:n_items]
    if items_path is not None:
        import json
        from pathlib import Path

        raw = json.loads(Path(items_path).read_text())
        recs = list(raw.values()) if isinstance(raw, dict) else raw
        return [ProbeItem.from_dict(r) for r in recs]
    return load_gsm8k(n_items, seed)
