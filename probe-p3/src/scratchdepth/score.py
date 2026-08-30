"""Residue match after ANS (or last number). Same matcher for direct/cot."""

from __future__ import annotations

import re

_NUM = re.compile(r"-?\d+")


def extract_after_ans(text: str) -> str | None:
    upper = text.upper()
    if "ANS" in upper:
        i = upper.rfind("ANS")
        tail = text[i + 3 :]
        m = _NUM.search(tail)
        if m:
            return m.group(0)
    if "####" in text:
        tail = text.rsplit("####", 1)[-1]
        m = _NUM.search(tail)
        if m:
            return m.group(0)
    return None


def last_number(text: str) -> str | None:
    found = _NUM.findall(text)
    return found[-1] if found else None


def is_correct(text: str, gold: int, mod: int = 17) -> bool:
    pred = extract_after_ans(text) or last_number(text)
    if pred is None:
        return False
    try:
        return int(pred) % mod == int(gold) % mod
    except ValueError:
        return False
