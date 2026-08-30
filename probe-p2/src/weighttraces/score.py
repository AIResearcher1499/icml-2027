"""GSM8K-style numeric match. Same matcher for F/P/N."""

from __future__ import annotations

import re


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def extract_after_hash(text: str) -> str | None:
    if "####" in text:
        tail = text.rsplit("####", 1)[-1]
        m = _NUM.search(tail)
        if m:
            return m.group(0)
    return None


def last_number(text: str) -> str | None:
    found = _NUM.findall(text)
    return found[-1] if found else None


def normalize_num(s: str) -> str:
    s = s.strip().replace(",", "")
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
        return str(v)
    except ValueError:
        return s


def answer_span(text: str) -> str:
    """Score the answer, not numbers inside a think block."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def is_correct(text: str, gold: str) -> bool:
    span = answer_span(text)
    pred = extract_after_hash(span) or last_number(span)
    if pred is None:
        return False
    return normalize_num(pred) == normalize_num(str(gold))
