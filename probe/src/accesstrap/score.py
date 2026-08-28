"""Answer extraction and correctness. No model I/O."""

from __future__ import annotations

import re
import string

GSM8K_CALC_RE = re.compile(r"<<([^=]+)=([^>]+)>>")
FINAL_HASH_RE = re.compile(r"####\s*([^\n]+)")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_gsm8k_calcs(solution: str) -> list[tuple[str, str]]:
    """Return (expr, value) pairs in order from a GSM8K gold solution."""
    out: list[tuple[str, str]] = []
    for expr, val in GSM8K_CALC_RE.findall(solution):
        out.append((expr.strip(), val.strip()))
    return out


def gold_intermediates(solution: str) -> list[tuple[str, str]]:
    """All calculator identities except the last (no final-answer leak)."""
    calcs = parse_gsm8k_calcs(solution)
    if len(calcs) < 2:
        return []
    return calcs[:-1]


def perturb_value(value: str) -> str:
    """Deterministic numeric distractor, never equal to gold."""
    raw = value.strip()
    try:
        if "." in raw:
            x = float(raw)
            y = x + 1.0 if x >= 0 else x - 1.0
            return str(y)
        x = int(raw)
        y = x + 1 if x >= 0 else x - 1
        return str(y)
    except ValueError:
        return raw + "_wrong"


def extract_gsm8k_gold_number(answer_field: str) -> str:
    """GSM8K `answer` field ends with `#### 72`."""
    m = FINAL_HASH_RE.search(answer_field)
    if m:
        return _canonical_number(m.group(1))
    nums = NUMBER_RE.findall(answer_field.replace(",", ""))
    if not nums:
        return ""
    return _canonical_number(nums[-1])


def extract_predicted_number(text: str) -> str:
    text = _strip_think(text)
    m = FINAL_HASH_RE.search(text)
    if m:
        return _canonical_number(m.group(1))
    nums = NUMBER_RE.findall(text.replace(",", ""))
    if not nums:
        return ""
    return _canonical_number(nums[-1])


def gsm8k_correct(prediction: str, gold: str) -> bool:
    return extract_predicted_number(prediction) == _canonical_number(gold)


def normalize_qa(text: str) -> str:
    text = _strip_think(text).lower()
    text = text.replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def qa_correct(prediction: str, gold: str) -> bool:
    pred = normalize_qa(prediction)
    gold_n = normalize_qa(gold)
    if not gold_n:
        return False
    if pred == gold_n:
        return True
    # Last-line fallback: many models put the short answer at the end.
    last = normalize_qa(prediction.strip().splitlines()[-1])
    if last == gold_n:
        return True
    return gold_n in pred and len(gold_n) >= 2


def _canonical_number(raw: str) -> str:
    s = raw.strip().replace(",", "")
    s = s.split()[0] if s.split() else s
    try:
        x = float(s)
        if x.is_integer():
            return str(int(x))
        return ("%g" % x)
    except ValueError:
        return s


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
