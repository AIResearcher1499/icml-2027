"""CoT span helpers. Token-id cuts happen in generate.py with the real tokenizer."""

from __future__ import annotations


def extract_think_cot(text: str) -> str:
    if "</think>" in text:
        head = text.split("</think>", 1)[0]
        if "<think>" in head:
            return head.split("<think>", 1)[1]
        return head
    if "<think>" in text:
        return text.split("<think>", 1)[1]
    return text


def extract_base_cot(text: str) -> str:
    if "####" in text:
        return text.split("####", 1)[0]
    return text


def extract_cot_text(text: str, arm: str) -> str:
    if arm == "think":
        return extract_think_cot(text)
    if arm == "base":
        return extract_base_cot(text)
    raise ValueError(arm)


def prefix_token_count(n_cot_tokens: int) -> int:
    """floor(0.5 * n). Empty CoT → 0 (P equals N)."""
    if n_cot_tokens <= 0:
        return 0
    return n_cot_tokens // 2


def prefix_ids(cot_ids: list[int]) -> list[int]:
    return cot_ids[: prefix_token_count(len(cot_ids))]
