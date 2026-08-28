"""Reduce samples → item stats → summary + verdict."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from accesstrap.connectives import MIN_LEXICAL_HITS
from accesstrap.data import ProbeItem
from accesstrap.entropy import choose_entropy_rule, lexical_entropies, topk_entropies
from accesstrap.generate import Sample, access_block
from accesstrap.metrics import unbiased_pass_at_k, venn_counts
from accesstrap.score import gsm8k_correct, qa_correct
from accesstrap.verdict import decide


KS_CANDIDATES = (1, 8, 32)


def is_correct(item: ProbeItem, text: str) -> bool:
    if item.split == "math":
        return gsm8k_correct(text, item.gold)
    return qa_correct(text, item.gold)


def aggregate(
    items: list[ProbeItem],
    samples: list[Sample],
    n_samples: int,
) -> dict[str, Any]:
    by_item = {it.item_id: it for it in items}
    grouped: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for s in samples:
        grouped[(s.item_id, s.condition)].append(s)

    lexical_hits = {"A": 0, "B": 0, "C": 0}
    per_item: list[dict[str, Any]] = []

    for item in items:
        row: dict[str, Any] = {"item_id": item.item_id, "split": item.split, "gold": item.gold}
        for cond in ("A", "B", "C"):
            ss = grouped.get((item.item_id, cond), [])
            corrects = [is_correct(item, s.text) for s in ss]
            c = sum(1 for x in corrects if x)
            n = len(ss)
            block = access_block(item, cond)
            lex: list[float] = []
            top: list[float] = []
            for s in ss:
                lex.extend(lexical_entropies(s.tokens, s.entropies, block))
                top.extend(topk_entropies(s.entropies))
            lexical_hits[cond] += len(lex)
            row[cond] = {
                "n": n,
                "c": c,
                "pass": {
                    str(k): unbiased_pass_at_k(n, c, k)
                    for k in KS_CANDIDATES
                    if k <= n_samples
                },
                "solved": c >= 1,
                "lexical_ents": lex,
                "top20_ents": top,
            }
        per_item.append(row)

    rule = choose_entropy_rule(lexical_hits)
    band = _primary_band(per_item, n_samples)
    band_rows = [r for r in per_item if r["item_id"] in band]

    def mean_ent(cond: str, rows: list[dict[str, Any]]) -> float | None:
        key = "lexical_ents" if rule == "lexical" else "top20_ents"
        vals: list[float] = []
        for r in rows:
            vals.extend(r[cond][key])
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    ks = [k for k in KS_CANDIDATES if k <= n_samples]
    mean_pass = {
        cond: {
            str(k): _mean([r[cond]["pass"][str(k)] for r in band_rows])
            for k in ks
        }
        for cond in ("A", "B", "C")
    }

    venn = venn_counts(
        (r["A"]["solved"] for r in band_rows),
        (r["B"]["solved"] for r in band_rows),
    )
    venn_ac = venn_counts(
        (r["A"]["solved"] for r in band_rows),
        (r["C"]["solved"] for r in band_rows),
    )

    summary = {
        "n_items_pool": len(per_item),
        "n_items_primary_band": len(band_rows),
        "primary_band_rule": "no-access pass@min(8,n) >= 0.5",
        "n_samples": n_samples,
        "entropy_rule": rule,
        "lexical_hits": lexical_hits,
        "min_lexical_hits": MIN_LEXICAL_HITS,
        "mean_pass_primary": mean_pass,
        "venn_AB": venn,
        "venn_AC": venn_ac,
        "entropy": {
            "A": mean_ent("A", band_rows),
            "B": mean_ent("B", band_rows),
            "C": mean_ent("C", band_rows),
        },
        "band_item_ids": sorted(band),
    }
    summary["verdict"] = decide(summary)
    return {"summary": summary, "items": per_item, "item_index": {it.item_id: it.to_dict() for it in items}}


def _primary_band(per_item: list[dict[str, Any]], n_samples: int) -> set[str]:
    k = min(8, n_samples)
    band: set[str] = set()
    for r in per_item:
        if r["A"]["pass"].get(str(k), 0.0) >= 0.5:
            band.add(r["item_id"])
    return band


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0
