"""Build probe items. Network/HF only when not dummy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from accesstrap.prompts import format_identities
from accesstrap.score import gold_intermediates, perturb_value


@dataclass
class ProbeItem:
    item_id: str
    split: str  # math | qa
    question: str
    gold: str
    gold_access: str
    distractor_access: str

    def to_dict(self) -> dict:
        return asdict(self)


def dummy_items() -> list[ProbeItem]:
    """Tiny fixtures so the pipeline runs without HuggingFace."""
    return [
        ProbeItem(
            item_id="math-dummy-0",
            split="math",
            question=(
                "Natalia sold 48 clips in April and half as many in May. "
                "How many clips did she sell altogether?"
            ),
            gold="72",
            gold_access=format_identities([("48/2", "24")]),
            distractor_access=format_identities([("48/2", "25")]),
        ),
        ProbeItem(
            item_id="math-dummy-1",
            split="math",
            question="A bakery made 12 cakes and then 3 more trays of 4 cakes. How many cakes in total?",
            gold="24",
            gold_access=format_identities([("3*4", "12")]),
            distractor_access=format_identities([("3*4", "13")]),
        ),
        ProbeItem(
            item_id="qa-dummy-0",
            split="qa",
            question="Which city is the capital of the country where the Eiffel Tower stands?",
            gold="Paris",
            gold_access="The Eiffel Tower is in France. Paris is the capital of France.",
            distractor_access="The Eiffel Tower inspired a replica in Las Vegas. Nevada's capital is Carson City.",
        ),
        ProbeItem(
            item_id="qa-dummy-1",
            split="qa",
            question="Who wrote the play that features the character Hamlet, Prince of Denmark?",
            gold="William Shakespeare",
            gold_access="Hamlet is a tragedy by William Shakespeare.",
            distractor_access="Goethe wrote Faust. Ibsen wrote A Doll's House.",
        ),
    ]


def load_items(*, n_math: int, n_qa: int, dummy: bool) -> list[ProbeItem]:
    if dummy:
        items = dummy_items()
        math_items = [x for x in items if x.split == "math"][:n_math]
        qa_items = [x for x in items if x.split == "qa"][:n_qa]
        return math_items + qa_items
    out: list[ProbeItem] = []
    if n_math:
        out.extend(load_gsm8k(n_math))
    if n_qa:
        out.extend(load_hotpot(n_qa))
    return out


def load_gsm8k(n: int) -> list[ProbeItem]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    items: list[ProbeItem] = []
    for i, row in enumerate(ds):
        question = row["question"]
        answer = row["answer"]
        mids = gold_intermediates(answer)
        if len(mids) < 1:
            continue
        from accesstrap.score import extract_gsm8k_gold_number

        gold = extract_gsm8k_gold_number(answer)
        if not gold:
            continue
        gold_block = format_identities(mids)
        dist_block = format_identities([(e, perturb_value(v)) for e, v in mids])
        items.append(
            ProbeItem(
                item_id=f"gsm8k-{i}",
                split="math",
                question=question,
                gold=gold,
                gold_access=gold_block,
                distractor_access=dist_block,
            )
        )
        if len(items) >= n:
            break
    if len(items) < n:
        raise RuntimeError(f"only found {len(items)} GSM8K items with intermediates, need {n}")
    return items


def load_hotpot(n: int) -> list[ProbeItem]:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    items: list[ProbeItem] = []
    for i, row in enumerate(ds):
        question = row["question"]
        gold = row["answer"]
        if not question or not gold:
            continue
        gold_block, dist_block = _hotpot_blocks(row)
        if not gold_block or not dist_block:
            continue
        items.append(
            ProbeItem(
                item_id=f"hotpot-{i}",
                split="qa",
                question=question,
                gold=gold,
                gold_access=gold_block,
                distractor_access=dist_block,
            )
        )
        if len(items) >= n:
            break
    if len(items) < n:
        raise RuntimeError(f"only found {len(items)} Hotpot items, need {n}")
    return items


def _hotpot_blocks(row: dict) -> tuple[str, str]:
    """Gold supporting sentences vs non-supporting paragraphs from the same context."""
    context = row["context"]
    titles = list(context["title"])
    sents_list = list(context["sentences"])
    by_title = {t: sents for t, sents in zip(titles, sents_list, strict=True)}
    gold_lines: list[str] = []
    used: set[tuple[str, int]] = set()
    sf = row["supporting_facts"]
    for title, sent_id in zip(sf["title"], sf["sent_id"], strict=True):
        sents = by_title.get(title)
        if sents is None or sent_id >= len(sents):
            continue
        gold_lines.append(sents[sent_id])
        used.add((title, int(sent_id)))
    dist_lines: list[str] = []
    for title, sents in by_title.items():
        for j, sent in enumerate(sents):
            if (title, j) in used:
                continue
            dist_lines.append(sent)
            if len(dist_lines) >= max(2, len(gold_lines)):
                break
        if len(dist_lines) >= max(2, len(gold_lines)):
            break
    return "\n".join(gold_lines).strip(), "\n".join(dist_lines).strip()
