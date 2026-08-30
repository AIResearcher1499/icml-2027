"""CLI: scratchdepth probe | verdict. Full train is not implemented (dummy only)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scratchdepth.checkpoint import (
    ITEMS_NAME,
    PROGRESS_NAME,
    SAMPLES_NAME,
    append_sample,
    load_samples,
    sample_key,
    truncate_torn_tail,
    write_progress,
)
from scratchdepth.data import load_items
from scratchdepth.dummy import CONDS, DEPTHS, HORIZONS, TASKS, ProbeItem, Sample, dummy_sample
from scratchdepth.metrics import mean_acc
from scratchdepth.score import is_correct
from scratchdepth.verdict import decide

SUMMARY_NAME = "summary.json"
VERDICT_NAME = "verdict.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scratchdepth")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="run the frozen 72h P3 probe (dummy only in this package)")
    p.add_argument("--dummy", action="store_true", help="no train; deterministic fakes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--fresh",
        action="store_true",
        help="rename existing samples.jsonl and start a new log",
    )

    v = sub.add_parser("verdict", help="recompute verdict from a saved summary.json")
    v.add_argument("summary", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "probe":
        return run_probe(args)
    return run_verdict(args.summary)


def run_probe(args: argparse.Namespace) -> int:
    dummy = bool(args.dummy)
    if not dummy:
        print(
            "full train is not implemented in this package; pass --dummy. "
            "Do not train 20M-300M or load 0.6B here.",
            flush=True,
        )
        return 2

    out_dir = args.out or _default_out()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / SAMPLES_NAME
    items_path = out_dir / ITEMS_NAME

    if args.fresh and samples_path.exists():
        bak = samples_path.with_suffix(samples_path.suffix + ".bak")
        samples_path.replace(bak)
        print(f"--fresh: moved old log to {bak}", flush=True)

    items = _load_or_create_items(items_path, dummy=True)
    truncate_torn_tail(samples_path)
    existing, done_keys = load_samples(samples_path)
    total = len(items) * len(DEPTHS) * len(CONDS)
    remaining = total - len(done_keys)
    print(
        f"run dir {out_dir} checkpointed={len(done_keys)} remaining={remaining} total={total}",
        flush=True,
    )

    by_key: dict[tuple[str, int, str], Sample] = {
        sample_key(s.item_id, s.depth, s.condition): s for s in existing
    }
    n_new = 0

    def _emit(s: Sample) -> None:
        nonlocal n_new
        append_sample(samples_path, s)
        key = sample_key(s.item_id, s.depth, s.condition)
        done_keys.add(key)
        by_key[key] = s
        n_new += 1
        write_progress(
            out_dir / PROGRESS_NAME,
            done=len(done_keys),
            total=total,
            last=f"{s.item_id} L={s.depth} {s.condition}",
        )

    for item in items:
        for depth in DEPTHS:
            for cond in CONDS:
                key = sample_key(item.item_id, depth, cond)
                if key in by_key:
                    continue
                _emit(dummy_sample(item, depth, cond))

    samples, _ = load_samples(samples_path)
    summary = aggregate(items, samples, dummy=True)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / VERDICT_NAME).write_text(_verdict_md(summary) + "\n")
    print(json.dumps(summary["verdict"], indent=2))
    print(f"wrote {out_dir} new_samples={n_new}")
    return 0


def _load_or_create_items(items_path: Path, *, dummy: bool) -> list[ProbeItem]:
    if items_path.exists():
        raw = json.loads(items_path.read_text())
        recs = list(raw.values()) if isinstance(raw, dict) else raw
        items = [ProbeItem.from_dict(r) for r in recs]
        print(f"reloaded {len(items)} items from {items_path.name}", flush=True)
        return items
    items = load_items(dummy=dummy)
    items_path.write_text(
        json.dumps({it.item_id: it.to_dict() for it in items}, indent=2, ensure_ascii=False) + "\n"
    )
    return items


def run_verdict(summary_path: Path) -> int:
    summary = json.loads(summary_path.read_text())
    dummy = bool(summary.get("dummy"))
    summary["verdict"] = decide(summary, dummy=dummy)
    print(json.dumps(summary["verdict"], indent=2))
    return 0


def aggregate(items: list[ProbeItem], samples: list[Sample], *, dummy: bool) -> dict:
    gold = {it.item_id: it.gold for it in items}
    meta = {it.item_id: it for it in items}
    buckets: dict[tuple[str, int, int, str], list[bool]] = defaultdict(list)

    for s in samples:
        item = meta[s.item_id]
        ok = s.correct if s.correct is not None else is_correct(s.text, gold[s.item_id])
        buckets[(item.task, s.depth, item.horizon, s.condition)].append(ok)

    acc: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for task in TASKS:
        acc[task] = {}
        for depth in DEPTHS:
            acc[task][str(depth)] = {}
            for h in HORIZONS:
                acc[task][str(depth)][str(h)] = {
                    c: mean_acc(buckets[(task, depth, h, c)]) for c in CONDS
                }

    summary = {
        "dummy": dummy,
        "n_items": len(items),
        "depths": list(DEPTHS),
        "horizons": list(HORIZONS),
        "acc": acc,
        "note": "copy-with-offset is log-only and is not in this dummy pool.",
    }
    summary["verdict"] = decide(summary, dummy=dummy)
    return summary


def _default_out() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parents[2] / "runs" / f"{stamp}-dummy"


def _verdict_md(summary: dict) -> str:
    v = summary["verdict"]
    lines = [
        f"# P3 probe verdict: {v['decision']}",
        "",
        v["note"],
        "",
        f"- n_items: {summary['n_items']}",
        f"- drop_h serial L=2 direct: {v['drop_h_serial_L2_direct']:.4f}",
        f"- restore serial L=2 h=16: {v['restore_serial_L2_h16']:.4f}",
        f"- drop_h parallel L=2 direct: {v['drop_h_parallel_L2_direct']:.4f}",
        f"- serial_collapse_live: {v['serial_collapse_live']}",
        f"- cot_restore_live: {v['cot_restore_live']}",
        f"- not_length_live: {v['not_length_live']}",
        "",
        f"- acc: {json.dumps(summary['acc'])}",
    ]
    if v["kill_reasons"]:
        lines.append("- kill_reasons:")
        lines.extend(f"  - {r}" for r in v["kill_reasons"])
    return "\n".join(lines)
