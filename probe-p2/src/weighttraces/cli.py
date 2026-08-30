"""CLI: weighttraces probe | init-items | merge | verdict."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from weighttraces.checkpoint import (
    ITEMS_NAME,
    PROGRESS_NAME,
    SAMPLES_NAME,
    append_sample,
    load_samples,
    sample_key,
    truncate_torn_tail,
    write_progress,
)
from weighttraces.data import apply_shard, load_items
from weighttraces.dummy import CONDS, MODELS, ProbeItem, Sample, dummy_sample
from weighttraces.metrics import mean_acc, unbiased_pass_at_k
from weighttraces.score import is_correct
from weighttraces.verdict import decide

SUMMARY_NAME = "summary.json"
VERDICT_NAME = "verdict.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="weighttraces")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="run the frozen 72h P2 probe")
    p.add_argument("--dummy", action="store_true", help="no model; deterministic fakes")
    p.add_argument("--smoke", action="store_true", help="0.6B, n=2, 8 items; never locks")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--arm", choices=("both", "base", "think"), default="both")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--n-items", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--shard", default=None, help="item slice, e.g. 0/2")
    p.add_argument("--items", type=Path, default=None, help="frozen items.json")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--fresh",
        action="store_true",
        help="rename existing samples.jsonl and start a new log",
    )

    ini = sub.add_parser("init-items", help="freeze GSM8K items.json (CPU)")
    ini.add_argument("--out", type=Path, required=True)
    ini.add_argument("--n-items", type=int, default=80)
    ini.add_argument("--seed", type=int, default=0)

    m = sub.add_parser("merge", help="merge shard run dirs into one verdict")
    m.add_argument("shards", nargs="+", type=Path)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--n-samples", type=int, default=8)
    m.add_argument("--n-items", type=int, default=80)

    v = sub.add_parser("verdict", help="recompute verdict from a saved summary.json")
    v.add_argument("summary", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "probe":
        return run_probe(args)
    if args.cmd == "init-items":
        return run_init_items(args)
    if args.cmd == "merge":
        return run_merge(args)
    return run_verdict(args.summary)


def run_init_items(args: argparse.Namespace) -> int:
    items = load_items(dummy=False, n_items=args.n_items, seed=args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / ITEMS_NAME
    path.write_text(
        json.dumps({it.item_id: it.to_dict() for it in items}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {len(items)} items -> {path}", flush=True)
    return 0


def run_probe(args: argparse.Namespace) -> int:
    dummy = bool(args.dummy)
    smoke = bool(args.smoke)
    if smoke and not dummy:
        args.model = "Qwen/Qwen3-0.6B"
        args.n_samples = 2
        if args.n_items == 80:
            args.n_items = 8
    if dummy:
        args.n_samples = min(args.n_samples, 4)
        args.n_items = min(args.n_items, 4)

    out_dir = args.out or _default_out(dummy=dummy, smoke=smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / SAMPLES_NAME
    items_path = out_dir / ITEMS_NAME

    if args.fresh and samples_path.exists():
        bak = samples_path.with_suffix(samples_path.suffix + ".bak")
        samples_path.replace(bak)
        print(f"--fresh: moved old log to {bak}", flush=True)

    items = _load_or_create_items(items_path, args, dummy)
    truncate_torn_tail(samples_path)
    existing, done_keys = load_samples(samples_path)
    arms = list(MODELS) if args.arm == "both" else [args.arm]
    total = len(items) * len(arms) * len(CONDS) * args.n_samples
    remaining = total - len(done_keys)
    print(
        f"run dir {out_dir} checkpointed={len(done_keys)} remaining={remaining} total={total}",
        flush=True,
    )

    gen = None
    if remaining > 0 and not dummy:
        from weighttraces.generate import HFGenerator

        gen = HFGenerator(args.model, temperature=args.temperature)

    by_key: dict[tuple[str, str, str, int], Sample] = {
        sample_key(s.item_id, s.model, s.condition, s.sample_idx): s for s in existing
    }
    n_new = 0

    def _emit(s: Sample) -> None:
        nonlocal n_new
        append_sample(samples_path, s)
        key = sample_key(s.item_id, s.model, s.condition, s.sample_idx)
        done_keys.add(key)
        by_key[key] = s
        n_new += 1
        write_progress(
            out_dir / PROGRESS_NAME,
            done=len(done_keys),
            total=total,
            last=f"{s.item_id} {s.model} {s.condition}#{s.sample_idx}",
        )
        if n_new % 10 == 0 or len(done_keys) == total:
            print(
                f"[{len(done_keys)}/{total}] {s.item_id} {s.model} {s.condition}#{s.sample_idx}",
                flush=True,
            )

    for item in items:
        for arm in arms:
            for i in range(args.n_samples):
                fk = sample_key(item.item_id, arm, "F", i)
                if fk not in by_key:
                    if dummy:
                        _emit(dummy_sample(item, arm, "F", i))
                    else:
                        _emit(gen.sample_F(item, arm, i, args.seed))
                f_sample = by_key[fk]
                pk = sample_key(item.item_id, arm, "P", i)
                if pk not in by_key:
                    if dummy:
                        _emit(dummy_sample(item, arm, "P", i))
                    else:
                        _emit(gen.sample_P(item, arm, f_sample, i, args.seed))
                nk = sample_key(item.item_id, arm, "N", i)
                if nk not in by_key:
                    if dummy:
                        _emit(dummy_sample(item, arm, "N", i))
                    else:
                        _emit(gen.sample_N(item, arm, i, args.seed))

    samples, _ = load_samples(samples_path)
    complete = (
        (args.shard is None)
        and (set(arms) == set(MODELS))
        and (not dummy)
        and (not smoke)
        and len(samples) == total
    )
    summary = aggregate(
        items,
        samples,
        n_samples=args.n_samples,
        dummy=dummy,
        smoke=smoke,
        shard=args.shard,
        complete=complete,
    )
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / VERDICT_NAME).write_text(_verdict_md(summary) + "\n")
    print(json.dumps(summary["verdict"], indent=2))
    print(f"wrote {out_dir} new_samples={n_new}")
    return 0


def _load_or_create_items(items_path: Path, args: argparse.Namespace, dummy: bool) -> list[ProbeItem]:
    if items_path.exists():
        raw = json.loads(items_path.read_text())
        recs = list(raw.values()) if isinstance(raw, dict) else raw
        items = [ProbeItem.from_dict(r) for r in recs]
        print(f"reloaded {len(items)} items from {items_path.name}", flush=True)
        return items
    src = args.items if not dummy else None
    items = load_items(dummy=dummy, n_items=args.n_items, seed=args.seed, items_path=src)
    if not dummy:
        items = apply_shard(items, args.shard)
    elif args.shard:
        items = apply_shard(items, args.shard)
    items_path.write_text(
        json.dumps({it.item_id: it.to_dict() for it in items}, indent=2, ensure_ascii=False) + "\n"
    )
    return items


def run_merge(args: argparse.Namespace) -> int:
    items: list[ProbeItem] = []
    seen_ids: set[str] = set()
    samples: list[Sample] = []
    seen_keys: set[tuple[str, str, str, int]] = set()
    inferred_n = 0
    for shard in args.shards:
        items_path = shard / ITEMS_NAME
        samples_path = shard / SAMPLES_NAME
        if not items_path.exists() or not samples_path.exists():
            print(f"incomplete shard: {shard}", flush=True)
            return 2
        raw = json.loads(items_path.read_text())
        recs = list(raw.values()) if isinstance(raw, dict) else raw
        for rec in recs:
            it = ProbeItem.from_dict(rec)
            if it.item_id in seen_ids:
                continue
            seen_ids.add(it.item_id)
            items.append(it)
        truncate_torn_tail(samples_path)
        shard_samples, _keys = load_samples(samples_path)
        for s in shard_samples:
            key = sample_key(s.item_id, s.model, s.condition, s.sample_idx)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            samples.append(s)
            inferred_n = max(inferred_n, s.sample_idx + 1)
        prog = shard / PROGRESS_NAME
        if prog.exists():
            p = json.loads(prog.read_text())
            if p.get("done", 0) < p.get("total", 0):
                print(f"warning: shard {shard} incomplete {p}", flush=True)

    dummy_any = False
    smoke_any = False
    for shard in args.shards:
        sp = shard / SUMMARY_NAME
        if sp.exists():
            blob = json.loads(sp.read_text())
            dummy_any = dummy_any or bool(blob.get("dummy"))
            smoke_any = smoke_any or bool(blob.get("smoke"))

    n_samples = args.n_samples or inferred_n
    if len(items) != args.n_items:
        print(f"merge incomplete: items={len(items)} expected {args.n_items}", flush=True)
        return 2
    models_seen = {s.model for s in samples}
    if models_seen != set(MODELS):
        print(f"merge incomplete: models={sorted(models_seen)} expected {list(MODELS)}", flush=True)
        return 2
    expected = len(items) * len(MODELS) * len(CONDS) * n_samples
    if len(seen_keys) != expected:
        print(
            f"merge incomplete: have {len(seen_keys)} samples, expected {expected} "
            f"(items={len(items)} models={len(MODELS)} conds={len(CONDS)} n={n_samples})",
            flush=True,
        )
        return 2

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ITEMS_NAME).write_text(
        json.dumps({it.item_id: it.to_dict() for it in items}, indent=2, ensure_ascii=False) + "\n"
    )
    summary = aggregate(
        items,
        samples,
        n_samples=n_samples,
        dummy=dummy_any,
        smoke=smoke_any,
        shard=None,
        complete=not dummy_any and not smoke_any,
    )
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / VERDICT_NAME).write_text(_verdict_md(summary) + "\n")
    print(json.dumps(summary["verdict"], indent=2))
    print(f"merged {len(args.shards)} shards -> {out_dir}")
    return 0


def run_verdict(summary_path: Path) -> int:
    summary = json.loads(summary_path.read_text())
    dummy = bool(summary.get("dummy"))
    smoke = bool(summary.get("smoke"))
    summary["verdict"] = decide(summary, dummy=dummy, smoke=smoke)
    print(json.dumps(summary["verdict"], indent=2))
    return 0


def aggregate(
    items: list[ProbeItem],
    samples: list[Sample],
    *,
    n_samples: int,
    dummy: bool,
    smoke: bool = False,
    shard: str | None = None,
    complete: bool = False,
) -> dict:
    gold = {it.item_id: it.gold for it in items}
    buckets: dict[tuple[str, str], list[bool]] = defaultdict(list)
    cot_len: dict[str, list[int]] = defaultdict(list)
    per_item: dict[tuple[str, str, str], list[bool]] = defaultdict(list)

    for s in samples:
        ok = is_correct(s.text, gold[s.item_id])
        buckets[(s.model, s.condition)].append(ok)
        per_item[(s.model, s.condition, s.item_id)].append(ok)
        if s.condition == "F":
            cot_len[s.model].append(s.n_cot_tokens)

    acc = {m: {c: mean_acc(buckets[(m, c)]) for c in CONDS} for m in MODELS}
    drop_p = {m: acc[m]["F"] - acc[m]["P"] for m in MODELS}
    drop_n = {m: acc[m]["F"] - acc[m]["N"] for m in MODELS}

    passk: dict[str, dict[str, dict[str, float]]] = {}
    for m in MODELS:
        passk[m] = {}
        for c in CONDS:
            item_cs = [sum(per_item[(m, c, it.item_id)]) for it in items]
            passk[m][c] = {
                f"pass@{k}": sum(unbiased_pass_at_k(n_samples, c_i, k) for c_i in item_cs)
                / max(len(item_cs), 1)
                for k in (1, 8)
                if k <= n_samples
            }

    summary = {
        "dummy": dummy,
        "smoke": smoke,
        "shard": shard,
        "n_samples": n_samples,
        "n_items": len(items),
        "acc": acc,
        "drop_P": drop_p,
        "drop_N": drop_n,
        "pass_at_k": passk,
        "mean_cot_tokens_F": {
            m: (sum(cot_len[m]) / len(cot_len[m]) if cot_len[m] else 0.0) for m in MODELS
        },
        "note": "pass@k is an Invisible Leash control, not a live gate.",
    }
    verdict = decide(summary, dummy=dummy, smoke=smoke)
    if shard and not dummy and not smoke:
        verdict["decision"] = "SHARD_SKIP"
        verdict["note"] = "Shard runs do not lock. Merge both shards, then read runs/p2/verdict.md."
    elif not complete and not dummy and not smoke:
        verdict["decision"] = "INCOMPLETE"
        verdict["note"] = "Pool is not the frozen 80-item both-arm run. Do not lock."
    summary["verdict"] = verdict
    return summary


def _default_out(*, dummy: bool, smoke: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "dummy" if dummy else ("smoke" if smoke else "full")
    return Path(__file__).resolve().parents[2] / "runs" / f"{stamp}-{tag}"


def _verdict_md(summary: dict) -> str:
    v = summary["verdict"]
    lines = [
        f"# P2 probe verdict: {v['decision']}",
        "",
        v["note"],
        "",
        f"- n_items: {summary['n_items']}",
        f"- n_samples: {summary['n_samples']}",
        f"- entropy/cot mean F tokens: {summary['mean_cot_tokens_F']}",
        f"- acc: {summary['acc']}",
        f"- drop_P: {summary['drop_P']}",
        f"- drop_N: {summary['drop_N']}",
        "",
        f"- capability_delta (think F - base F): {v['capability_delta']:.4f}",
        f"- empty_gap (drop_N base - think): {v['empty_gap']:.4f}",
        f"- n_minus_p_think: {v['n_minus_p_think']:.4f}",
        f"- capability_live: {v['capability_live']}",
        f"- empty_gap_live: {v['empty_gap_live']}",
        f"- not_jet_live: {v['not_jet_live']}",
        f"- jet_replica: {v['jet_replica']}",
    ]
    if v["kill_reasons"]:
        lines.append("- kill_reasons:")
        lines.extend(f"  - {r}" for r in v["kill_reasons"])
    return "\n".join(lines)
