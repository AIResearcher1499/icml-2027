"""CLI: committrap probe | init-items | merge | verdict."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from committrap.checkpoint import (
    ITEMS_NAME,
    PROGRESS_NAME,
    SAMPLES_NAME,
    append_event,
    load_events,
    truncate_torn_tail,
    write_progress,
)
from committrap.data import apply_shard, is_live_item, load_items
from committrap.dummy import (
    Event,
    ProbeItem,
    dummy_F,
    dummy_probe,
    dummy_refork,
    event_key,
)
from committrap.entropy import probe_cuts, tstar_index
from committrap.score import is_correct
from committrap.verdict import N_LIVE_ITEMS, decide

K_REFORK = 4
SUMMARY_NAME = "summary.json"
VERDICT_NAME = "verdict.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="committrap")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="run the frozen 72h S1 probe")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--smoke", action="store_true", help="8B, n=2, 8 items; never locks")
    p.add_argument(
        "--bench",
        action="store_true",
        help="8B, 1 item x 1 sample, same generate flags as A6000; times tok/s; never locks",
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--n-items", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--shard", default=None, help="item slice, e.g. 0/2")
    p.add_argument("--items", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fresh", action="store_true")

    ini = sub.add_parser("init-items", help="freeze GSM8K items.json")
    ini.add_argument("--out", type=Path, required=True)
    ini.add_argument("--n-items", type=int, default=80)
    ini.add_argument("--seed", type=int, default=0)
    ini.add_argument("--items", type=Path, default=None)

    m = sub.add_parser("merge", help="merge shard run dirs")
    m.add_argument("shards", nargs="+", type=Path)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--n-samples", type=int, default=8)
    m.add_argument("--n-items", type=int, default=80)

    v = sub.add_parser("verdict", help="recompute verdict from summary.json")
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
    items = load_items(dummy=False, n_items=args.n_items, seed=args.seed, items_path=args.items)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ITEMS_NAME
    path.write_text(
        json.dumps({it.item_id: it.to_dict() for it in items}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {len(items)} items -> {path}", flush=True)
    return 0


def run_probe(args: argparse.Namespace) -> int:
    dummy = bool(args.dummy)
    bench = bool(args.bench)
    smoke = bool(args.smoke) or bench
    args.model = args.model or "Qwen/Qwen3-8B"
    if bench:
        args.n_samples = 1
        args.n_items = 1
    elif smoke and not dummy:
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
    existing, done_keys = load_events(samples_path)
    by_key = {event_key(e): e for e in existing}

    gen = None
    remaining_f = sum(
        1
        for it in items
        for i in range(args.n_samples)
        if ("F", it.item_id, i) not in done_keys
    )
    print(
        f"run dir {out_dir} checkpointed={len(done_keys)} remaining_F={remaining_f} items={len(items)}",
        flush=True,
    )
    if remaining_f > 0 and not dummy:
        from committrap.generate import HFGenerator

        gen = HFGenerator(args.model, temperature=args.temperature)

    n_new = 0
    gold = {it.item_id: it.gold for it in items}

    def _emit(e: Event) -> None:
        nonlocal n_new
        append_event(samples_path, e)
        done_keys.add(event_key(e))
        by_key[event_key(e)] = e
        n_new += 1
        write_progress(
            out_dir / PROGRESS_NAME,
            done=len(done_keys),
            total=max(len(done_keys), 1),
            last=f"{e.kind} {e.item_id}#{e.sample_idx} cut={e.cut}",
        )
        if n_new % 10 == 0:
            print(f"[{len(done_keys)}] {e.kind} {e.item_id}#{e.sample_idx} cut={e.cut}", flush=True)

    for item in items:
        for i in range(args.n_samples):
            fk = ("F", item.item_id, i)
            if fk not in by_key:
                _emit(dummy_F(item, i) if dummy else gen.sample_F(item, i, args.seed))
            f_ev = by_key[fk]
            cuts = probe_cuts(f_ev.n_cot_tokens)
            half = f_ev.n_cot_tokens // 2
            if half > 0 and half not in cuts:
                cuts = sorted(cuts + [half])
            hs: list[float] = []
            cut_to_probe: dict[int, Event] = {}
            for cut in cuts:
                pk = ("probe", item.item_id, i, cut)
                if pk not in by_key:
                    if dummy:
                        _emit(dummy_probe(item, i, cut))
                    else:
                        _emit(gen.probe_at(item, f_ev, cut, i, args.seed))
                pr = by_key[pk]
                cut_to_probe[cut] = pr
                hs.append(float(pr.H or 0.0))
            t_idx = tstar_index(hs)
            t_cut = cuts[t_idx] if t_idx is not None else None
            need_cuts: list[int] = []
            if t_cut is not None and not is_correct(cut_to_probe[t_cut].text, gold[item.item_id]):
                need_cuts.append(t_cut)
            if half > 0 and half in cut_to_probe and not is_correct(
                cut_to_probe[half].text, gold[item.item_id]
            ):
                if half not in need_cuts:
                    need_cuts.append(half)
            for cut in need_cuts:
                for k in range(K_REFORK):
                    rk = ("refork", item.item_id, i, cut, k)
                    if rk in by_key:
                        continue
                    if dummy:
                        _emit(dummy_refork(item, i, cut, k))
                    else:
                        _emit(gen.refork_at(item, f_ev, cut, i, k, args.seed))

    events, _ = load_events(samples_path)
    n_f = sum(1 for e in events if e.kind == "F")
    expected_f = len(items) * args.n_samples
    complete = (
        args.shard is None
        and not dummy
        and not smoke
        and n_f == expected_f
        and len(items) == args.n_items
    )
    summary = aggregate(
        items,
        events,
        n_samples=args.n_samples,
        dummy=dummy,
        smoke=smoke,
        shard=args.shard,
        complete=complete,
    )
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / VERDICT_NAME).write_text(_verdict_md(summary) + "\n")
    print(json.dumps(summary["verdict"], indent=2))
    print(f"wrote {out_dir} new_events={n_new} F={n_f}/{expected_f}")
    if bench and gen is not None and getattr(gen, "last_tok_s", None):
        tok_s = float(gen.last_tok_s)
        # 80 items x 8 F x ~1500 CoT tokens (P2 think mean), single GPU.
        hours = (80 * 8 * 1500) / max(tok_s, 1e-6) / 3600.0
        print(
            f"bench tok/s={tok_s:.1f}  rough 80x8 F-only hours@{tok_s:.0f}tok/s={hours:.1f} "
            f"(same 8B flags as A6000; probes+refork extra)",
            flush=True,
        )
    return 0


def _load_or_create_items(items_path: Path, args: argparse.Namespace, dummy: bool) -> list[ProbeItem]:
    if items_path.exists():
        raw = json.loads(items_path.read_text())
        recs = list(raw.values()) if isinstance(raw, dict) else raw
        items = [ProbeItem.from_dict(r) for r in recs]
        print(f"reloaded {len(items)} items from {items_path.name}", flush=True)
        return items
    items = load_items(dummy=dummy, n_items=args.n_items, seed=args.seed, items_path=args.items)
    items = apply_shard(items, args.shard)
    items_path.write_text(
        json.dumps({it.item_id: it.to_dict() for it in items}, indent=2, ensure_ascii=False) + "\n"
    )
    return items


def run_merge(args: argparse.Namespace) -> int:
    items: list[ProbeItem] = []
    seen_ids: set[str] = set()
    events: list[Event] = []
    seen_keys: set[tuple] = set()
    dummy_any = False
    smoke_any = False
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
        shard_events, _ = load_events(samples_path)
        for e in shard_events:
            key = event_key(e)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            events.append(e)
        sp = shard / SUMMARY_NAME
        if sp.exists():
            blob = json.loads(sp.read_text())
            dummy_any = dummy_any or bool(blob.get("dummy"))
            smoke_any = smoke_any or bool(blob.get("smoke"))

    n_f = sum(1 for e in events if e.kind == "F")
    expected_f = args.n_items * args.n_samples
    if len(items) != args.n_items or n_f != expected_f:
        print(
            f"merge incomplete: items={len(items)} expected {args.n_items}; "
            f"F={n_f} expected {expected_f}",
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
        events,
        n_samples=args.n_samples,
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
    events: list[Event],
    *,
    n_samples: int,
    dummy: bool,
    smoke: bool = False,
    shard: str | None = None,
    complete: bool = False,
) -> dict:
    gold = {it.item_id: it.gold for it in items}
    f_by: dict[tuple[str, int], Event] = {}
    probes: dict[tuple[str, int], list[Event]] = defaultdict(list)
    reforks: dict[tuple[str, int, int], list[Event]] = defaultdict(list)
    for e in events:
        if e.kind == "F":
            f_by[(e.item_id, e.sample_idx)] = e
        elif e.kind == "probe":
            probes[(e.item_id, e.sample_idx)].append(e)
        elif e.kind == "refork":
            reforks[(e.item_id, e.sample_idx, int(e.cut or 0))].append(e)

    n_defined = 0
    cell_cont: list[bool] = []
    cell_ref: list[float] = []
    cell50_cont: list[bool] = []
    cell50_ref: list[float] = []
    tokens_after: list[int] = []

    live_items = [it for it in items if is_live_item(it.item_id)]
    for it in live_items:
        for i in range(n_samples):
            f_ev = f_by.get((it.item_id, i))
            if f_ev is None:
                continue
            prs = sorted(probes[(it.item_id, i)], key=lambda p: int(p.cut or 0))
            hs = [float(p.H or 0.0) for p in prs]
            t_idx = tstar_index(hs)
            if t_idx is not None:
                n_defined += 1
                t_cut = int(prs[t_idx].cut or 0)
                trunc_ok = is_correct(prs[t_idx].text, gold[it.item_id])
                cont_ok = is_correct(f_ev.text, gold[it.item_id])
                tokens_after.append(max(f_ev.n_cot_tokens - t_cut, 0))
                if not trunc_ok:
                    rf = reforks.get((it.item_id, i, t_cut), [])
                    if rf:
                        cell_cont.append(cont_ok)
                        cell_ref.append(sum(is_correct(r.text, gold[it.item_id]) for r in rf) / len(rf))
            half = f_ev.n_cot_tokens // 2
            half_probe = next((p for p in prs if int(p.cut or 0) == half), None)
            if half_probe is not None and not is_correct(half_probe.text, gold[it.item_id]):
                rf = reforks.get((it.item_id, i, half), [])
                if rf:
                    cell50_cont.append(is_correct(f_ev.text, gold[it.item_id]))
                    cell50_ref.append(sum(is_correct(r.text, gold[it.item_id]) for r in rf) / len(rf))

    def _mean(xs: list[float] | list[bool]) -> float:
        if not xs:
            return 0.0
        return float(sum(xs) / len(xs))

    recovery = _mean(cell_cont)
    acc_ref = _mean(cell_ref)
    rec50 = _mean(cell50_cont)
    acc50 = _mean(cell50_ref)
    summary = {
        "dummy": dummy,
        "smoke": smoke,
        "shard": shard,
        "n_samples": n_samples,
        "n_items": len(items),
        "n_live_items": len(live_items),
        "n_cell": len(cell_cont),
        "n_cell_50": len(cell50_cont),
        "n_defined": n_defined,
        "recovery_continue": recovery,
        "acc_refork_cell": acc_ref,
        "gap_tstar": acc_ref - recovery,
        "recovery_continue_50": rec50,
        "acc_refork_50": acc50,
        "gap_50": acc50 - rec50,
        "mean_tokens_after_tstar": _mean(tokens_after),
        "note": "Live pool is gsm-016..gsm-079. Calibration gsm-000..015 excluded.",
    }
    verdict = decide(summary, dummy=dummy, smoke=smoke)
    if shard and not dummy and not smoke:
        verdict["decision"] = "SHARD_SKIP"
        verdict["note"] = "Shard runs do not lock. Merge both shards, then read runs/s1/verdict.md."
    elif not complete and not dummy and not smoke:
        verdict["decision"] = "INCOMPLETE"
        verdict["note"] = "Pool is not the frozen 80-item run. Do not lock."
    summary["verdict"] = verdict
    return summary


def _default_out(*, dummy: bool, smoke: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "dummy" if dummy else ("smoke" if smoke else "full")
    return Path(__file__).resolve().parents[2] / "runs" / f"{stamp}-{tag}"


def _verdict_md(summary: dict) -> str:
    v = summary["verdict"]
    lines = [
        f"# S1 probe verdict: {v['decision']}",
        "",
        v["note"],
        "",
        f"- n_items: {summary['n_items']} (live {summary['n_live_items']})",
        f"- n_samples: {summary['n_samples']}",
        f"- n_cell (wrong-at-t*): {summary['n_cell']}",
        f"- n_defined: {summary['n_defined']}",
        f"- defined_frac: {v['defined_frac']:.4f}  (need >= 0.50 of {N_LIVE_ITEMS}*n)",
        f"- recovery_continue: {summary['recovery_continue']:.4f}",
        f"- acc_refork_cell: {summary['acc_refork_cell']:.4f}",
        f"- gap_tstar: {summary['gap_tstar']:.4f}",
        f"- gap_50: {summary['gap_50']:.4f}",
        f"- mean tokens after t*: {summary['mean_tokens_after_tstar']:.1f}",
        "",
        f"- cell_live: {v['cell_live']}",
        f"- phase_live: {v['phase_live']}",
        f"- lockin_live: {v['lockin_live']}",
        f"- gap_live: {v['gap_live']}",
    ]
    if v["kill_reasons"]:
        lines.append("- kill_reasons:")
        lines.extend(f"  - {r}" for r in v["kill_reasons"])
    return "\n".join(lines)
