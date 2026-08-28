"""CLI: accesstrap probe | verdict."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from accesstrap.aggregate import aggregate
from accesstrap.checkpoint import (
    ITEMS_NAME,
    PROGRESS_NAME,
    SAMPLES_NAME,
    append_sample,
    load_samples,
    sample_key,
    truncate_torn_tail,
    write_progress,
)
from accesstrap.data import ProbeItem, load_items
from accesstrap.generate import CONDITIONS, HFGenerator, dummy_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="accesstrap")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="run the frozen 72h probe")
    p.add_argument("--dummy", action="store_true", help="no model; deterministic fakes")
    p.add_argument("--smoke", action="store_true", help="tiny real-model run (0.6B, n=4, 8 items)")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--n-math", type=int, default=80)
    p.add_argument("--n-qa", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="run directory; if samples.jsonl exists, skip finished keys and append",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignore existing samples.jsonl in --out and start a new log (renames old file)",
    )

    v = sub.add_parser("verdict", help="recompute verdict from a saved summary.json")
    v.add_argument("summary", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "probe":
        return run_probe(args)
    return run_verdict(args.summary)


def run_probe(args: argparse.Namespace) -> int:
    dummy = bool(args.dummy)
    if args.smoke and not dummy:
        args.model = "Qwen/Qwen3-0.6B"
        args.n_samples = 4
        args.n_math = 4
        args.n_qa = 4
    if dummy:
        args.n_math = min(args.n_math, 2)
        args.n_qa = min(args.n_qa, 2)
        args.n_samples = min(args.n_samples, 8)

    out_dir = args.out or _default_out(dummy=dummy, smoke=args.smoke)
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
    total = len(items) * len(CONDITIONS) * args.n_samples
    remaining = total - len(done_keys)
    print(
        f"run dir {out_dir} checkpointed={len(done_keys)} remaining={remaining} total={total}",
        flush=True,
    )

    gen = None
    if remaining > 0 and not dummy:
        gen = HFGenerator(args.model, args.max_new_tokens, args.temperature)

    n_new = 0
    for item in items:
        for cond in CONDITIONS:
            for i in range(args.n_samples):
                key = sample_key(item.item_id, cond, i)
                if key in done_keys:
                    continue
                if dummy:
                    s = dummy_sample(item, cond, i)
                else:
                    s = gen.sample(item, cond, i, args.seed)
                append_sample(samples_path, s)
                done_keys.add(key)
                existing.append(s)
                n_new += 1
                done_n = len(done_keys)
                write_progress(
                    out_dir / PROGRESS_NAME,
                    done=done_n,
                    total=total,
                    last=f"{item.item_id} {cond}#{i}",
                )
                if n_new % 10 == 0 or done_n == total:
                    print(f"[{done_n}/{total}] {item.item_id} {cond}#{i}", flush=True)

    samples, _ = load_samples(samples_path)
    blob = aggregate(items, samples, args.n_samples)
    (out_dir / "summary.json").write_text(json.dumps(blob["summary"], indent=2) + "\n")
    (out_dir / "per_item.json").write_text(
        json.dumps(_strip_raw_ents(blob["items"]), indent=2) + "\n"
    )
    verdict = blob["summary"]["verdict"]
    (out_dir / "verdict.md").write_text(_verdict_md(blob["summary"]))
    print(json.dumps(verdict, indent=2))
    print(f"wrote {out_dir} new_samples={n_new}")
    return 0 if verdict["decision"] in {"LIVE", "KILL"} else 1


def _load_or_create_items(items_path: Path, args: argparse.Namespace, dummy: bool) -> list[ProbeItem]:
    if items_path.exists():
        raw = json.loads(items_path.read_text())
        if isinstance(raw, dict):
            recs = list(raw.values())
        else:
            recs = raw
        items = [ProbeItem.from_dict(r) for r in recs]
        print(f"reloaded {len(items)} items from {items_path.name}", flush=True)
        return items
    items = load_items(n_math=args.n_math, n_qa=args.n_qa, dummy=dummy)
    payload = {it.item_id: it.to_dict() for it in items}
    items_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return items


def run_verdict(path: Path) -> int:
    from accesstrap.verdict import decide

    summary = json.loads(path.read_text())
    if "venn_AB" not in summary:
        summary = summary["summary"]
    v = decide(summary)
    print(json.dumps(v, indent=2))
    return 0


def _default_out(*, dummy: bool, smoke: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "dummy" if dummy else ("smoke" if smoke else "full")
    root = Path(__file__).resolve().parents[2] / "runs" / f"{stamp}-{tag}"
    return root


def _strip_raw_ents(items: list[dict]) -> list[dict]:
    slim = []
    for r in items:
        d = {k: v for k, v in r.items() if k not in {"A", "B", "C"}}
        for cond in ("A", "B", "C"):
            cell = dict(r[cond])
            cell["n_lexical"] = len(cell.pop("lexical_ents", []))
            cell["n_top20"] = len(cell.pop("top20_ents", []))
            d[cond] = cell
        slim.append(d)
    return slim


def _verdict_md(summary: dict) -> str:
    v = summary["verdict"]
    lines = [
        f"# Probe verdict: {v['decision']}",
        "",
        f"- entropy rule: `{summary['entropy_rule']}`",
        f"- primary-band items: {summary['n_items_primary_band']} / {summary['n_items_pool']}",
        f"- Venn A-B: {summary['venn_AB']}",
        f"- entropy A/B/C: {summary['entropy']}",
        f"- mean pass primary: {summary['mean_pass_primary']}",
        "",
        "## Flags",
        f"- coverage_live: {v['coverage_live']}",
        f"- gold_entropy_drop: {v['gold_entropy_drop']}",
        f"- distractor_dissociation: {v['distractor_dissociation']}",
        "",
    ]
    if v["kill_reasons"]:
        lines.append("## Kill reasons")
        lines.extend(f"- {r}" for r in v["kill_reasons"])
        lines.append("")
    lines.append("Smoke/dummy never locks the paper. See KILL.md.")
    lines.append("")
    return "\n".join(lines)
