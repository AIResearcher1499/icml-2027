# Frozen 72h probe criteria (S1)

Frozen: 2026-08-30. Do not edit thresholds after seeing a number.
Amendments go below as dated notes, never as silent edits.

Source: `docs/11-s1-novelty.md`. This file is the executable spec.
Dummy never locks. Do not lock S1 in `docs/05-decision.md` from this probe.

Do not change P1 `KILL.md` or P2 `KILL-p2.md`.

## Protocol (frozen)

- Model: `Qwen/Qwen3-8B`, `enable_thinking=True`
- Temperature 0.7, top_p 1.0, seed 0
- F: `max_new_tokens` 2048; answer probes / refork answers: `max_new_tokens` 64
- n=8 F samples/item
- Pool: 80 GSM8K test items, shuffle seed 0, ids `gsm-000`…`gsm-079` (same freeze as P2 if `probe-p2/runs/p2/items.json` exists)
- Calibration (detector only): `gsm-000`…`gsm-015`. **Not** in the live metrics.
- Live pool: `gsm-016`…`gsm-079` (64 items)
- MATH-500 is log-only; does not enter the live test

### Entropy detector (Xu-style, not next-token CoT entropy)

After every **256** generated CoT tokens, and at the end of F, probe an intermediate answer with the same stop used by Xu et al. (close think, `The answer is`).

`H_i` = mean next-token entropy (nats) of that probed answer span (Xu eq. 2 analogue: average per answer token).

`t*` on a completed F trace = probe index of the **sharpest drop** `H_{i-1} - H_i` (i≥2). This is Xu’s visualization alignment, applied offline. If every drop `< 0.05`, `t*` is undefined.

Do not fit CUSUM `h` on the live pool. Do not switch to next-token entropy of the CoT body.

### Conditions (same F prefix)

Defined only when `t*` exists.

- **Trunc:** probed answer at `t*` (already computed)
- **Continue:** the original F final answer (tokens after `t*` are the committed suffix)
- **Refork:** from the prefix at `t*`, sample **k=4** new continuations (`max_new_tokens=512`, temp 0.7), score each final answer. `acc_refork` = mean of the 4 (avg@4). Log pass@1 and pass@4; they are **not** live gates
- **JET control:** same Trunc / Continue / Refork using cut `floor(0.5 * n_cot_tokens)` instead of `t*`

Scoring: GSM8K numeric match after `####` or last number in the answer span. Same matcher as P2.

## Metrics (frozen)

Let a **cell** be one `(item, sample_idx)` on the live pool with defined `t*` and **Trunc wrong**.

- `n_cell` = number of such cells
- `n_defined` = live traces with defined `t*`
- `recovery_continue` = mean 1[Continue correct] on the cell
- `acc_refork_cell` = mean `acc_refork` on the cell
- `gap_tstar` = `acc_refork_cell - recovery_continue`
- `gap_50` = the same gap using the JET 50% cut, on cells where Trunc-at-50% is wrong (may be a different set; do not mix)

Log (not live): mean tokens after `t*`; fraction of live traces with defined `t*`; two-phase plot; length of F.

## Live (all four required)

1. Coverage of the cell: `n_cell >= 40`
2. Two-phase exists: `n_defined / (64 * 8) >= 0.50`
3. Lock-in: `recovery_continue <= 0.05`
4. Refork beats continue, not a length replica: `gap_tstar >= 0.10` **and** `gap_tstar - gap_50 >= 0.05`

## Kill (any one)

- (1) fails: wrong-at-`t*` cell too small (Xu High Reliability; no paper)
- (2) fails: no two-phase under this detector
- (3) fails: original suffix often recovers after `t*` (von Recum-style self-correction; not a trap)
- (4) fails: refork does not beat continue, **or** 50% cut shows the same gap → JET replica
- Smoke/dummy never decides

## GPU

- One 48 GB GPU is enough. Prefer 2×A6000 item split 0/2 | 1/2 then merge. No tensor-parallel. Do not run until this file exists (it does).
- Budget sketch: 64×8 F (2048) + probes every 256 tokens (~8 probes × 64 tokens) + 4×512 refork on the cell only. Think CoTs are long; plan **~0.5 GPU-day**. Calibration 16 items included in F.

## Amendments

- 2026-08-30: Runner lives in `probe-s1/` (`uv run committrap probe`). Dummy on Mac never locks. Full run: CUDA (2×A6000 item split) or MPS (same CLI; not the paper path). Not a threshold change.
- 2026-08-30: `--bench` is smoke-class (**Qwen3-8B**, 1 item × 1 sample, same generate flags as the full run). Prints tok/s. Never locks. Not a threshold change. The 72h probe is inference-only (no GRPO).
