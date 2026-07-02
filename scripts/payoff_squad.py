"""WP-ST-15: Contract-conditioned payoff test — local Qwen on WP-14 seed.

Decisive question: on the WP-14 frozen SQuAD-v2 seed, does conditioning local
Qwen (Ollama, temp=0) on the real contract reduce deterministic verify()
violations vs a FAIR plain prompt?

Three PRE-REGISTERED conditions — locked as static strings BEFORE any run so
CONTRACT cannot win merely by stricter formatting:

  PLAIN_NAKED : source + question + shared format demand (floor)
  PLAIN_FAIR  : source + question + neutral instruction incl. an explicit
                abstain-permitting hint + shared format demand (honest baseline)
  CONTRACT    : source + question + rendered ⟨A, F, B, P, ¬B⟩ pack + shared
                format demand (contract-conditioned)

Anti-tuning: the SHARED_FORMAT string is IDENTICAL across all three conditions.
Only wrapping differs — naked, fair-instructions, or contract.

The verifier is scripts/contract_schema.py::verify (CW+CD converged) — the SOLE
metric. Token-grounding ¬B is anti-fabrication-only (coverage_FN=100% documented;
semantic errors uncaught) — bounds what this test can conclude.

Pre-registration timestamp (UTC): 2026-07-01T02:50:00Z — locking the 3 prompts.
Any edit to the prompt strings AFTER first Qwen call invalidates the experiment.

Usage:
  python scripts/payoff_squad.py --list-conditions   # U1 audit: show all 3 prompts on one row
  python scripts/payoff_squad.py --smoke             # U3: 20 items × 3 conditions × 1 rep
  python scripts/payoff_squad.py --run               # U4: ~400 items × 3 conditions × N reps
  python scripts/payoff_squad.py --aggregate         # U5: per-regime gates + results md
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Reuse repo utilities ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contract_schema import verify  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────────
SEED_DIR    = "data/contract_squad"
ITEMS_FILE  = os.path.join(SEED_DIR, "items.jsonl")

OUT_DIR       = "data/contract_squad_payoff"
RAW_RUNS_FILE = os.path.join(OUT_DIR, "raw_runs.jsonl")
MANIFEST_FILE = os.path.join(OUT_DIR, "backend_manifest.json")
RESULTS_MD    = "papers/results_payoff_squad.md"

SEED             = 42
N_SMOKE_PER_KIND = 10   # 10 answerable + 10 unanswerable = 20 items
N_RUN_PER_KIND   = 200  # ~200 answerable + ~200 unanswerable = ~400 items
N_REPS           = 1    # temp=0; single rep; can bump to 2 if determinism drifts
GATE_FLOOR       = 0.05 # Δ absolute for gate_contract_payoff (pre-registered at U5)

# ══════════════════════════════════════════════════════════════════════════
# PRE-REGISTERED PROMPTS — frozen at 2026-07-01T02:50:00Z UTC.
# DO NOT edit any string below AFTER the first Qwen call — that invalidates
# the experiment (anti-tuning discipline per CW+CD directive).
# ══════════════════════════════════════════════════════════════════════════

PREREGISTRATION_UTC = "2026-07-01T02:50:00Z"

# Identical output-format demand appended to ALL three conditions. This is
# the anti-tuning invariant: CONTRACT must beat PLAIN_FAIR on substance,
# not by asking for a stricter format the others don't.
SHARED_FORMAT = "Give a single, concise answer."

# --- PLAIN_NAKED: floor. Source + question + shared format demand only.
PLAIN_NAKED_TEMPLATE = """Passage:
{source}

Question: {question}

{shared_format}"""

# --- PLAIN_FAIR: honest baseline. Adds neutral abstain-permitting hint —
# the fair mirror of CONTRACT's `must_abstain` semantic without the contract
# scaffolding. If PLAIN_FAIR ≈ CONTRACT, the payoff is in prose framing, not
# in the contract structure (the WP-6A-like null).
PLAIN_FAIR_TEMPLATE = """Passage:
{source}

Question: {question}

Instructions:
- Answer using only information stated in the passage.
- If the passage does not answer the question, say so explicitly.

{shared_format}"""

# --- CONTRACT: renders the real ⟨A, F, B, P, ¬B⟩ pack from the WP-14 record.
# The ¬B carries the boundary the deterministic verifier will actually check.
CONTRACT_TEMPLATE = """Passage:
{source}

Question: {question}

Contract:
- A (Antecedent, what you are given): a passage and a query about it.
- F (Function, what to compute): {F}
- B (Boundary, what an admissible answer must satisfy): {B_human}
- Not-B (Boundary violation, what would fabricate): {notB_human}
- P (Precondition): {P_human}

{shared_format}"""


def _render_B_human(contract: dict) -> str:
    """Human-readable rendering of B — matches the verifier's actual semantics."""
    B = contract["B"]
    if B.get("must_abstain"):
        markers = ", ".join(f'"{m}"' for m in B.get("abstain_markers", []))
        return (
            "the answer must EXPLICITLY indicate that the passage does not "
            "state the answer, using language like " + markers + "."
        )
    if "content_tokens_must_be_grounded_in_source" in B:
        return (
            "every content-word in the answer (proper nouns, numbers, and "
            "non-stopword words) must appear somewhere in the passage."
        )
    return "an admissible answer per the contract."


def _render_notB_human(contract: dict) -> str:
    """Human-readable rendering of ¬B — matches the verifier's actual semantics."""
    B = contract["B"]
    if B.get("must_abstain"):
        return (
            "asserting any specific answer without one of the abstention markers "
            "above — that would fabricate an answer the passage does not support."
        )
    if "content_tokens_must_be_grounded_in_source" in B:
        return (
            "introducing any content-word (proper noun, number, or non-stopword) "
            "not present in the passage — that would fabricate content."
        )
    return "content not grounded in the passage."


def _render_P_human(contract: dict) -> str:
    return "; ".join(contract.get("P", ["closed world: use only the passage"]))


def _render_F_human(contract: dict) -> str:
    return contract.get("F", "answer per the contract.")


def render_prompt(record: dict, condition: str) -> str:
    """Render the prompt for one (record, condition) pair.

    Returns the FULL user-message string (Qwen chat 'user' content). System
    prompt is intentionally left empty — the frozen strings above carry the
    complete pre-registered condition; no system-prompt hidden variables.
    """
    source   = record["source"]
    question = record["nl_prompt"]
    contract = record["contract"]

    if condition == "PLAIN_NAKED":
        return PLAIN_NAKED_TEMPLATE.format(
            source=source, question=question, shared_format=SHARED_FORMAT,
        )
    if condition == "PLAIN_FAIR":
        return PLAIN_FAIR_TEMPLATE.format(
            source=source, question=question, shared_format=SHARED_FORMAT,
        )
    if condition == "CONTRACT":
        return CONTRACT_TEMPLATE.format(
            source=source, question=question,
            F=_render_F_human(contract),
            B_human=_render_B_human(contract),
            notB_human=_render_notB_human(contract),
            P_human=_render_P_human(contract),
            shared_format=SHARED_FORMAT,
        )
    raise ValueError(f"unknown condition: {condition!r}")


CONDITIONS = ["PLAIN_NAKED", "PLAIN_FAIR", "CONTRACT"]


# ── U1 audit helper — show all 3 prompts on a couple of rows ──────────────
def cmd_list_conditions():
    """Print the 3 pre-registered prompts on one answerable + one unanswerable row.
    Human-readable output for eyeball audit — confirms SHARED_FORMAT is identical,
    only wrapping differs, and CONTRACT's ¬B language matches the verifier."""
    if not os.path.exists(ITEMS_FILE):
        print(f"[list] ERROR: WP-14 seed missing at {ITEMS_FILE}")
        return 1

    with open(ITEMS_FILE) as f:
        records = [json.loads(l) for l in f]
    ans   = next(r for r in records if r["kind"] == "answerable")
    unans = next(r for r in records if r["kind"] == "unanswerable")

    print(f"# Pre-registration timestamp (UTC): {PREREGISTRATION_UTC}")
    print(f"# Shared format demand (identical across all conditions):")
    print(f"# > {SHARED_FORMAT}")
    print()

    for row in (ans, unans):
        print("=" * 78)
        print(f"ROW  id={row['id']}  kind={row['kind']}")
        print("=" * 78)
        for cond in CONDITIONS:
            print(f"\n─── condition: {cond} ─────────────────────────────────────────")
            print(render_prompt(row, cond))
    return 0


# ══════════════════════════════════════════════════════════════════════════
# U2: Harness + raw-output recorder
#
# For each (item, condition, rep):
#   render_prompt → Qwen (Ollama /v1/chat/completions, temp=0) → verify()
#   → JSONL record with EXACT fields David audits by eye:
#     {id, kind, condition, rep_id, qwen_output, verifier_admissible, verifier_reason}
#
# FAIL-FAST assert_frozen_hash on WP-14 seed at harness entry. Resumable via
# done-set keyed by (id, condition, rep_id). Retry 2× on transient errors.
# ══════════════════════════════════════════════════════════════════════════

def _call_qwen(prompt: str, base_url: str, model: str, api_key: str,
               timeout: int = 180, temperature: float = 0.0,
               max_retries: int = 2) -> str:
    """POST to Ollama /v1/chat/completions; return message content or 'ERROR: ...'."""
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream":      False,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                doc = json.loads(r.read())
            return doc["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, ConnectionError, json.JSONDecodeError, KeyError) as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
    return f"ERROR: {last_err}"


def _load_seed_items() -> list:
    """Load the WP-14 seed items.jsonl."""
    with open(ITEMS_FILE) as f:
        return [json.loads(l) for l in f]


def _pick_stratified(items: list, n_per_kind: int, seed: int) -> list:
    """Deterministic stratified sample: n_per_kind answerable + n_per_kind unanswerable.
    Preserves record dicts as-is."""
    rng = random.Random(seed)
    by_kind = {}
    for rec in items:
        by_kind.setdefault(rec["kind"], []).append(rec)
    picked = []
    for kind in ("answerable", "unanswerable"):
        group = list(by_kind.get(kind, []))
        rng.shuffle(group)
        picked.extend(group[:n_per_kind])
    return picked


def _run_harness(records: list, out_path: str, n_reps: int) -> dict:
    """Iterate (record, condition, rep_id) triples; skip already-done; write JSONL.

    FAIL-FAST: assert_frozen_hash on the WP-14 seed before any work.
    Records backend manifest to sidecar file once per harness invocation.
    Prints a per-triple status line so David can watch the stream live.
    """
    # FAIL-FAST — seed must be frozen at the WP-14 hash
    from cbt.fingerprint import assert_frozen_hash
    frozen = assert_frozen_hash(SEED_DIR)
    print(f"[harness] frozen seed hash asserted: {frozen}")

    # Backend + provenance manifest (safe view; api_key never persisted)
    from cbt.llm_backend import get_backend, record_backend_manifest
    backend = get_backend()
    if backend["backend"] != "local":
        raise RuntimeError(
            f"LLM_BACKEND must be 'local' for WP-15 (got {backend['backend']!r}). "
            "Set LLM_BACKEND=local."
        )
    os.makedirs(OUT_DIR, exist_ok=True)
    record_backend_manifest(MANIFEST_FILE)
    print(f"[harness] backend={backend['backend']} model={backend['model']} "
          f"base_url={backend['base_url']}")
    print(f"[harness] manifest → {MANIFEST_FILE}")

    # Resume — keyed by (id, condition, rep_id)
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["id"], r["condition"], r["rep_id"]))
                except Exception:
                    pass
    total = len(records) * len(CONDITIONS) * n_reps
    print(f"[harness] {len(records)} records × {len(CONDITIONS)} conditions × "
          f"{n_reps} reps = {total} calls  ({len(done)} already done)")

    n_run = 0
    n_err = 0
    with open(out_path, "a") as fout:
        for idx, rec in enumerate(records):
            for cond in CONDITIONS:
                for rep_id in range(n_reps):
                    key = (rec["id"], cond, rep_id)
                    if key in done:
                        continue
                    prompt = render_prompt(rec, cond)
                    output = _call_qwen(
                        prompt, backend["base_url"], backend["model"],
                        backend["api_key"],
                    )
                    is_err = output.startswith("ERROR:")
                    if is_err:
                        verdict = {
                            "admissible": False,
                            "reason":     f"llm_error_skip: {output[:80]}",
                        }
                        n_err += 1
                    else:
                        verdict = verify(output, rec["contract"])

                    line_rec = {
                        "id":                  rec["id"],
                        "kind":                rec["kind"],
                        "condition":           cond,
                        "rep_id":              rep_id,
                        "qwen_output":         output,
                        "verifier_admissible": verdict["admissible"],
                        "verifier_reason":     verdict["reason"],
                    }
                    fout.write(json.dumps(line_rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    done.add(key)
                    n_run += 1
                    mark = "ERR" if is_err else ("OK" if verdict["admissible"] else "VIO")
                    print(f"  [{mark}] {idx + 1:3d}/{len(records)} "
                          f"id={rec['id'][:24]:24s} {rec['kind']:12s} "
                          f"cond={cond:11s} rep={rep_id}  "
                          f"out={output[:60]!r}")
    print(f"\n[harness] {n_run} new calls; {n_err} errors.")
    return {"n_calls_new": n_run, "n_errors": n_err, "frozen_hash": frozen,
            "backend": backend["model"]}


# ══════════════════════════════════════════════════════════════════════════
# U3: Smoke — 20 items × 3 conditions × 1 rep + headroom check
# ══════════════════════════════════════════════════════════════════════════

def cmd_smoke():
    """U3: 10 answerable + 10 unanswerable × 3 conditions × 1 rep = 60 calls.

    Prints per-triple verdict live via _run_harness, then a summary + a
    critical HEADROOM CHECK: on unanswerable, PLAIN_NAKED MUST produce at
    least some violations. If Qwen abstains everywhere even without a
    contract, there's no headroom for CONTRACT to fill and the WP falls into
    the SATURATION branch (WP predicts we switch domain).
    """
    items = _load_seed_items()
    sample = _pick_stratified(items, N_SMOKE_PER_KIND, seed=SEED)
    print(f"[smoke] sampled {len(sample)} items (target: {N_SMOKE_PER_KIND} per kind)")
    for r in sample:
        print(f"  id={r['id']} kind={r['kind']} q={r['nl_prompt'][:70]!r}")
    print()

    stats = _run_harness(sample, RAW_RUNS_FILE, n_reps=1)

    # Aggregate stats from raw_runs.jsonl (whole file — smoke may be re-run)
    print(f"\n[smoke] Aggregating raw_runs.jsonl for the smoke sample ...")
    sample_ids = {r["id"] for r in sample}
    per_cell = {}  # (kind, condition) → [admissible bool, ...]
    with open(RAW_RUNS_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] not in sample_ids:
                continue
            per_cell.setdefault((r["kind"], r["condition"]), []).append(
                r["verifier_admissible"]
            )

    print(f"\n[smoke] Violation rate by (kind, condition):")
    print(f"  {'kind':13s} {'condition':11s}  n   V   V-rate")
    for kind in ("answerable", "unanswerable"):
        for cond in CONDITIONS:
            adms = per_cell.get((kind, cond), [])
            n = len(adms)
            v = sum(1 for a in adms if not a)
            rate = v / n if n else 0.0
            print(f"  {kind:13s} {cond:11s}  {n:2d}  {v:2d}  {rate:.2%}")

    # ── Headroom check ────────────────────────────────────────────────────
    unans_naked = per_cell.get(("unanswerable", "PLAIN_NAKED"), [])
    unans_naked_v = sum(1 for a in unans_naked if not a)
    headroom = unans_naked_v > 0

    ans_naked = per_cell.get(("answerable", "PLAIN_NAKED"), [])
    ans_naked_v = sum(1 for a in ans_naked if not a)

    print(f"\n[smoke] HEADROOM CHECK (the critical smoke question):")
    print(f"  unanswerable × PLAIN_NAKED violations: {unans_naked_v}/{len(unans_naked)}"
          f"  → headroom exists: {headroom}")
    print(f"  answerable  × PLAIN_NAKED violations: {ans_naked_v}/{len(ans_naked)}"
          f"  (expected small — source is given)")

    if headroom:
        print(f"\n[smoke] PASS — headroom exists on unanswerable regime; U4 full run is meaningful")
    else:
        print(f"\n[smoke] FLAG — Qwen NAKED abstains everywhere on unanswerable → "
              f"no headroom → CONTRACT payoff on this domain is UNTESTABLE")
        print(f"        Consider switching to a fabrication-prone domain (WP directive).")

    return {
        "n_sample":           len(sample),
        "per_cell":           {f"{k}|{c}": v for (k, c), v in per_cell.items()},
        "unans_naked_viols":  unans_naked_v,
        "ans_naked_viols":    ans_naked_v,
        "headroom":           headroom,
        **stats,
    }


# ══════════════════════════════════════════════════════════════════════════
# U4: Full run — stratified ~400 items × 3 conditions × N reps
#
# Deterministic sample is a SUPERSET of the U3 smoke sample (same seed),
# so the resume-set covers the 60 smoke triples automatically.
# ══════════════════════════════════════════════════════════════════════════

def cmd_run():
    """U4: 200 answerable + 200 unanswerable × 3 conditions × N_REPS calls."""
    items = _load_seed_items()
    sample = _pick_stratified(items, N_RUN_PER_KIND, seed=SEED)
    n_ans = sum(1 for r in sample if r["kind"] == "answerable")
    n_uns = sum(1 for r in sample if r["kind"] == "unanswerable")
    print(f"[run] sampled {len(sample)} items ({n_ans} ans + {n_uns} unans, seed={SEED})")
    stats = _run_harness(sample, RAW_RUNS_FILE, n_reps=N_REPS)
    return {"n_sample": len(sample), "n_ans": n_ans, "n_unans": n_uns, **stats}


# ══════════════════════════════════════════════════════════════════════════
# U5: Aggregate + gates PER REGIME
#
# Pre-registered floors LOCKED in code BEFORE aggregation runs:
#   DELTA_FLOOR = 0.05  absolute violation-rate difference
#   D_FLOOR     = 0.5   paired Cohen's d threshold
# Reason regime is required: WP explicitly directs "gate_contract_payoff
# reported PER REGIME"; the headline is the UNANSWERABLE cell.
# ══════════════════════════════════════════════════════════════════════════

DELTA_FLOOR = GATE_FLOOR  # 0.05, aliased for gate readability
D_FLOOR     = 0.5


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _paired_d(a: list, b: list):
    """Paired Cohen's d on same-length item vectors (a[i], b[i]).
    Returns None if std(diff)=0 (deterministic floor — signal is in Δ, not dispersion)."""
    diffs = [a[i] - b[i] for i in range(len(a))]
    s = _std(diffs)
    if s == 0:
        return None
    return _mean(diffs) / s


def _per_item_vec(runs: list, cond: str, kind: str) -> list:
    """Sorted-by-id vector of per-item violation flags (1.0=violation, 0.0=admissible)."""
    subset = [r for r in runs if r["condition"] == cond and r["kind"] == kind]
    by_item = {}
    for r in subset:
        by_item[r["id"]] = 0.0 if r["verifier_admissible"] else 1.0
    return [by_item[k] for k in sorted(by_item)]


def cmd_aggregate() -> dict:
    """U5: per-regime gates + write papers/results_payoff_squad.md."""
    runs = [json.loads(l) for l in open(RAW_RUNS_FILE)]

    regimes = ["answerable", "unanswerable"]
    # ── Per-(kind, condition) rates ───────────────────────────────────────
    cells = {}
    for kind in regimes:
        for cond in CONDITIONS:
            vec = _per_item_vec(runs, cond, kind)
            cells[(kind, cond)] = {
                "n":         len(vec),
                "n_viol":    int(sum(vec)),
                "V":         _mean(vec),
                "std_items": _std(vec),
                "vec":       vec,
            }

    # ── Paired contrasts (item-paired within regime) ───────────────────────
    contrasts = {}
    for kind in regimes:
        a = cells[(kind, "CONTRACT")]["vec"]
        b = cells[(kind, "PLAIN_FAIR")]["vec"]
        c = cells[(kind, "PLAIN_NAKED")]["vec"]
        contrasts[kind] = {
            "CONTRACT_vs_PLAIN_FAIR":  {"delta": _mean(a) - _mean(b), "d": _paired_d(a, b), "n": len(a)},
            "CONTRACT_vs_PLAIN_NAKED": {"delta": _mean(a) - _mean(c), "d": _paired_d(a, c), "n": len(a)},
            "PLAIN_FAIR_vs_PLAIN_NAKED": {"delta": _mean(b) - _mean(c), "d": _paired_d(b, c), "n": len(b)},
        }

    # ── PRE-REGISTERED gate per WP CONTRACT.B (Δ ≥ 0.05 alone) ────────────
    # Paired Cohen's d reported as a SUPPLEMENTAL diagnostic (see WP-6A precedent
    # for d ≥ 0.5 as a "strong effect-size" flag) — does NOT override the gate.
    gate_by_regime = {}
    for kind in regimes:
        c_vs_f = contrasts[kind]["CONTRACT_vs_PLAIN_FAIR"]
        rates = [cells[(kind, cond)]["V"] for cond in CONDITIONS]
        saturated = all(r < DELTA_FLOOR for r in rates)
        if saturated:
            verdict = "SATURATED"
            reason  = f"all V in {kind} regime < {DELTA_FLOOR}; no headroom to demonstrate payoff"
        elif c_vs_f["delta"] <= -DELTA_FLOOR:
            verdict = "PASS"
            reason  = f"Δ(CONTRACT − PLAIN_FAIR) = {c_vs_f['delta']:+.3f} ≤ −{DELTA_FLOOR}"
        else:
            verdict = "FAIL"
            reason  = (f"Δ(CONTRACT − PLAIN_FAIR) = {c_vs_f['delta']:+.3f}; "
                       f"need ≤ −{DELTA_FLOOR}")
        # Supplemental diagnostic: |d| ≥ D_FLOOR ⇒ strong paired effect size
        d_val = c_vs_f["d"]
        if d_val is None:
            strong_effect = None
            effect_note   = "deterministic paired-diff floor (std(diff)=0); Δ carries the signal"
        else:
            strong_effect = abs(d_val) >= D_FLOOR
            effect_note   = (f"|d| = {abs(d_val):.3f} "
                             f"{'≥' if strong_effect else '<'} "
                             f"{D_FLOOR} ({'strong' if strong_effect else 'moderate'} paired effect)")
        gate_by_regime[kind] = {
            "delta":         c_vs_f["delta"],
            "d":             d_val,
            "verdict":       verdict,      # per WP CONTRACT: Δ-only gate
            "reason":        reason,
            "saturated":     saturated,
            "strong_effect": strong_effect,  # supplemental diagnostic, NOT gate
            "effect_note":   effect_note,
        }

    # ── Baseline-fairness diagnostic (CD) ─────────────────────────────────
    # If PLAIN_FAIR ≈ CONTRACT, a fair prompt suffices (WP-6A-like null).
    # If PLAIN_NAKED alone was the outlier and PLAIN_FAIR already crossed,
    # note the contract's edge came from formatting, not boundary.
    baseline_fairness = {}
    for kind in regimes:
        f_vs_n = contrasts[kind]["PLAIN_FAIR_vs_PLAIN_NAKED"]  # negative = fair beats naked
        c_vs_f = contrasts[kind]["CONTRACT_vs_PLAIN_FAIR"]     # negative = contract beats fair
        f_reduced_naked = f_vs_n["delta"] <= -DELTA_FLOOR
        c_reduced_fair  = c_vs_f["delta"] <= -DELTA_FLOOR
        if f_reduced_naked and c_reduced_fair:
            note = "PLAIN_FAIR beats PLAIN_NAKED AND CONTRACT beats PLAIN_FAIR — additive gains"
        elif f_reduced_naked and not c_reduced_fair:
            note = "PLAIN_FAIR captured most of the boundary gain; CONTRACT does not add ≥ floor over fair prompt (WP-6A-like null)"
        elif not f_reduced_naked and c_reduced_fair:
            note = "PLAIN_FAIR did not reduce vs NAKED; CONTRACT's win is boundary-carried"
        else:
            note = "neither fair nor contract crossed floor over naked — regime may lack headroom"
        baseline_fairness[kind] = {
            "delta_fair_vs_naked":     f_vs_n["delta"],
            "delta_contract_vs_fair":  c_vs_f["delta"],
            "note": note,
        }

    # ── Assemble output ───────────────────────────────────────────────────
    eval_out = {
        "n_records":         len(runs),
        "cells":             {f"{k}|{c}": {kk: vv for kk, vv in v.items() if kk != "vec"}
                              for (k, c), v in cells.items()},
        "contrasts":         contrasts,
        "gate_by_regime":    gate_by_regime,
        "baseline_fairness": baseline_fairness,
        "pre_registered_floors": {"delta_floor": DELTA_FLOOR, "d_floor": D_FLOOR},
        "headline_regime":   "unanswerable",
    }
    with open(os.path.join(OUT_DIR, "eval_results.json"), "w") as f:
        json.dump(eval_out, f, indent=2, default=str)

    _write_results_md(eval_out, cells, contrasts, gate_by_regime, baseline_fairness)

    # Console summary
    print(f"[aggregate] Floors: Δ≥{DELTA_FLOOR}, |d|≥{D_FLOOR} (locked in code)")
    print(f"[aggregate] Cells:")
    for kind in regimes:
        for cond in CONDITIONS:
            c = cells[(kind, cond)]
            print(f"  V({kind:12s} × {cond:11s}) = {c['V']:.3f}  (n_viol={c['n_viol']}/{c['n']})")
    print(f"\n[aggregate] gate_contract_payoff (headline = unanswerable):")
    for kind in regimes:
        g = gate_by_regime[kind]
        print(f"  {kind:12s} → {g['verdict']:9s}  {g['reason']}")
    print(f"\n[aggregate] wrote {RESULTS_MD}")
    return eval_out


# ══════════════════════════════════════════════════════════════════════════
# U6 supplement: VOCABULARY-INDEPENDENT REAL-HALLUCINATION RE-SCORE
#
# The verifier's `must_abstain` check accepts a specific `abstain_markers`
# list — anyone reviewing the WP-15 payoff can reasonably ask whether the
# CONTRACT win is a vocabulary artifact (contract mentions the markers,
# Qwen parrots them; without the markers, "real" hallucination rate would
# tie). This re-score answers that with a broader, vocabulary-independent
# refusal detector on unanswerable rows — a real-hallucination flags iff
# the response contains NO refusal signal from a broad regex-based family
# of negation / uncertainty patterns.
#
# The re-score is reported alongside the verifier verdict — NOT as a
# replacement. If real-hallucination rates track the verifier within ~1pp,
# the payoff signal is not a marker-vocabulary artifact.
# ══════════════════════════════════════════════════════════════════════════

# Broad refusal patterns — vocabulary-independent of the verifier's markers
# list. Match any of these on the lowercased output → treat as refusal.
#
# Design: only patterns that indicate the model is REFUSING to answer, NOT
# patterns that appear in negated-answer fabrications ("Comcast did not
# give..." is a fabrication, not a refusal; "the passage does not state..."
# IS a refusal). The bare negation `\b(is|are|does|did|has) n't\b` was
# dropped in U6 rescore-tightening — it over-matched hedged fabrications.
_REFUSAL_PATTERNS = [
    # explicit "not stated / mentioned / provided / specified" family
    # — this is the core refusal signal, not any-negation.
    r"\bnot\s+(state|mention|specif|provid|indicat|address|discuss|explain|"
        r"describ|refer|clear|available|given|known|answer|shown|documented|"
        r"establish|confirm|say|contain)\w*",
    # "no information / mention / answer / data / details / indication / ..."
    r"\bno\s+(information|mention|answer|data|details|indication|reference|"
        r"specific|explicit|clear)\b",
    # "nothing (about|regarding|specific|explicit|mentioned)"
    r"\bnothing\s+(about|regarding|in|is|to|specific|explicit|mentioned)\b",
    # can't / cannot / unable
    r"\bcannot\b",
    r"\bcan(no|')t\b",
    r"\bunable\s+to\b",
    # unknown / unclear / unspecified / insufficient / n/a — as refusal words
    r"\b(unknown|unclear|unspecified|indeterminate|insufficient|ambiguous)\b",
    r"\bn\.?/?a\.?\b",
    # "the passage/text/context/source doesn't / does not / is not"
    r"\b(passage|text|context|source|excerpt)\s+(is|are|was|were|does|do|did)"
        r"\s*n[o']?t\b",
    r"\b(passage|text|context|source|excerpt)\s+(isn|aren|wasn|weren|doesn|don|didn)'?t\b",
    # explicit "I don't know / see / find" style
    r"\bdon'?t\s+(know|see|find|have)\b",
    # "cannot be determined / answered / inferred / found / located"
    r"\bcannot\s+be\s+(determined|answered|inferred|found|located)\b",
    # generic uncertain-hedge phrasing
    r"\b(unfortunately|regrettably|apologies|sorry)[,.]?\b.{0,80}\b(can|no|not|"
        r"unable|no information)\b",
]

_REFUSAL_REGEX = None


def _get_refusal_regex():
    global _REFUSAL_REGEX
    if _REFUSAL_REGEX is None:
        import re as _re
        joined = "|".join(f"(?:{p})" for p in _REFUSAL_PATTERNS)
        _REFUSAL_REGEX = _re.compile(joined, _re.IGNORECASE)
    return _REFUSAL_REGEX


def _broad_refusal_detected(text: str) -> bool:
    """True iff the text contains any pattern from the broad refusal family."""
    if not text:
        return False
    return bool(_get_refusal_regex().search(text))


def cmd_rescore():
    """U6: re-score unanswerable rows via a broad, vocabulary-independent
    refusal detector. Report real-hallucination rate per condition and
    compare to the verifier's `must_abstain` rate.

    Also compute an "extra-refusal" rate on answerable rows (some
    fraction of answerable violations are actually refusals in disguise —
    Qwen sometimes over-refuses; this is a diagnostic, not a re-score of
    the answerable regime, which is uninterpretable anyway).
    """
    runs = [json.loads(l) for l in open(RAW_RUNS_FILE)]

    def _rate(subset, keyfn):
        n = len(subset)
        v = sum(1 for r in subset if keyfn(r))
        return {"n": n, "n_hit": v, "rate": v / n if n else 0.0}

    print(f"[rescore] Loaded {len(runs)} records from {RAW_RUNS_FILE}")
    print(f"[rescore] Vocabulary-independent refusal detector: "
          f"{len(_REFUSAL_PATTERNS)} regex patterns")
    print()
    print("REAL-HALLUCINATION rate on UNANSWERABLE rows")
    print(f"  ('real hallucination' = qwen_output contains NO broad-refusal signal)")
    print()
    print(f"  {'condition':11s} {'n':4s} verifier_V  real_hall_V  delta   |  diff")
    for cond in CONDITIONS:
        subset = [r for r in runs if r["kind"] == "unanswerable" and r["condition"] == cond]
        ver_V   = _rate(subset, lambda r: not r["verifier_admissible"])
        real_V  = _rate(subset, lambda r: not _broad_refusal_detected(r["qwen_output"]))
        delta   = real_V["rate"] - ver_V["rate"]
        pp      = round(delta * 100, 1)
        print(f"  {cond:11s} {ver_V['n']:4d}  {ver_V['rate']:.3f}       "
              f"{real_V['rate']:.3f}      {delta:+.3f}  |  {pp:+.1f}pp")
    print()
    print("  Reading: if real_hall_V tracks verifier_V within ~1pp, the payoff signal")
    print("  is NOT a marker-vocabulary artifact — it's real hallucination reduction.")
    print()

    # Answerable diagnostic — how many "verifier violations" are actually
    # refusals in disguise (over-refusal by Qwen)?
    print("Diagnostic on ANSWERABLE (which is uninterpretable per U6 correction):")
    for cond in CONDITIONS:
        subset = [r for r in runs
                  if r["kind"] == "answerable" and r["condition"] == cond
                  and not r["verifier_admissible"]]
        n = len(subset)
        n_refusal = sum(1 for r in subset if _broad_refusal_detected(r["qwen_output"]))
        print(f"  {cond:11s}  verifier_violations={n:3d}  "
              f"of which broad-refusal-detected={n_refusal:3d} "
              f"({(n_refusal/n if n else 0):.1%})  "
              f"— remainder = paraphrase-FP-and-content-drift artifacts")
    return None


def _fmt_d(d):
    return "—" if d is None else f"{d:+.2f}"


def _write_results_md(out, cells, contrasts, gate_by_regime, baseline_fairness):
    regimes = ["answerable", "unanswerable"]
    lines = []
    lines.append("# WP-ST-15: Contract-conditioned Payoff on WP-14 Seed — Results")
    lines.append("")
    lines.append(f"**Subject:** local Qwen (qwen2.5:32b-instruct-q8_0), temp=0, Ollama backend  ")
    lines.append(f"**Corpus:** WP-14 SQuAD-v2 token-grounding seed "
                 f"(frozen hash `bdc404d760819e19`)  ")
    lines.append(f"**Sample:** 400 items (200 answerable + 200 unanswerable), 3 conditions, "
                 f"1 rep, 1200 records total, 0 errors  ")
    lines.append(f"**Verifier:** `scripts/contract_schema.py::verify` (CW+CD converged; "
                 f"token-grounding ¬B — anti-fabrication only; coverage_FN=100% documented)  ")
    lines.append(f"**Pre-registered gate (per WP CONTRACT.B):** Δ ≥ {DELTA_FLOOR} absolute — "
                 f"LOCKED in code before aggregation.  ")
    lines.append(f"**Supplemental diagnostic (not gate-overriding):** paired Cohen's d "
                 f"vs |d| ≥ {D_FLOOR} strong-effect threshold (WP-6A precedent).  ")
    lines.append(f"**Headline regime:** `unanswerable` (per WP directive — the only regime "
                 "where PLAIN_NAKED has meaningful headroom)")
    lines.append("")
    lines.append("---")
    lines.append("## Violation rate by condition × regime")
    lines.append("")
    lines.append("| Condition   | V(answerable) | n_viol / n | V(unanswerable) | n_viol / n |")
    lines.append("|---|---|---|---|---|")
    for cond in CONDITIONS:
        a = cells[("answerable", cond)]
        u = cells[("unanswerable", cond)]
        lines.append(f"| {cond} | {_fmt_pct(a['V'])} | {a['n_viol']}/{a['n']} | "
                     f"{_fmt_pct(u['V'])} | {u['n_viol']}/{u['n']} |")
    lines.append("")
    lines.append("---")
    lines.append("## Paired contrasts (item-paired within regime)")
    lines.append("")
    lines.append("Δ = V(A) − V(B). Negative Δ = A is BETTER (fewer violations).")
    lines.append("d = paired Cohen's d over per-item paired difference vector.")
    lines.append("")
    for kind in regimes:
        lines.append(f"### {kind}")
        lines.append("")
        lines.append("| Contrast | Δ | d | n |")
        lines.append("|---|---|---|---|")
        for name, c in contrasts[kind].items():
            lines.append(f"| {name} | {c['delta']:+.3f} | {_fmt_d(c['d'])} | {c['n']} |")
        lines.append("")
    lines.append("---")
    lines.append("## Pre-registered gate: `gate_contract_payoff` per regime")
    lines.append("")
    lines.append("`gate_contract_payoff` (per WP CONTRACT.B) = V(CONTRACT) < V(PLAIN_FAIR) "
                 f"by absolute floor Δ ≤ −{DELTA_FLOOR}. Paired Cohen's d reported as a "
                 "supplemental effect-size diagnostic (does NOT override the gate).")
    lines.append("")
    lines.append("| Regime | Δ(C − F) | d | Verdict (Δ-gate) | Effect-size flag |")
    lines.append("|---|---|---|---|---|")
    for kind in regimes:
        g = gate_by_regime[kind]
        strong = "—" if g["strong_effect"] is None else ("strong" if g["strong_effect"] else "moderate")
        lines.append(f"| **{kind}** | {g['delta']:+.3f} | {_fmt_d(g['d'])} | "
                     f"**{g['verdict']}** | {strong} |")
    lines.append("")
    for kind in regimes:
        g = gate_by_regime[kind]
        lines.append(f"- **{kind}**: {g['reason']}. Effect: {g['effect_note']}.")
    lines.append("")
    lines.append("---")
    lines.append("## Baseline-fairness diagnostic (CD)")
    lines.append("")
    lines.append("Question: did PLAIN_FAIR's neutral abstain-permitting hint already capture "
                 "most of the boundary gain, or does CONTRACT add real payoff over a fair "
                 "prompt? Reported per regime.")
    lines.append("")
    lines.append("| Regime | Δ(FAIR − NAKED) | Δ(CONTRACT − FAIR) | Interpretation |")
    lines.append("|---|---|---|---|")
    for kind in regimes:
        bf = baseline_fairness[kind]
        lines.append(f"| **{kind}** | {bf['delta_fair_vs_naked']:+.3f} | "
                     f"{bf['delta_contract_vs_fair']:+.3f} | {bf['note']} |")
    lines.append("")
    lines.append("---")
    lines.append("## Headline")
    lines.append("")
    headline = gate_by_regime["unanswerable"]
    lines.append(f"On the **unanswerable** regime — where headroom exists — "
                 f"`gate_contract_payoff` = **{headline['verdict']}** with "
                 f"Δ(CONTRACT − PLAIN_FAIR) = **{headline['delta']:+.3f}** "
                 f"(d = {_fmt_d(headline['d'])}).")
    lines.append("")
    lines.append("On the **answerable** regime, `gate_contract_payoff` = "
                 f"**{gate_by_regime['answerable']['verdict']}** with Δ = "
                 f"**{gate_by_regime['answerable']['delta']:+.3f}** — reported for "
                 "completeness; the verifier's token-grounding ¬B is strict on "
                 "natural-language paraphrases regardless of contract, so this cell "
                 "is not the WP's decisive headline.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Subject: local Qwen (qwen2.5:32b-instruct-q8_0) | 400 items × 3 "
                 f"conds × 1 rep | seed=42 | gate Δ≥{DELTA_FLOOR} (per WP CONTRACT) + "
                 f"|d|≥{D_FLOOR} supplemental | See papers/claim_payoff_squad.md for "
                 "the bounded claim (U6 — awaiting sign-off).*")

    os.makedirs(os.path.dirname(RESULTS_MD), exist_ok=True)
    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main entry ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-conditions", action="store_true",
                    help="U1 audit: show all 3 prompts on one answerable + one unanswerable row")
    ap.add_argument("--smoke",     action="store_true",
                    help="U3: 20 items × 3 conditions × 1 rep + headroom check")
    ap.add_argument("--run",       action="store_true",
                    help="U4: ~400 items × 3 conditions × N reps → raw_runs.jsonl")
    ap.add_argument("--aggregate", action="store_true",
                    help="U5: per-regime gates + results_payoff_squad.md")
    ap.add_argument("--rescore",   action="store_true",
                    help="U6: vocabulary-independent broad-refusal re-score (unanswerable)")
    args = ap.parse_args()

    if args.list_conditions:
        return cmd_list_conditions()
    if args.smoke:
        cmd_smoke()
    if args.run:
        cmd_run()
    if args.aggregate:
        cmd_aggregate()
    if args.rescore:
        cmd_rescore()
    if not (args.list_conditions or args.smoke or args.run or args.aggregate or args.rescore):
        print(f"[banner] pre-registration UTC: {PREREGISTRATION_UTC}")
        print(f"[banner] conditions: {CONDITIONS}")
        print(f"[banner] seed dir:   {SEED_DIR}")
        print(f"[banner] out dir:    {OUT_DIR}  (local-only, gitignored)")
        print("[banner] --list-conditions shows the 3 frozen prompts on real rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
