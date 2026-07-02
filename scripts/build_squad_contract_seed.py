#!/usr/bin/env python3
"""WP-ST-14: SQuAD v2 → TOKEN-GROUNDING ANTI-FABRICATION Task-contract seed.

Turns SQuAD v2 rows into contract-shaped items whose ¬B boundary is machine-
checkable by scripts/contract_schema.py::verify (CW+CD converged patch). NO LLM
in the loop — this WP produces training-data-shaped items, not LLM outputs.

Two contract kinds, derived from SQuAD v2's own structure:

  - answerable   (answers.text non-empty)
        → facts-only Task contract:
          A = {source: context, query: question}
          F = "answer only from source (token-grounding)"
          B = {content_tokens_must_be_grounded_in_source: context}
          P = ["closed world: only the source counts as evidence"]
          ¬B = {content-tokens absent from source = FABRICATION}

  - unanswerable (answers.text empty)
        → abstain contract:
          A = {source: context, query: question}
          F = "abstain if source is silent on the query"
          B = {must_abstain: True, abstain_markers: [...]}
          P = ["closed world"]
          ¬B = {any specific asserted answer, i.e. no abstain marker present}

HONEST NAMING (CW+CD converged, 2026-07-01): this is a TOKEN-GROUNDING anti-
fabrication seed, NOT a "correct facts-only" seed. The ¬B catches fabricated
CONTENT (tokens absent from source). The ¬B does NOT catch semantic errors:
role-swap (all source tokens, wrong roles), negation, temporal/relation
distortion, synonym/morphology drift ('famous' vs 'fame'). Two blind spots
are documented in contract_schema._test_grounding (morphology FP + role-swap FN)
and re-surfaced by U5's coverage probe — the seed publishes these limits.

Each item is filtered through SELF-VERIFY probes:
  answerable (4 real + 1 coverage):
    - gold answer                    → MUST be admissible
    - grounded paraphrase            → MUST be admissible (source-token rearrangement)
    - realistic near-miss fabrication → MUST be violation (plausible wrong entity/year)
    - sentinel fabrication            → MUST be violation (distinctive glyph pattern)
    - coverage probe (role-swap)     → EXPECTED admissible = documented false-negative
                                        (reported, NOT used for retain)
  unanswerable (2 real):
    - gold-abstain phrase → MUST be admissible
    - fabricated answer   → MUST be violation

Items where all applicable REAL probes fire correctly are retained. Coverage
FN admissions are counted so the seed publishes its blind-spot rate.

Env note: prefer `.venv` pip when a project venv is present; fall back to
`python3 -m pip install ... --break-system-packages` only when no venv is
available. (This module has no install action of its own — env prep happens
at U1 via shell.)

Usage:
  python scripts/build_squad_contract_seed.py --precheck        # U1: 20-row grounding sanity
  python scripts/build_squad_contract_seed.py --precheck-full   # U1: ≥1000-row grounding percentage
  python scripts/build_squad_contract_seed.py --smoke           # U6: 50-item end-to-end + raw report
  python scripts/build_squad_contract_seed.py --run             # U7: ~2k full run → items/train/test + hash

Discipline: execute-first; raw numbers before conclusions; smoke-before-full;
frozen hash over train.jsonl+test.jsonl (per cbt/fingerprint.compute_dataset_hash);
raw HF corpus gitignored, only seed + splits + hash + this script committed.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

# ── Reuse repo utilities (READ-ONLY imports) ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contract_schema import verify, _norm, _content_tokens  # noqa: E402


# ── Constants (WP-ST-14 scoped) ───────────────────────────────────────────
DATA_DIR    = "data/contract_squad"
ITEMS_FILE  = os.path.join(DATA_DIR, "items.jsonl")
FROZEN_FILE = os.path.join(DATA_DIR, "frozen_dataset_hash.json")

# HuggingFace dataset id (public)
HF_DATASET  = "rajpurkar/squad_v2"

# Deterministic sampling seed — locked across all runs, do NOT tune
SEED        = 42

# Precheck / smoke / full sample sizes (per directive: 20 / 50 / ~2000)
N_PRECHECK           = 20   # 10 answerable + 10 unanswerable (smoke — grounding invariant)
N_PRECHECK_FULL      = 1000 # answerable-only rows from train split — the ≥95% grounding claim
N_SMOKE_PER_KIND     = 25   # 25 answerable + 25 unanswerable = 50
N_RUN_TARGET         = 2000 # ~1000 answerable + ~1000 unanswerable (stratified)


# ── U4: Probe generators (module-level, deterministic) ────────────────────
# Sentinel fabrication — a distinctive glyph pattern absent from natural text.
# NOTE (U7 iteration, 2026-07-01): the initial sentinel "Zqx-Faux-Entity-7X-4291"
# tokenised to content-tokens {"zqx", "faux", "entity", "4291"} — "entity" and
# "faux" collided with 7/2000 SQuAD train contexts (Latin/English adjectives
# appear in real prose). Strengthened to pure random-glyph clusters below:
# content-tokens are all consonant-heavy nonsense strings + a distinctive
# 5-digit numeric that never appears in natural text.
FAB_SPAN   = "Xqzfvbm-Kjplmr-Wntzyq-84391"
FAB_ANSWER = f"The answer is {FAB_SPAN}."

# Near-miss fabrication candidates: plausible-looking wrong entities/years
# with unusual content-tokens (rarely appear in SQuAD contexts). Try each in
# order; first candidate whose content-tokens are disjoint from source
# content-tokens is used. Sentinel is the ultimate fallback.
NEAR_MISS_ENTITIES = [
    "Nikolai Petrov",   # Slavic proper-noun pair
    "Emily Vance",      # rare Anglo pair
    "Rajiv Menon",      # South-Asian pair
    "Zora Lindberg",    # unusual pair
    "1543",             # year shape — pre-modern, rarely in SQuAD contexts
    "2287",             # year shape — future, absent from historical corpora
]


def _sentinel_collides(context: str) -> bool:
    """True iff any FAB_SPAN content-token is already present in context content-tokens.
    U6 smoke asserts this is False on every row (else the sentinel isn't a real fabrication)."""
    return not _content_tokens(FAB_SPAN).isdisjoint(_content_tokens(context))


def make_near_miss(row: dict) -> str:
    """Realistic near-miss fabrication (answerable only).

    Returns a string whose content-tokens contain ≥1 token NOT in the source
    content-tokens → the verifier MUST classify as violation. Deterministic:
    same input row → same output string. Falls back to FAB_ANSWER only when
    every entity in NEAR_MISS_ENTITIES collides with source (rare).
    """
    ctx_tokens = _content_tokens(row["context"])
    for candidate in NEAR_MISS_ENTITIES:
        cand_tokens = _content_tokens(candidate)
        if cand_tokens and cand_tokens.isdisjoint(ctx_tokens):
            return f"The answer is {candidate}."
    return FAB_ANSWER  # fallback: sentinel


def make_grounded_paraphrase(row: dict) -> str:
    """Grounded paraphrase (answerable only) — source-token re-arrangement
    containing the gold answer.

    Strategy: find the sentence in context that contains the gold answer,
    return it with a trailing period. Content-tokens ⊆ source content-tokens
    (trivially — sentence is subset of context) → grounding admits. May fail
    strict substring at the trailing-punctuation boundary → demonstrates the
    substring/grounding delta empirically.
    """
    gold = row["answers"]["text"][0]
    ctx  = row["context"]
    sentences = re.split(r"(?<=[.!?])\s+", ctx)
    for sent in sentences:
        if gold.lower() in sent.lower():
            sent = sent.strip()
            if not sent.endswith((".", "!", "?")):
                sent += "."
            return sent
    # Fallback: gold + trailing period
    return gold.strip() + "."


def make_coverage_probe(row: dict) -> str | None:
    """Role-swap coverage probe (answerable only) — uses ONLY source content-tokens.

    Verifier is EXPECTED to admit (documented false-negative). This is NOT
    used to filter items; the admission count is REPORTED as the seed's
    published blind spot. Returns None when the source has fewer than 2
    content-tokens (rare) → coverage measurement skipped for that row.
    """
    ctx_tokens = list(_content_tokens(row["context"]))
    if len(ctx_tokens) < 2:
        return None
    # Pick 2 salient (long) tokens deterministically: longest first, then alphabetical
    picks = sorted(ctx_tokens, key=lambda t: (-len(t), t))[:2]
    tok_a, tok_b = picks
    # Role-swap-ish template — token order inverted vs typical source arrangement.
    # "was" is a stopword → not scored; content_tokens(output) = {tok_a, tok_b} ⊆ source.
    return f"{tok_b} was {tok_a}."


# ── U3: Contract builder (dispatch by row kind) ───────────────────────────
ABSTAIN_MARKERS = [
    "not stated", "does not", "not mentioned", "no information",
    "unknown", "cannot", "cannot be determined", "insufficient",
    "does not say",
]

# The single representative abstain phrase used by the U5 gold-abstain probe.
# Chosen because it contains multiple markers ("does not", "does not say")
# and is a plausible model output for an abstain contract.
GOLD_ABSTAIN_PHRASE = "The passage does not say."


def build_contract(row: dict) -> dict:
    """SQuAD row → real contract (facts-only or abstain, by row kind).

    CW+CD converged shape (2026-07-01):

      - answerable   → B.content_tokens_must_be_grounded_in_source = context
                        (main anti-fabrication check; ¬B is carried by B itself)
      - unanswerable → B.must_abstain = True + abstain_markers = [...]
                        (must_abstain IS the real check; not_B blacklist NOT
                        relied upon as primary fabrication catch)

    kind field is EXPLICIT so downstream consumers don't have to sniff B.
    """
    is_answerable = bool(row["answers"]["text"])
    common = {
        "id":         row["id"],
        "level":      "Task",
        "nl_prompt":  row["question"],
        "source":     row["context"],
        "check":      "predicate_spec",
        "A":          {"source": row["context"], "query": row["question"]},
        "P":          ["closed world"],
    }
    if is_answerable:
        return {
            **common,
            "kind":  "answerable",
            "F":     "answer only from source (token-grounding)",
            "B":     {"content_tokens_must_be_grounded_in_source": row["context"]},
            "not_B": {"must_not_contain_any": []},  # empty by design; B carries ¬B
        }
    return {
        **common,
        "kind":  "unanswerable",
        "F":     "abstain if source is silent on the query",
        "B":     {"must_abstain": True, "abstain_markers": ABSTAIN_MARKERS},
        "not_B": {"must_not_contain_any": []},  # empty by design; must_abstain in B is the real check
    }


# ── U5: Self-verify probe runner (CW+CD converged) ────────────────────────
# answerable  : 4 REAL probes set `retain` (gold+paraphrase→adm, near-miss+sentinel→viol)
#               + 1 COVERAGE probe (role-swap, EXPECTED admitted = documented FN;
#               NOT used for retain; reported so seed publishes its blind spot)
# unanswerable: 2 REAL probes set `retain` (gold-abstain→adm, fab-answer→viol);
#               no coverage probe (no token-grounding analogue for must_abstain)

def self_verify(row: dict, contract: dict) -> dict:
    """Run the applicable probe subset for the contract's kind (CW+CD converged).

    Returns:
      {
        probes: {name: {admissible, reason}, ...},
        retain: bool,                     # 4 real probes fire correctly (answerable)
                                          # 2 real probes fire correctly (unanswerable)
        reason: str,                      # 'OK' or which real probe(s) misfired
        coverage_admitted: bool | None,   # True = BLIND SPOT #2 reproduced (documented FN)
                                          # None = N/A (unanswerable or <2 source tokens)
        near_miss_fell_back: bool | None, # True = near-miss fell back to sentinel
                                          # None = N/A (unanswerable)
      }
    """
    if contract["kind"] == "answerable":
        gold  = row["answers"]["text"][0]
        para  = make_grounded_paraphrase(row)
        nm    = make_near_miss(row)
        cov   = make_coverage_probe(row)

        p_gold = verify(gold,     contract)
        p_para = verify(para,     contract)
        p_nm   = verify(nm,       contract)
        p_sent = verify(FAB_SPAN, contract)
        p_cov  = verify(cov, contract) if cov is not None else None

        real_ok = (
            p_gold["admissible"]
            and p_para["admissible"]
            and (not p_nm["admissible"])
            and (not p_sent["admissible"])
        )
        reason_parts = []
        if not p_gold["admissible"]:
            reason_parts.append(f"gold NOT admissible ({p_gold['reason']})")
        if not p_para["admissible"]:
            reason_parts.append(f"paraphrase NOT admissible ({p_para['reason']})")
        if p_nm["admissible"]:
            reason_parts.append(f"near-miss admitted (want violation): {p_nm['reason']}")
        if p_sent["admissible"]:
            reason_parts.append(f"sentinel admitted (want violation): {p_sent['reason']}")
        reason = "OK" if real_ok else "; ".join(reason_parts)

        probes = {
            "p_gold":       p_gold,
            "p_paraphrase": p_para,
            "p_near_miss":  p_nm,
            "p_sentinel":   p_sent,
        }
        if p_cov is not None:
            probes["p_coverage"] = p_cov

        return {
            "probes":              probes,
            "retain":              real_ok,
            "reason":              reason,
            "coverage_admitted":   (p_cov["admissible"] if p_cov is not None else None),
            "near_miss_fell_back": (nm == FAB_ANSWER),
        }

    # ── unanswerable: 2 real probes (no coverage on this shape) ──
    p_gold = verify(GOLD_ABSTAIN_PHRASE, contract)
    p_fab  = verify(FAB_ANSWER,          contract)
    real_ok = p_gold["admissible"] and (not p_fab["admissible"])
    reason_parts = []
    if not p_gold["admissible"]:
        reason_parts.append(f"gold-abstain NOT admissible ({p_gold['reason']})")
    if p_fab["admissible"]:
        reason_parts.append(f"fab-answer admitted (want violation): {p_fab['reason']}")
    reason = "OK" if real_ok else "; ".join(reason_parts)

    return {
        "probes":              {"p_gold_abstain": p_gold, "p_fab_answer": p_fab},
        "retain":              real_ok,
        "reason":              reason,
        "coverage_admitted":   None,  # N/A for unanswerable
        "near_miss_fell_back": None,  # N/A for unanswerable
    }


# ── U1: SQuAD loader + substring pre-check ────────────────────────────────
def _load_split(split: str):
    """Load rajpurkar/squad_v2 split via HuggingFace datasets."""
    from datasets import load_dataset  # local import → clean --help without deps
    return load_dataset(HF_DATASET, split=split)


def _pick_stratified(ds, n_per_kind: int, seed: int) -> tuple:
    """Deterministic stratified sample: n_per_kind answerable + n_per_kind unanswerable."""
    rng = random.Random(seed)
    ans_indices, unans_indices = [], []
    for i, row in enumerate(ds):
        (ans_indices if row["answers"]["text"] else unans_indices).append(i)
    rng.shuffle(ans_indices)
    rng.shuffle(unans_indices)
    ans_pick   = ans_indices[:n_per_kind]
    unans_pick = unans_indices[:n_per_kind]
    return ans_pick, unans_pick


def _grounding_check(gold: str, context: str) -> tuple:
    """CW+CD converged: `_content_tokens(gold) ⊆ _content_tokens(context)`.

    Returns (grounded: bool, ungrounded_tokens: set) using the SAME helper
    the verifier uses — single source of truth, no drift risk.
    """
    ungrounded = _content_tokens(gold) - _content_tokens(context)
    return (len(ungrounded) == 0, ungrounded)


def cmd_precheck():
    """U1 smoke: grounding-invariant sanity on 10 answerable + 10 unanswerable.

    The invariant: `_content_tokens(gold_answer) ⊆ _content_tokens(context)`.
    Grounding is a SUPERSET of substring — any substring is trivially grounded,
    but a source-token paraphrase is grounded and not a substring. Substring
    counts are also reported for historical continuity.
    """
    print(f"[precheck] loading {HF_DATASET} validation split ...")
    ds = _load_split("validation")
    print(f"[precheck] loaded {len(ds)} rows")

    ans_idx, unans_idx = _pick_stratified(ds, N_PRECHECK // 2, seed=SEED)

    grounding_pass = 0
    substring_pass = 0
    fails = []
    for i in ans_idx:
        row  = ds[i]
        gold = row["answers"]["text"][0]
        # Grounding invariant (main check)
        grounded, ungrounded = _grounding_check(gold, row["context"])
        if grounded:
            grounding_pass += 1
        else:
            fails.append(("grounding", row["id"],
                          f"gold={gold!r} ungrounded={sorted(ungrounded)}",
                          row["context"][:120]))
        # Substring (historical / continuity)
        if _norm(gold).strip() in _norm(row["context"]).strip():
            substring_pass += 1

    unanswerable_empty_pass = 0
    for i in unans_idx:
        row = ds[i]
        if row["answers"]["text"] == []:
            unanswerable_empty_pass += 1
        else:
            fails.append(("unans_not_empty", row["id"], str(row["answers"]), ""))

    n   = len(ans_idx)
    n_u = len(unans_idx)
    print(f"[precheck] grounding_smoke:      {grounding_pass}/{n}")
    print(f"[precheck] substring_smoke:      {substring_pass}/{n}  (historical; grounding is main check)")
    print(f"[precheck] unanswerable_empty:   {unanswerable_empty_pass}/{n_u}")

    all_pass = (
        grounding_pass == n
        and unanswerable_empty_pass == n_u
    )
    if all_pass:
        print(f"[precheck] SMOKE PASS — all {n + n_u} sanity checks OK "
              f"(run --precheck-full for the ≥95% claim)")
    else:
        print(f"[precheck] SMOKE FAIL — {len(fails)} rows broke assumption:")
        for kind, rid, detail, ctx in fails[:5]:
            print(f"           [{kind}] id={rid}  {detail}  ctx_head={ctx!r}")

    return {"n_ans": n, "n_unans": n_u,
            "grounding_pass": grounding_pass,
            "substring_pass": substring_pass,
            "unanswerable_empty_pass": unanswerable_empty_pass,
            "smoke_pass": all_pass,
            "fails": fails[:20]}


def cmd_precheck_full(n_target: int = N_PRECHECK_FULL):
    """U1 full: grounding invariant on ≥1000 answerable rows from train split.

    This is the sample the ≥95% grounding claim actually requires. Reports
    a percentage and lists the first few rows that broke the invariant (if
    any) so U3's answerable contract shape can be re-scoped if needed.
    """
    print(f"[precheck-full] loading {HF_DATASET} train split ...")
    ds = _load_split("train")
    print(f"[precheck-full] loaded {len(ds)} rows")

    rng = random.Random(SEED)
    ans_indices = [i for i, row in enumerate(ds) if row["answers"]["text"]]
    rng.shuffle(ans_indices)
    sample = ans_indices[:n_target]
    n = len(sample)

    grounded = 0
    substring = 0
    fails = []
    for i in sample:
        row  = ds[i]
        gold = row["answers"]["text"][0]
        ok, ungrounded = _grounding_check(gold, row["context"])
        if ok:
            grounded += 1
        else:
            fails.append({"id": row["id"], "gold": gold,
                          "ungrounded": sorted(ungrounded),
                          "context_head": row["context"][:120]})
        if _norm(gold).strip() in _norm(row["context"]).strip():
            substring += 1

    pct  = grounded / n if n else 0.0
    pctS = substring / n if n else 0.0
    print(f"[precheck-full] grounding_pass:  {grounded}/{n}  ({pct:.3%})")
    print(f"[precheck-full] substring_pass:  {substring}/{n}  ({pctS:.3%})  [historical, for delta]")
    print(f"[precheck-full] grounding − substring gap = {grounded - substring} rows "
          f"(rows the grounding check accepts that substring rejects)")

    threshold = 0.95
    if pct >= threshold:
        print(f"[precheck-full] PASS — grounding rate {pct:.3%} ≥ {threshold:.0%} "
              f"→ U3's answerable contract shape (content-token grounding) is valid")
    else:
        print(f"[precheck-full] FAIL — grounding rate {pct:.3%} < {threshold:.0%} "
              f"→ U3 must re-scope the answerable contract; showing first 5 failures:")
        for f in fails[:5]:
            print(f"           id={f['id']} gold={f['gold']!r} ungrounded={f['ungrounded']}")

    return {"n": n, "grounded": grounded, "substring": substring,
            "grounding_rate": pct, "substring_rate": pctS,
            "pass_95pct": pct >= threshold, "fails": fails[:20]}


# ── U6: End-to-end smoke — 50 items (25 ans + 25 unans) with raw report ───

SMOKE_OUT_PATH = "/tmp/squad_smoke.jsonl"


def cmd_smoke():
    """U6: 50-item end-to-end smoke on validation split.

    Prints: contract JSONs (3 ans + 3 unans), per-item probe verdicts, and a
    summary row with retention per kind + coverage FN count + near-miss
    fallback count + sentinel-collision count.
    """
    print(f"[smoke] loading {HF_DATASET} validation split ...")
    ds = _load_split("validation")
    print(f"[smoke] loaded {len(ds)} rows")

    ans_idx, unans_idx = _pick_stratified(ds, N_SMOKE_PER_KIND, seed=SEED)
    print(f"[smoke] sampled {len(ans_idx)} answerable + {len(unans_idx)} unanswerable "
          f"(seed={SEED})")

    # Sentinel-collision guard — assert BEFORE any per-row work
    coll = 0
    for i in ans_idx + unans_idx:
        if _sentinel_collides(ds[i]["context"]):
            coll += 1
    if coll > 0:
        print(f"[smoke] HALT — sentinel collides with {coll} contexts (FAB_SPAN reusable)")
        return None
    print(f"[smoke] sentinel_collisions:  0/{N_SMOKE_PER_KIND * 2} (guard clean)")

    # First 3 answerable + 3 unanswerable contract JSONs (eyeball surface)
    print("\n[smoke] Contract JSON — first 3 answerable:")
    for i in ans_idx[:3]:
        c = build_contract(ds[i])
        print(f"  --- id={c['id']} kind={c['kind']} ---")
        print(json.dumps({k: c[k] for k in ("kind", "F", "B", "not_B")}, indent=2))
    print("\n[smoke] Contract JSON — first 3 unanswerable:")
    for i in unans_idx[:3]:
        c = build_contract(ds[i])
        print(f"  --- id={c['id']} kind={c['kind']} ---")
        print(json.dumps({k: c[k] for k in ("kind", "F", "B", "not_B")}, indent=2))

    # Per-row self_verify + tally
    retained_ans      = []
    retained_unans    = []
    cov_admitted_ans  = 0
    nm_fallback_ans   = 0

    print(f"\n[smoke] Per-item probe verdicts (answerable):")
    print(f"  {'id':30s} p_gold p_para p_nm  p_sent p_cov retain")
    for i in ans_idx:
        row = ds[i]
        c   = build_contract(row)
        r   = self_verify(row, c)
        p   = r["probes"]
        g   = "T" if p["p_gold"]["admissible"]       else "F"
        pa  = "T" if p["p_paraphrase"]["admissible"] else "F"
        nm  = "T" if p["p_near_miss"]["admissible"]  else "F"
        se  = "T" if p["p_sentinel"]["admissible"]   else "F"
        cv  = "-" if "p_coverage" not in p else ("T" if p["p_coverage"]["admissible"] else "F")
        rt  = "Y" if r["retain"] else "n"
        print(f"  {row['id'][:28]:30s}   {g}      {pa}     {nm}     {se}     {cv}     {rt}")
        if r["retain"]:
            retained_ans.append({
                "id":        row["id"],
                "kind":      c["kind"],
                "nl_prompt": c["nl_prompt"],
                "source":    c["source"],
                "contract":  c,
                "probes":    p,
                "coverage_admitted":   r["coverage_admitted"],
                "near_miss_fell_back": r["near_miss_fell_back"],
            })
        if r["coverage_admitted"]:
            cov_admitted_ans += 1
        if r["near_miss_fell_back"]:
            nm_fallback_ans += 1

    print(f"\n[smoke] Per-item probe verdicts (unanswerable):")
    print(f"  {'id':30s} p_gold_abstain p_fab_ans retain")
    for i in unans_idx:
        row = ds[i]
        c   = build_contract(row)
        r   = self_verify(row, c)
        p   = r["probes"]
        g   = "T" if p["p_gold_abstain"]["admissible"] else "F"
        f_  = "T" if p["p_fab_answer"]["admissible"]   else "F"
        rt  = "Y" if r["retain"] else "n"
        print(f"  {row['id'][:28]:30s}        {g}              {f_}      {rt}")
        if r["retain"]:
            retained_unans.append({
                "id":        row["id"],
                "kind":      c["kind"],
                "nl_prompt": c["nl_prompt"],
                "source":    c["source"],
                "contract":  c,
                "probes":    p,
                "coverage_admitted":   None,
                "near_miss_fell_back": None,
            })

    # Summary row
    n_ans   = len(ans_idx)
    n_unans = len(unans_idx)
    r_ans   = len(retained_ans)
    r_unans = len(retained_unans)
    overall = (r_ans + r_unans) / (n_ans + n_unans) if (n_ans + n_unans) else 0.0
    cov_rate = cov_admitted_ans / n_ans if n_ans else 0.0
    nm_rate  = nm_fallback_ans  / n_ans if n_ans else 0.0

    print(f"\n[smoke] SUMMARY:")
    print(f"  loaded={n_ans + n_unans} | answerable={n_ans} unanswerable={n_unans}")
    print(f"  retained_ans={r_ans}/{n_ans} retained_unans={r_unans}/{n_unans} "
          f"overall_retention={overall:.1%}")
    print(f"  coverage_FN_ans={cov_admitted_ans}/{n_ans} (rate={cov_rate:.1%})  "
          f"[BLIND SPOT #2 — reported, NOT filter]")
    print(f"  near_miss_fallbacks={nm_fallback_ans}/{n_ans} (rate={nm_rate:.1%})  "
          f"[shape U4 missed if > 10%]")
    print(f"  sentinel_collisions=0/{n_ans + n_unans}  [guard clean]")

    # Flag design issues (report-only; U7 pauses only if human deems necessary)
    flags = []
    if n_ans   and r_ans   / n_ans   < 0.80:
        flags.append(f"answerable retention {r_ans}/{n_ans} < 80%")
    if n_unans and r_unans / n_unans < 0.80:
        flags.append(f"unanswerable retention {r_unans}/{n_unans} < 80%")
    if n_ans   and nm_rate                 > 0.10:
        flags.append(f"near-miss fallback rate {nm_rate:.1%} > 10%")
    if flags:
        print(f"\n[smoke] FLAGS (U6 audit — review before U7):")
        for f in flags:
            print(f"  - {f}")
    else:
        print(f"\n[smoke] PASS — no retention/fallback flags; ready for U7")

    # Write retained items to /tmp (NOT committed)
    with open(SMOKE_OUT_PATH, "w") as f:
        for rec in retained_ans + retained_unans:
            f.write(json.dumps(rec) + "\n")
    print(f"\n[smoke] wrote {len(retained_ans) + len(retained_unans)} retained items → {SMOKE_OUT_PATH}")

    return {
        "n_ans":            n_ans,
        "n_unans":          n_unans,
        "r_ans":            r_ans,
        "r_unans":          r_unans,
        "overall":          overall,
        "cov_admitted_ans": cov_admitted_ans,
        "nm_fallback_ans":  nm_fallback_ans,
        "sentinel_coll":    0,
        "flags":            flags,
    }


# ── U7: Full ~2k run + train/test split + frozen hash ────────────────────

TRAIN_FRAC = 0.80  # 80/20 split


def _stratified_split(records: list, seed: int, train_frac: float = TRAIN_FRAC) -> tuple:
    """Deterministic stratified split by `kind` field."""
    rng = random.Random(seed)
    by_kind: dict = {}
    for rec in records:
        by_kind.setdefault(rec["kind"], []).append(rec)
    train, test = [], []
    for kind, group in sorted(by_kind.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        cut = int(round(len(shuffled) * train_frac))
        train.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def cmd_run():
    """U7: full ~2000-item build → items.jsonl + train.jsonl + test.jsonl + frozen hash.

    Deterministic stratified sample (~1000 ans + ~1000 unans, seed=42) from
    train split; runs the full self_verify → filter; writes items.jsonl (all
    retained) + 80/20 train/test split (stratified by kind); freezes hash
    via cbt.fingerprint.freeze_dataset_hash (which hashes train.jsonl+test.jsonl).
    """
    from cbt.fingerprint import compute_dataset_hash, freeze_dataset_hash

    print(f"[run] loading {HF_DATASET} train split ...")
    ds = _load_split("train")
    print(f"[run] loaded {len(ds)} rows")

    n_per_kind = N_RUN_TARGET // 2
    ans_idx, unans_idx = _pick_stratified(ds, n_per_kind, seed=SEED)
    print(f"[run] sampled {len(ans_idx)} answerable + {len(unans_idx)} unanswerable "
          f"(seed={SEED})")

    coll = 0
    for i in ans_idx + unans_idx:
        if _sentinel_collides(ds[i]["context"]):
            coll += 1
    if coll > 0:
        print(f"[run] HALT — sentinel collides with {coll} contexts")
        return None
    print(f"[run] sentinel_collisions: 0/{len(ans_idx) + len(unans_idx)}")

    retained         = []
    cov_admitted_ans = 0
    nm_fallback_ans  = 0
    for i in ans_idx + unans_idx:
        row = ds[i]
        c   = build_contract(row)
        r   = self_verify(row, c)
        if not r["retain"]:
            continue
        rec = {
            "id":        row["id"],
            "kind":      c["kind"],
            "nl_prompt": c["nl_prompt"],
            "source":    c["source"],
            "contract":  c,
            "probes":    r["probes"],
            "coverage_admitted":   r["coverage_admitted"],
            "near_miss_fell_back": r["near_miss_fell_back"],
        }
        retained.append(rec)
        if c["kind"] == "answerable":
            if r["coverage_admitted"]:
                cov_admitted_ans += 1
            if r["near_miss_fell_back"]:
                nm_fallback_ans += 1

    r_ans   = sum(1 for r in retained if r["kind"] == "answerable")
    r_unans = sum(1 for r in retained if r["kind"] == "unanswerable")
    n_ans_scanned   = len(ans_idx)
    n_unans_scanned = len(unans_idx)

    train, test = _stratified_split(retained, seed=SEED)
    train_ans   = sum(1 for r in train if r["kind"] == "answerable")
    train_unans = sum(1 for r in train if r["kind"] == "unanswerable")
    test_ans    = sum(1 for r in test  if r["kind"] == "answerable")
    test_unans  = sum(1 for r in test  if r["kind"] == "unanswerable")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ITEMS_FILE, "w") as f:
        for rec in retained:
            f.write(json.dumps(rec) + "\n")
    train_path = os.path.join(DATA_DIR, "train.jsonl")
    test_path  = os.path.join(DATA_DIR, "test.jsonl")
    with open(train_path, "w") as f:
        for rec in train:
            f.write(json.dumps(rec) + "\n")
    with open(test_path, "w") as f:
        for rec in test:
            f.write(json.dumps(rec) + "\n")

    h = compute_dataset_hash(DATA_DIR)
    freeze_path = freeze_dataset_hash(
        DATA_DIR, h,
        notes="WP-ST-14 SQuAD v2 token-grounding anti-fabrication seed",
        probe_results={
            "n_items_total":     len(retained),
            "n_train":           len(train),
            "n_test":            len(test),
            "n_retained_ans":    r_ans,
            "n_retained_unans":  r_unans,
            "n_scanned_ans":     n_ans_scanned,
            "n_scanned_unans":   n_unans_scanned,
            "coverage_FN_ans":   cov_admitted_ans,
            "near_miss_fallbacks_ans": nm_fallback_ans,
            "sentinel_collisions": 0,
            "split_seed":        SEED,
            "train_frac":        TRAIN_FRAC,
            "hf_dataset":        HF_DATASET,
            "hf_split":          "train",
        },
    )

    print(f"\n[run] RAW REPORT")
    print(f"  loaded={n_ans_scanned + n_unans_scanned}  "
          f"answerable_scanned={n_ans_scanned}  unanswerable_scanned={n_unans_scanned}")
    print(f"  retained_ans={r_ans}/{n_ans_scanned}  "
          f"retained_unans={r_unans}/{n_unans_scanned}  "
          f"total_retained={len(retained)}")
    ret_ans_pct  = r_ans   / n_ans_scanned   if n_ans_scanned   else 0
    ret_uns_pct  = r_unans / n_unans_scanned if n_unans_scanned else 0
    print(f"  retention_ans={ret_ans_pct:.1%}  retention_unans={ret_uns_pct:.1%}")
    cov_rate = cov_admitted_ans / r_ans if r_ans else 0
    nm_rate  = nm_fallback_ans  / r_ans if r_ans else 0
    print(f"  coverage_FN_ans={cov_admitted_ans}/{r_ans} (rate={cov_rate:.1%})  "
          f"[BLIND SPOT #2 — published blind-spot floor]")
    print(f"  near_miss_fallbacks_ans={nm_fallback_ans}/{r_ans} (rate={nm_rate:.1%})")
    print(f"  sentinel_collisions=0/{n_ans_scanned + n_unans_scanned}")
    print(f"\n  train={len(train)} (ans={train_ans}, unans={train_unans})")
    print(f"  test ={len(test)}  (ans={test_ans}, unans={test_unans})")
    print(f"\n  files:")
    print(f"    {ITEMS_FILE}  ({len(retained)} items)")
    print(f"    {train_path}  ({len(train)} items)")
    print(f"    {test_path}   ({len(test)} items)")
    print(f"    {freeze_path}  (frozen)")
    print(f"\n  frozen_dataset_hash: {h}")

    return {
        "hash":            h,
        "n_retained":      len(retained),
        "n_train":         len(train),
        "n_test":          len(test),
        "retained_ans":    r_ans,
        "retained_unans":  r_unans,
        "coverage_FN_ans": cov_admitted_ans,
        "nm_fallback_ans": nm_fallback_ans,
    }


# ── Main entry point ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precheck",      action="store_true",
                    help="U1 smoke: 20-row grounding-invariant sanity")
    ap.add_argument("--precheck-full", action="store_true",
                    help="U1 full: ≥1000-row grounding percentage on train split (≥95% claim)")
    ap.add_argument("--smoke",         action="store_true",
                    help="U6: 50-item end-to-end pipeline + raw retention report")
    ap.add_argument("--run",           action="store_true",
                    help="U7: full ~2000-item build + train/test split + frozen hash")
    args = ap.parse_args()

    if args.precheck:
        cmd_precheck()
    if getattr(args, "precheck_full", False):
        cmd_precheck_full()
    if args.smoke:
        cmd_smoke()
    if args.run:
        cmd_run()

    if not (args.precheck or getattr(args, "precheck_full", False)
            or args.smoke or args.run):
        print(f"[banner] dataset={HF_DATASET}  seed={SEED}  "
              f"data_dir={DATA_DIR}")
        print("[banner] --precheck (20-row smoke) then --precheck-full (≥1000-row grounding %)")


if __name__ == "__main__":
    main()
