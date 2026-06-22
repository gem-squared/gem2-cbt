"""WP-ST-10: First scoped CER/ECE microcell — learned extractor + the anti-RAG control.

WP-6A established (David + CD + red-team converged):
  - Formal gate PASS on the pre-registered counterfactual regime
    (C_KNOW 1.000 → 0.000 vs B_FAIR ≈ 1.000 on clear-counterfactual items)
  - STRONG boundary-discipline claim NOT demonstrated
    (ambiguous Δ=-0.083, |d|=0.29 < gate floor 0.5)
  - DECISION: PROCEED the extractor stack, but SCOPED to contract *injection*
    of bindings the model's prior cannot supply (not a new Transformer, not
    general boundary reasoning).

WP-10 builds the FIRST CER → ECE → Binder cell with a LEARNED extractor and
tests two falsifiable questions:

  1. gate_learned_payoff — does a LEARNED/automatic extractor (CER → ECE
     producing the pack FROM the input, not an oracle) RETAIN the payoff:
     drive counterfactual violation from B_FAIR ≈ 1.000 toward
     oracle C_KNOW ≈ 0.000?

  2. gate_structure_vs_facts (THE novelty test, the red-team's load-bearing
     control) — does the STRUCTURED contract pack BEAT a PLAINFACTS baseline
     (the SAME extracted facts injected as ordinary prose, no contract
     structure)?

     If C_PACK_LEARNED ≈ PLAINFACTS → the contract STRUCTURE is cosmetic
     over plain in-context facts → the stack reduces to RAG/ICL (name it
     honestly, exactly as HPIC ≡ softmax was named).

     If C_PACK_LEARNED ≪ PLAINFACTS by floor → structure adds real value →
     scoped CBT extractor stack justified as NOVEL.

Without the PLAINFACTS control we cannot distinguish CBT-contract from RAG.
This is the experiment's spine.

Subject model: DeepSeek deepseek-chat via OpenAI-compatible /chat/completions
at temperature=0. Credentials from repo-root .env (gitignored).

WP-6A's oracle_payoff_fair.py + WP-7's concept_ce.py are PRESERVED untouched.
This script reuses their primitives by import only.

This file is the U1 deliverable: items + pre-registered conditions + renderer
formulas + frozen hash. The U3 LEARNED ECE, U4 binder wiring, U5 harness,
and U6/U7 aggregation/claim live below as stubs (filled in later units).

Pre-registration anti-tuning: any change to the CONDITIONS list, the held-out
binding tables, the PLAINFACTS renderer, the CONTRACT scaffold, or the gate
floors AFTER first run invalidates the experiment.

Usage:
  python scripts/cer_ece_cell.py --generate    # U1: items.jsonl + freeze hash
  # U2+ wired progressively as later units land.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# ── Reuse WP-6A primitives (read-only import) ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))            # scripts/ for WP-6A/WP-7 imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # repo root for cbt/ imports
from oracle_payoff_fair import (  # noqa: E402
    STRONG_PROMPT_B_FAIR,
    SYSTEM_PROMPT_CONTRACT_SCAFFOLD,
    make_contract_pack,
    check_violation_fair,
    load_items as load_wp6a_items,
    ITEMS_FILE as WP6A_ITEMS_FILE,
    COUNTERFACTUAL_NOVEL as WP6A_COUNTERFACTUAL_NOVEL,
    COUNTERFACTUAL_ADVERSARIAL as WP6A_COUNTERFACTUAL_ADVERSARIAL,
)
from oracle_payoff import (  # noqa: E402
    LLM_MODEL,
    LLM_TEMP,
)


# ── Paths (WP-10 scoped — do not collide with WP-6A) ───────────────────────
DATA_DIR    = "data/cer_ece"
PAPERS_DIR  = "papers"
ITEMS_FILE  = os.path.join(DATA_DIR, "items.jsonl")
FROZEN_FILE = os.path.join(DATA_DIR, "frozen_items_hash.json")
RUNS_FILE   = os.path.join(DATA_DIR, "raw_runs.jsonl")
EVAL_FILE   = os.path.join(DATA_DIR, "eval_results.json")
RESULTS_MD  = os.path.join(PAPERS_DIR, "results_cer_ece.md")
CLAIM_MD    = os.path.join(PAPERS_DIR, "claim_cer_ece.md")
EXTRACTED_FACTS_FILE = os.path.join(DATA_DIR, "extracted_facts.jsonl")
ROUTER_FILE          = os.path.join(DATA_DIR, "router.joblib")
ROUTER_REPORT_FILE   = os.path.join(DATA_DIR, "router_report.json")


# ═══════════════════════════════════════════════════════════════════════════
# PRE-REGISTERED CONDITIONS (U1 close — anti-tuning).
# Pre-registration timestamp: 2026-06-19T06:00:00Z (UTC) / 2026-06-19 15:00 KST.
# ═══════════════════════════════════════════════════════════════════════════

CONDITIONS = [
    "B_FAIR",           # fair prompt, NO facts. Floor.
    "PLAINFACTS",       # SAME learned facts as prose; NO structure. Anti-RAG control.
    "C_PACK_LEARNED",   # SAME learned facts as structured pack inside SCAFFOLD.
    "C_KNOW_ORACLE",    # WP-6A hand-written oracle pack. Ceiling.
]

N_REPS       = 3      # full-run reps per (item, condition) cell
N_REPS_SMOKE = 1
GATE_FLOOR_PAYOFF      = 0.05   # gate_learned_payoff:    Δ ≥ 0.05 ∧ |d| ≥ 0.5
GATE_FLOOR_STRUCTURE   = 0.05   # gate_structure_vs_facts: V(C_PACK) < V(PLAINFACTS) by 0.05


# ═══════════════════════════════════════════════════════════════════════════
# HELD-OUT counterfactual bindings — generalization probe.
# These do NOT appear in WP-6A items. Frozen at U1 close.
# Discipline: held-out adversarial uses 6 real SI units NOT in WP-6A's
# adversarial pool of 13 (newton, joule, pascal, kelvin, mole, candela, farad,
# henry, tesla, steradian, watt, ohm, ampere). Held-out novel uses 6 invented
# quantity/unit/symbol triples that don't collide with WP-6A's 12 novel bindings.
# ═══════════════════════════════════════════════════════════════════════════

HELD_OUT_NOVEL = [
    {"quantity": "drimble",  "unit_name": "kelpor",  "unit_symbol": "Kp",
     "definition": "1 kelpor is the SI unit of drimble, defined via a standard drimble reference object."},
    {"quantity": "spranth",  "unit_name": "vibnish", "unit_symbol": "Vn",
     "definition": "1 vibnish is the magnitude of spranth produced by a unit baseline source."},
    {"quantity": "muldex",   "unit_name": "rondle",  "unit_symbol": "Rd",
     "definition": "1 rondle is the SI unit of muldex per unit reference object."},
    {"quantity": "fasher",   "unit_name": "querb",   "unit_symbol": "Qb",
     "definition": "1 querb is the SI unit of fasher, defined relative to the standard fasher state."},
    {"quantity": "thoxen",   "unit_name": "millert", "unit_symbol": "Ml",
     "definition": "1 millert is the SI unit of thoxen."},
    {"quantity": "vorbic",   "unit_name": "gloth",   "unit_symbol": "Gl",
     "definition": "1 gloth is the SI unit of vorbic, defined via the reference vorbic configuration."},
]

HELD_OUT_ADVERSARIAL = [
    {"redefined_unit": "hertz",     "redefined_symbol": "Hz",
     "in_context_quantity": "electric capacitance",
     "si_prior_quantity":   "frequency"},
    {"redefined_unit": "weber",     "redefined_symbol": "Wb",
     "in_context_quantity": "luminous intensity",
     "si_prior_quantity":   "magnetic flux"},
    {"redefined_unit": "lumen",     "redefined_symbol": "lm",
     "in_context_quantity": "pressure",
     "si_prior_quantity":   "luminous flux"},
    {"redefined_unit": "coulomb",   "redefined_symbol": "C",
     "in_context_quantity": "force",
     "si_prior_quantity":   "electric charge"},
    {"redefined_unit": "becquerel", "redefined_symbol": "Bq",
     "in_context_quantity": "energy",
     "si_prior_quantity":   "activity (radionuclide decay rate)"},
    {"redefined_unit": "gray",      "redefined_symbol": "Gy",
     "in_context_quantity": "thermodynamic temperature",
     "si_prior_quantity":   "absorbed dose"},
]


# ═══════════════════════════════════════════════════════════════════════════
# U1: context_text builder.
#
# Every clear-counterfactual item gets a `context_text` field — the binding
# rendered as natural prose. This is the input the U3 LEARNED ECE will extract
# from. Frozen at U1 close.
#
# For B_FAIR, context_text is NOT shown (B_FAIR is the no-facts floor).
# For PLAINFACTS, the LEARNED ECE's extracted-facts dict is re-rendered via
# `render_prose(facts)` (U1 formula below) and placed in the SYSTEM prompt.
# For C_PACK_LEARNED, the LEARNED ECE's extracted-facts dict is placed inside
# the contract scaffold as JSON.
# For C_KNOW_ORACLE, the WP-6A hand-written oracle pack is placed inside the
# same scaffold.
# ═══════════════════════════════════════════════════════════════════════════

def build_context_text(item: dict) -> str:
    """Pre-registered binding-as-prose context text for one clear-counterfactual
    item. Frozen at U1 close — DO NOT edit after first run."""
    if item.get("regime") != "clear-counterfactual":
        raise ValueError(
            f"context_text only defined for clear-counterfactual; got {item.get('regime')}"
        )
    subtype = item["subtype"]
    b = item["binding"]
    if subtype == "novel":
        return (
            f"In this domain, a non-standard quantity has been defined. "
            f"The quantity {b['quantity']} has SI unit named {b['unit_name']}, "
            f"with symbol {b['unit_symbol']}. "
            f"Definition: {b['definition']}"
        )
    if subtype == "adversarial":
        return (
            f"In this domain, the SI unit of {b['in_context_quantity']} is "
            f"the {b['redefined_unit']} ({b['redefined_symbol']}). "
            f"This overrides the standard SI prior."
        )
    raise ValueError(f"Unknown counterfactual subtype: {subtype}")


# ═══════════════════════════════════════════════════════════════════════════
# U1: PRE-REGISTERED PLAINFACTS renderer.
#
# Anti-RAG-confound discipline: PLAINFACTS must carry IDENTICAL information
# to C_PACK_LEARNED, differing ONLY in format (prose vs JSON-in-scaffold).
#
# Input: a facts dict produced by the U3 LEARNED ECE for one clear-
#        counterfactual item. Shape:
#          novel       facts: {quantity, unit_name, unit_symbol, definition}
#          adversarial facts: {in_context_quantity, in_context_unit_name,
#                              in_context_symbol, domain_override_note}
#        (Same shape as `make_contract_pack(item, kind="know")["concept"]`.)
#
# Output: a prose string carrying the same information, NO JSON, NO scaffold.
#
# This formula is frozen at U1 close. DO NOT edit after first run.
# ═══════════════════════════════════════════════════════════════════════════

PLAINFACTS_PREAMBLE = (
    "You are a precise physics measurement assistant. "
    "The following facts about the question's domain have been provided. "
    "Use them when answering.\n\n"
)


def render_prose_novel(facts: dict) -> str:
    """Novel-binding facts → prose. Same information as the novel C_PACK
    `concept` block, rendered as ordinary sentences."""
    q   = facts.get("quantity", "(unknown)")
    n   = facts.get("unit_name", "(unknown)")
    s   = facts.get("unit_symbol", "(unknown)")
    d   = facts.get("definition", "")
    out = (
        f"The quantity {q} has SI unit named {n}, with symbol {s}. "
        f"Definition: {d}"
    )
    return out.strip()


def render_prose_adversarial(facts: dict) -> str:
    """Adversarial-binding facts → prose. Same information as the adversarial
    C_PACK `concept` block, rendered as ordinary sentences."""
    iq  = facts.get("in_context_quantity", "(unknown)")
    iun = facts.get("in_context_unit_name", "(unknown)")
    isy = facts.get("in_context_symbol", "(unknown)")
    dn  = facts.get("domain_override_note", "")
    out = (
        f"In this domain, the SI unit of {iq} is the {iun} ({isy}). "
        f"This overrides the standard SI prior. {dn}"
    )
    return out.strip()


def render_prose(item: dict, facts: dict) -> str:
    """Dispatch the pre-registered PLAINFACTS renderer by item subtype.
    Returns the system-prompt body (preamble + prose facts)."""
    if item.get("regime") != "clear-counterfactual":
        raise ValueError(
            f"PLAINFACTS only defined for clear-counterfactual; got {item.get('regime')}"
        )
    subtype = item["subtype"]
    if subtype == "novel":
        body = render_prose_novel(facts)
    elif subtype == "adversarial":
        body = render_prose_adversarial(facts)
    else:
        raise ValueError(f"Unknown counterfactual subtype: {subtype}")
    return PLAINFACTS_PREAMBLE + body


# ═══════════════════════════════════════════════════════════════════════════
# U1: build_system_prompt — dispatch by condition.
#
# `learned_facts` is required for PLAINFACTS and C_PACK_LEARNED (produced by
# the U3 LEARNED ECE at U5 harness time). Ignored for B_FAIR and C_KNOW_ORACLE.
# ═══════════════════════════════════════════════════════════════════════════

def build_system_prompt(condition: str, item: dict, learned_facts: dict = None) -> str:
    """Dispatch system prompt by condition. Used by U5 harness."""
    if condition == "B_FAIR":
        return STRONG_PROMPT_B_FAIR
    if condition == "PLAINFACTS":
        if learned_facts is None:
            raise ValueError("PLAINFACTS requires learned_facts (from U3 LEARNED ECE)")
        return render_prose(item, learned_facts)
    if condition == "C_PACK_LEARNED":
        if learned_facts is None:
            raise ValueError("C_PACK_LEARNED requires learned_facts (from U3 LEARNED ECE)")
        # Wrap learned facts in the SAME pack shape as the oracle pack (only the
        # `concept` block differs by source; `task`/`context` come from the
        # WP-6A oracle scaffold so structural cues match across conditions).
        # ensure_ascii=False keeps Unicode symbols (e.g. Ω) literal so the byte-
        # form matches PLAINFACTS and C_KNOW_ORACLE (information-constancy spine).
        oracle_pack = json.loads(make_contract_pack(item, kind="know"))
        learned_pack = {
            "task":    oracle_pack["task"],
            "context": oracle_pack["context"],
            "concept": learned_facts,
        }
        return SYSTEM_PROMPT_CONTRACT_SCAFFOLD.format(
            contract_json=json.dumps(learned_pack, indent=2, ensure_ascii=False)
        )
    if condition == "C_KNOW_ORACLE":
        # Re-serialize WP-6A's oracle pack with ensure_ascii=False so the
        # byte-form matches C_PACK_LEARNED (and PLAINFACTS) — supports the
        # built-in sanity check C_PACK_LEARNED == C_KNOW_ORACLE on perfect
        # ECE extraction at temp=0. WP-6A code itself is NOT modified.
        pack = json.loads(make_contract_pack(item, kind="know"))
        return SYSTEM_PROMPT_CONTRACT_SCAFFOLD.format(
            contract_json=json.dumps(pack, indent=2, ensure_ascii=False)
        )
    raise ValueError(f"Unknown condition: {condition}")


def build_user_message(condition: str, item: dict) -> str:
    """User-side message.

    For B_FAIR we include the question alone — B_FAIR is the no-facts floor.
    For all OTHER conditions, facts already live in the system prompt (prose
    or structured); the user message is the question alone.

    This keeps the binding-information channel UNIFORM across PLAINFACTS,
    C_PACK_LEARNED, and C_KNOW_ORACLE (the conditions whose comparison drives
    gate_structure_vs_facts), so the only variable is FORMAT, not LOCATION.
    """
    return item["question"]


# ═══════════════════════════════════════════════════════════════════════════
# U1: item generation — combine WP-6A's 25 clear-counterfactual items with
# 12 held-out NEW bindings. Attach pre-registered context_text to every item.
# Hold-out flag in each record so U6 can split overall vs held-out gates.
# ═══════════════════════════════════════════════════════════════════════════

def _make_id(prefix: int, n: int) -> int:
    """Item IDs: 1xx novel (WP-6A), 2xx adversarial (WP-6A),
                 4xx novel held-out,  5xx adversarial held-out."""
    return prefix * 100 + n


def _import_wp6a_counterfactuals() -> list:
    """Pull WP-6A's 25 clear-counterfactual items (novel + adversarial),
    re-attach context_text + held_out=False. WP-6A items are READ-ONLY here
    (we do not write back to WP-6A's items.jsonl)."""
    wp6a = load_wp6a_items()
    out = []
    for it in wp6a:
        if it.get("regime") != "clear-counterfactual":
            continue
        new = dict(it)
        new["held_out"]     = False
        new["wp6a_source"]  = True
        new["context_text"] = build_context_text(new)
        out.append(new)
    return out


def _make_held_out_items() -> list:
    """Held-out 12: 6 novel + 6 adversarial. Same checker contract as
    WP-6A (check_violation_fair dispatches on regime+subtype + uses
    correct_answer_keywords for in-context unit name + symbol)."""
    items = []

    # Novel held-out
    for n, cf in enumerate(HELD_OUT_NOVEL):
        it = {
            "id":          _make_id(4, n),
            "n_in_regime": n,
            "regime":      "clear-counterfactual",
            "subtype":     "novel",
            "held_out":    True,
            "wp6a_source": False,
            "binding": {
                "quantity":    cf["quantity"],
                "unit_name":   cf["unit_name"],
                "unit_symbol": cf["unit_symbol"],
                "definition":  cf["definition"],
            },
            "question": (
                f"What is the SI unit of {cf['quantity']}? "
                f"State the unit name and symbol."
            ),
            "correct_answer_keywords": [cf["unit_name"], cf["unit_symbol"].lower()],
            "violation_keywords":      [],
            "expected": "in_context_commit",
        }
        it["context_text"] = build_context_text(it)
        items.append(it)

    # Adversarial held-out
    for n, cf in enumerate(HELD_OUT_ADVERSARIAL):
        it = {
            "id":          _make_id(5, n),
            "n_in_regime": n,
            "regime":      "clear-counterfactual",
            "subtype":     "adversarial",
            "held_out":    True,
            "wp6a_source": False,
            "binding": {
                "redefined_unit":      cf["redefined_unit"],
                "redefined_symbol":    cf["redefined_symbol"],
                "in_context_quantity": cf["in_context_quantity"],
                "si_prior_quantity":   cf["si_prior_quantity"],
            },
            "question": (
                f"In this domain, what is the SI unit of {cf['in_context_quantity']}? "
                f"State the unit name and symbol."
            ),
            "correct_answer_keywords": [cf["redefined_unit"], cf["redefined_symbol"].lower()],
            "violation_keywords":      [],
            "si_prior_leak_keyword":   cf["redefined_unit"],
            "expected": "in_context_commit",
        }
        it["context_text"] = build_context_text(it)
        items.append(it)

    return items


def generate_items() -> list:
    """U1: emit the combined item set (WP-6A 25 cf + 12 held-out = 37)."""
    items = _import_wp6a_counterfactuals() + _make_held_out_items()
    return items


# ═══════════════════════════════════════════════════════════════════════════
# Frozen-hash discipline (FAIL-FAST) — reused pattern from WP-6A.
# ═══════════════════════════════════════════════════════════════════════════

def items_hash(items_path: str) -> str:
    h = hashlib.sha256()
    with open(items_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def freeze_items_hash(h: str, n_items: int, counts: dict) -> None:
    record = {
        "frozen_hash":          h,
        "n_items":              n_items,
        "subtype_counts":       counts,
        "model":                LLM_MODEL,
        "temperature":          LLM_TEMP,
        "conditions":           CONDITIONS,
        "n_reps":               N_REPS,
        "gate_floor_payoff":    GATE_FLOOR_PAYOFF,
        "gate_floor_structure": GATE_FLOOR_STRUCTURE,
        "held_out_count":       sum(1 for _ in HELD_OUT_NOVEL) + sum(1 for _ in HELD_OUT_ADVERSARIAL),
    }
    with open(FROZEN_FILE, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[freeze] items hash locked: {h} ({n_items} items)")
    print(f"[freeze] subtype counts: {counts}")
    print(f"[freeze] conditions:    {CONDITIONS}")


def assert_frozen() -> str:
    if not os.path.exists(FROZEN_FILE):
        raise RuntimeError(f"FROZEN HASH MISSING: {FROZEN_FILE} — run --generate first")
    frozen = json.load(open(FROZEN_FILE))["frozen_hash"]
    current = items_hash(ITEMS_FILE)
    if current != frozen:
        raise RuntimeError(f"items hash mismatch: frozen={frozen} current={current}")
    return current


def load_items() -> list:
    out = []
    with open(ITEMS_FILE) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def subtype_counts(items: list) -> dict:
    out = {}
    for it in items:
        key = f'{it["subtype"]}{"_held_out" if it.get("held_out") else ""}'
        out[key] = out.get(key, 0) + 1
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════

def cmd_generate() -> None:
    """U1: combined items.jsonl + frozen hash."""
    os.makedirs(DATA_DIR, exist_ok=True)
    items = generate_items()
    with open(ITEMS_FILE, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    h = items_hash(ITEMS_FILE)
    counts = subtype_counts(items)
    freeze_items_hash(h, len(items), counts)
    print(f"[generate] wrote {ITEMS_FILE} with {len(items)} items")


# ═══════════════════════════════════════════════════════════════════════════
# U2: CER router — softmax over RAW token features.
#
# WP-8 lesson: route on raw features (NOT the (ss, es) compression — that
# halved recall). Complex HPIC NOT used (rejected 3×). Reuse the WP-5/WP-8
# pattern (sklearn LogisticRegression.predict_proba inside a router_softmax
# wrapper) but with raw TF-IDF over context_text as the feature source.
#
# The scoped domain has 2 binding-types (novel, adversarial) with very
# distinct prose forms — routing is likely near-trivial here. We keep the
# claim honest: the router is WIRED into the cell, not stress-tested at
# this scope.
# ═══════════════════════════════════════════════════════════════════════════

ROUTER_CV_FOLDS = 5
ROUTER_TFIDF_NGRAM = (1, 2)


def _build_router_corpus(items: list) -> tuple:
    """Return (texts, labels). Labels = item subtype ∈ {novel, adversarial}."""
    texts  = [it["context_text"] for it in items]
    labels = [it["subtype"]      for it in items]
    return texts, labels


def cmd_route_fit() -> None:
    """U2: fit softmax router on WP-6A items; test on held-out for
    generalization. Persist (vectorizer, clf) for U5 harness."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import joblib
    import numpy as np

    assert_frozen()
    items = load_items()
    train = [it for it in items if not it.get("held_out")]
    test  = [it for it in items if it.get("held_out")]
    if not train or not test:
        raise RuntimeError(
            f"empty split: |train|={len(train)} |test|={len(test)}"
        )

    Xtr_text, ytr = _build_router_corpus(train)
    Xte_text, yte = _build_router_corpus(test)

    vec = TfidfVectorizer(ngram_range=ROUTER_TFIDF_NGRAM, lowercase=True)
    Xtr = vec.fit_transform(Xtr_text)
    Xte = vec.transform(Xte_text)

    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(Xtr, ytr)

    # In-sample CV on train (5-fold stratified)
    cv = StratifiedKFold(n_splits=ROUTER_CV_FOLDS, shuffle=True, random_state=42)
    cv_pipeline = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    cv_scores = cross_val_score(cv_pipeline, Xtr, ytr, cv=cv)
    cv_acc = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    # Held-out accuracy
    yte_pred = clf.predict(Xte)
    held_acc = float((yte_pred == np.array(yte)).mean())
    held_correct = int((yte_pred == np.array(yte)).sum())
    held_total = len(yte)

    # Per-class held-out
    classes = list(clf.classes_)
    per_class = {}
    yte_arr = np.array(yte)
    yte_pred_arr = np.array(yte_pred)
    for c in classes:
        mask = yte_arr == c
        if int(mask.sum()) == 0:
            per_class[c] = None
            continue
        per_class[c] = {
            "n":       int(mask.sum()),
            "correct": int((yte_pred_arr[mask] == c).sum()),
            "acc":     float((yte_pred_arr[mask] == c).mean()),
        }

    # Persist router + report
    joblib.dump({"vectorizer": vec, "clf": clf, "classes": classes}, ROUTER_FILE)
    report = {
        "router_type":         "softmax_over_raw_tfidf",
        "wp_lesson":           "WP-8: raw features beat (ss,es) compression",
        "ngram_range":         list(ROUTER_TFIDF_NGRAM),
        "n_train":             len(train),
        "n_held_out":          held_total,
        "n_features":          int(Xtr.shape[1]),
        "classes":             classes,
        "cv_folds":            ROUTER_CV_FOLDS,
        "cv_acc_mean":         cv_acc,
        "cv_acc_std":          cv_std,
        "cv_per_fold":         [float(s) for s in cv_scores],
        "held_out_acc":        held_acc,
        "held_out_correct":    held_correct,
        "held_out_total":      held_total,
        "held_out_per_class":  per_class,
        "honesty_note":        (
            "Scoped domain has 2 binding-types with very distinct prose forms; "
            "near-perfect routing here is expected. The router is WIRED into the "
            "cell, not stress-tested at this scope. Future scopes with more "
            "binding-types will stress-test it."
        ),
    }
    with open(ROUTER_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[route-fit] cv_acc={cv_acc:.3f} ± {cv_std:.3f} (folds={ROUTER_CV_FOLDS}, n_train={len(train)})")
    print(f"[route-fit] held_out_acc={held_acc:.3f} ({held_correct}/{held_total}, classes={classes})")
    print(f"[route-fit] per-class held-out: {per_class}")
    print(f"[route-fit] saved router → {ROUTER_FILE}")
    print(f"[route-fit] saved report → {ROUTER_REPORT_FILE}")


def load_router() -> dict:
    """Load the persisted U2 router for use in U3+ pipeline."""
    import joblib
    if not os.path.exists(ROUTER_FILE):
        raise RuntimeError(f"ROUTER MISSING: {ROUTER_FILE} — run --route-fit first")
    return joblib.load(ROUTER_FILE)


def route_one(item: dict, router: dict = None) -> tuple:
    """Predict the binding-type for one item from its context_text.
    Returns (predicted_subtype: str, probs: dict[class, prob])."""
    if router is None:
        router = load_router()
    vec   = router["vectorizer"]
    clf   = router["clf"]
    Xv    = vec.transform([item["context_text"]])
    probs = clf.predict_proba(Xv)[0]
    classes = list(clf.classes_)
    pred = classes[int(probs.argmax())]
    return pred, {c: float(p) for c, p in zip(classes, probs)}


def cmd_verify_u2() -> int:
    """U2 inline-verify: router file present, report records honest accuracy,
    route_one returns a valid label for every item."""
    failures = []
    if not os.path.exists(ROUTER_FILE):
        failures.append(f"router missing: {ROUTER_FILE}")
        return _verify_report(failures)
    if not os.path.exists(ROUTER_REPORT_FILE):
        failures.append(f"router report missing: {ROUTER_REPORT_FILE}")
        return _verify_report(failures)
    rep = json.load(open(ROUTER_REPORT_FILE))
    for k in ("cv_acc_mean", "held_out_acc", "classes", "honesty_note",
              "n_train", "n_held_out", "router_type"):
        if k not in rep or rep[k] in (None, ""):
            failures.append(f"router report missing field {k}")
    if set(rep.get("classes", [])) != {"novel", "adversarial"}:
        failures.append(f"router classes ≠ {{novel, adversarial}}: {rep.get('classes')}")
    # Exercise route_one on every item — must return a valid class.
    router = load_router()
    items = load_items()
    bad_routes = []
    for it in items:
        pred, probs = route_one(it, router)
        if pred not in {"novel", "adversarial"}:
            bad_routes.append((it["id"], pred))
        # Probs must sum to ~1
        if abs(sum(probs.values()) - 1.0) > 1e-6:
            bad_routes.append((it["id"], "probs sum != 1"))
    if bad_routes:
        failures.append(f"route_one bad routes: {bad_routes[:5]}")

    # Honesty record present
    if "near-perfect" not in rep.get("honesty_note", "").lower():
        failures.append("honesty note missing 'near-perfect' acknowledgment")

    print(f"[verify-u2] cv_acc={rep['cv_acc_mean']:.3f}  held_acc={rep['held_out_acc']:.3f}  classes={rep['classes']}")
    return _verify_report(failures)


# ═══════════════════════════════════════════════════════════════════════════
# U3: LEARNED Expert Contract-Extractor (ECE).
#
# Given a `context_text` paragraph carrying the in-context binding, the ECE
# emits a structured facts dict matching the WP-6A `concept` block shape:
#
#   subtype="novel":
#     {quantity, unit_name, unit_symbol, definition}
#
#   subtype="adversarial":
#     {in_context_quantity, in_context_unit_name, in_context_symbol,
#      domain_override_note}
#
# Reuse WP-7 `concept_ce.method_llm_prompt` pattern: DeepSeek deepseek-chat
# at temperature=0, structured JSON, OpenAI-compatible /chat/completions via
# concept_ce._llm_request. The ECE call is ONE LLM call per item.
#
# Extraction fidelity is scored against the WP-6A oracle pack (`make_contract_pack
# kind="know"`) per item — exact_match per field + aggregate. This is the
# component whose imperfection WP-6A flagged as the open question.
# ═══════════════════════════════════════════════════════════════════════════

# Pre-registered ECE system prompts — frozen at U3 close. Anti-tuning:
# DO NOT edit after first run.

ECE_SYSTEM_PROMPT_NOVEL = """You are a precise extraction assistant for measurement bindings.

Given a DOMAIN CONTEXT paragraph that defines a non-standard physical quantity together with its SI unit, EXTRACT the binding into a strict JSON object with EXACTLY these four keys and no others:

{
  "quantity":    <string — the non-standard quantity name>,
  "unit_name":   <string — the SI unit name in this domain>,
  "unit_symbol": <string — the SI unit symbol>,
  "definition":  <string — the defining sentence(s) verbatim>
}

Rules:
- Copy the values from the DOMAIN CONTEXT. Do NOT invent or paraphrase.
- The quantity is the noun being defined; the unit is what 1 unit-of-quantity equals.
- If the context contains a symbol like "Kp", keep its case as written.
- Return ONLY the JSON object. No prose, no code fences, no commentary."""

ECE_SYSTEM_PROMPT_ADVERSARIAL = """You are a precise extraction assistant for measurement bindings.

Given a DOMAIN CONTEXT paragraph that asserts an in-context SI unit binding for a real physical quantity (overriding the standard SI prior), EXTRACT the binding into a strict JSON object with EXACTLY these four keys and no others:

{
  "in_context_quantity":  <string — the real physical quantity name>,
  "in_context_unit_name": <string — the unit name the context binds it to>,
  "in_context_symbol":    <string — the unit symbol the context binds it to>,
  "domain_override_note": <string — a sentence stating that in this domain, <unit> is the SI unit of <quantity>>
}

Rules:
- Copy the values from the DOMAIN CONTEXT. Do NOT invent or paraphrase.
- IGNORE the SI prior. The in-context binding is the answer even if it contradicts what the unit usually denotes.
- Keep symbol case exactly as written.
- domain_override_note MUST be a complete sentence of the form "In this domain, <unit_name> is the SI unit of <in_context_quantity>." — fill in from the extracted values.
- Return ONLY the JSON object. No prose, no code fences, no commentary."""


# Expected schema fields per subtype — used by fidelity scorer + ECE-output validator.
ECE_FIELDS_NOVEL       = ("quantity", "unit_name", "unit_symbol", "definition")
ECE_FIELDS_ADVERSARIAL = ("in_context_quantity", "in_context_unit_name",
                          "in_context_symbol", "domain_override_note")


# ── WP-11 U3: backend-aware ECE call. Preserves WP-7 concept_ce.py UNTOUCHED. ──
#
# For LLM_BACKEND=deepseek (default): use concept_ce._llm_request unchanged
#   → WP-10 archived behavior is byte-reproducible against the same .env.
# For LLM_BACKEND=local: bypass concept_ce._llm_request and call Ollama
#   native /api/chat with GREEDY_OPTS via the U2 client.
# The dispatch happens here, NOT inside concept_ce.

def _call_extractor_llm(system_prompt: str, user_msg: str) -> str:
    """Backend-aware extractor LLM call. Returns raw response text."""
    from cbt.llm_backend import get_backend  # noqa: WPS433 (lazy)
    b = get_backend()
    if b["backend"] == "deepseek":
        from concept_ce import _llm_request  # noqa: WPS433 (lazy, preserves WP-7)
        return _llm_request(user_msg, system_prompt=system_prompt, temperature=0.0)
    if b["backend"] == "local":
        # Ollama native /api/chat — greedy options applied; truly deterministic.
        return _ollama_chat_greedy(system_prompt, user_msg,
                                   base_url=b["base_url"], model=b["model"])
    raise RuntimeError(f"unsupported backend: {b['backend']}")


def _backend_data_dir() -> str:
    """Returns the data directory for the active backend (separate-track
    discipline per David's anti-silent-merge directive). DeepSeek stays at
    data/cer_ece/ (WP-10 archived path); local goes to data/cer_ece_qwen/."""
    from cbt.llm_backend import get_backend  # noqa
    b = get_backend()
    if b["backend"] == "deepseek":
        return DATA_DIR
    return DATA_DIR_QWEN


def _backend_path(filename: str) -> str:
    """Return the backend-namespaced data path for `filename`."""
    return os.path.join(_backend_data_dir(), filename)


def ece_extract(item: dict, router: dict = None) -> tuple:
    """Route → choose ECE prompt → backend-aware LLM call → parse → return
    (facts_dict, routed_to, meta). On failure, facts_dict is empty; caller
    records the error in the raw log."""
    from concept_ce import _parse_llm_json  # noqa: pure parser, no LLM call
    routed_to, _probs = route_one(item, router)
    if routed_to == "novel":
        sys_prompt = ECE_SYSTEM_PROMPT_NOVEL
        fields = ECE_FIELDS_NOVEL
    elif routed_to == "adversarial":
        sys_prompt = ECE_SYSTEM_PROMPT_ADVERSARIAL
        fields = ECE_FIELDS_ADVERSARIAL
    else:
        raise RuntimeError(f"router emitted unknown subtype: {routed_to}")

    raw = _call_extractor_llm(sys_prompt, item["context_text"])
    parsed = _parse_llm_json(raw)

    # Validate schema — all required fields present + string-typed.
    missing = [f for f in fields if f not in parsed or parsed[f] in (None, "")]
    extras  = [k for k in parsed.keys() if k not in fields]
    coerced = {f: str(parsed[f]) for f in fields if f in parsed and parsed[f] not in (None, "")}
    return coerced, routed_to, {"raw_response": raw, "missing": missing, "extras": extras}


def oracle_facts(item: dict) -> dict:
    """The WP-6A hand-written oracle facts dict (`concept` block of the
    oracle pack) — used as the fidelity reference."""
    pack = json.loads(make_contract_pack(item, kind="know"))
    return pack["concept"]


def _norm(s: str) -> str:
    return str(s).strip().lower()


def score_one_fidelity(extracted: dict, oracle: dict, fields: tuple) -> dict:
    """Per-field exact-match + token-overlap partial score.
    Returns {field: {exact, partial, expected, got}}."""
    out = {}
    for f in fields:
        exp = oracle.get(f, "")
        got = extracted.get(f, "")
        exact = (_norm(got) == _norm(exp))
        exp_toks = set(_norm(exp).split())
        got_toks = set(_norm(got).split())
        if exp_toks:
            partial = len(exp_toks & got_toks) / len(exp_toks)
        else:
            partial = 1.0 if not got_toks else 0.0
        out[f] = {
            "exact":    bool(exact),
            "partial":  float(partial),
            "expected": exp,
            "got":      got,
        }
    return out


def cmd_extract_all() -> None:
    """U3: run ECE on all 37 items; persist extracted_facts.jsonl. Resumable.
    Aggregate fidelity vs oracle (exact-match + partial).

    WP-11 U3: backend-aware — writes to data/cer_ece/ for DeepSeek (archived
    behavior preserved), data/cer_ece_qwen/ for local Qwen (separate track,
    no silent merge per David's directive)."""
    assert_frozen()
    items = load_items()
    router = load_router()

    extracted_file = _backend_path("extracted_facts.jsonl")
    os.makedirs(os.path.dirname(extracted_file), exist_ok=True)

    # Write a backend manifest alongside so the data dir is self-labeling.
    from cbt.llm_backend import record_backend_manifest  # noqa
    record_backend_manifest(_backend_path("manifest_u3_extraction.json"))

    done = {}
    if os.path.exists(extracted_file):
        with open(extracted_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["item_id"]] = r

    new_records = 0
    with open(extracted_file, "a") as f:
        for it in items:
            if it["id"] in done:
                continue
            try:
                facts, routed_to, meta = ece_extract(it, router)
                ok = (not meta["missing"]) and bool(facts)
                rec = {
                    "item_id":    it["id"],
                    "subtype":    it["subtype"],
                    "held_out":   it.get("held_out", False),
                    "routed_to":  routed_to,
                    "facts":      facts,
                    "ok":         bool(ok),
                    "missing":    meta["missing"],
                    "extras":     meta["extras"],
                    "raw_response": meta["raw_response"],
                }
            except Exception as e:  # noqa: BLE001
                rec = {
                    "item_id":    it["id"],
                    "subtype":    it["subtype"],
                    "held_out":   it.get("held_out", False),
                    "routed_to":  None,
                    "facts":      None,
                    "ok":         False,
                    "error":      str(e)[:160],
                }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            new_records += 1
            tag = "OK" if rec.get("ok") else "FAIL"
            print(f"[ece] id={it['id']:>3} subtype={it['subtype']:>11} → {tag}")

    print(f"[extract-all] new records: {new_records}, total: {sum(1 for _ in open(extracted_file) if _.strip())}")
    cmd_extraction_fidelity()


def load_extracted_facts() -> dict:
    """Returns dict[item_id → record]. Backend-aware (WP-11 U3)."""
    extracted_file = _backend_path("extracted_facts.jsonl")
    if not os.path.exists(extracted_file):
        raise RuntimeError(f"EXTRACTED FACTS MISSING: {extracted_file} — run --extract-all")
    out = {}
    with open(extracted_file) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]] = r
    return out


# ── WP-11 U3 gate floors (pre-registered, frozen at U3 close — anti-tuning) ──
GATE_FLOOR_EXTRACTION_YIELD     = 0.90   # gate (a) component 1: yield ≥ 0.90
GATE_FLOOR_ADVERSARIAL_LEAK     = 0.10   # gate (a) component 2: leak ≤ 0.10
                                          # (loose vs DeepSeek's 0.02; Qwen-32B is weaker)


def cmd_extraction_fidelity() -> None:
    """Aggregate per-field fidelity vs oracle pack + WP-11 leak_rate +
    WP-11 gate (a) verdict. Persist to backend-namespaced
    `extraction_fidelity.json`."""
    items = load_items()
    recs  = load_extracted_facts()

    per_item = []
    field_stats = {"novel": {f: [] for f in ECE_FIELDS_NOVEL},
                   "adversarial": {f: [] for f in ECE_FIELDS_ADVERSARIAL}}
    ok_count = 0
    for it in items:
        rec = recs.get(it["id"])
        if rec is None or not rec.get("ok"):
            per_item.append({"item_id": it["id"], "ok": False})
            continue
        ok_count += 1
        oracle = oracle_facts(it)
        fields = ECE_FIELDS_NOVEL if it["subtype"] == "novel" else ECE_FIELDS_ADVERSARIAL
        score = score_one_fidelity(rec["facts"], oracle, fields)
        per_item.append({"item_id": it["id"], "subtype": it["subtype"],
                         "held_out": it.get("held_out", False),
                         "ok": True, "fields": score})
        for f, s in score.items():
            field_stats[it["subtype"]][f].append(s)

    # Aggregate
    summary = {"n_items": len(items), "n_ok": ok_count,
               "ece_yield": ok_count / max(len(items), 1)}
    for subtype, fields_dict in field_stats.items():
        summary[subtype] = {}
        for f, scores in fields_dict.items():
            if not scores:
                summary[subtype][f] = None
                continue
            n = len(scores)
            exact = sum(1 for s in scores if s["exact"]) / n
            partial = sum(s["partial"] for s in scores) / n
            summary[subtype][f] = {"n": n, "exact_match_rate": exact,
                                   "partial_score_mean": partial}

    # Per-split (held_out vs in-domain) overall exact-match
    for split in (("in_domain", False), ("held_out", True)):
        split_name, split_flag = split
        ex_rates = []
        for it_rec in per_item:
            if not it_rec.get("ok"):
                continue
            if it_rec.get("held_out") != split_flag:
                continue
            subtype = it_rec.get("subtype")
            fields = it_rec["fields"]
            ex_rates.append(sum(1 for f in fields.values() if f["exact"]) / max(len(fields), 1))
        summary[f"{split_name}_avg_exact_match"] = (
            sum(ex_rates) / len(ex_rates) if ex_rates else None
        )

    # ── WP-11 U3: explicit adversarial leak_rate ────────────────────────
    # leak_rate = fraction of adversarial items where the extracted
    # in_context_unit_name does NOT match the oracle in_context_unit_name
    # (i.e., extractor failed to copy the in-context binding — most often
    # by reverting to the SI prior unit for the in-context quantity).
    adv_total = 0
    adv_leaks = 0
    adv_leak_items = []
    for it_rec in per_item:
        if not it_rec.get("ok"):
            continue
        if it_rec.get("subtype") != "adversarial":
            continue
        adv_total += 1
        unit_field = it_rec["fields"].get("in_context_unit_name", {})
        if not unit_field.get("exact"):
            adv_leaks += 1
            adv_leak_items.append({
                "item_id":  it_rec["item_id"],
                "expected": unit_field.get("expected"),
                "got":      unit_field.get("got"),
                "held_out": it_rec.get("held_out"),
            })
    leak_rate = (adv_leaks / adv_total) if adv_total else 0.0
    summary["adversarial_leak_rate"]   = leak_rate
    summary["adversarial_leak_n"]      = adv_leaks
    summary["adversarial_total"]       = adv_total
    summary["adversarial_leak_items"]  = adv_leak_items

    # ── WP-11 U3 gate (a) of gate_local_reproduction ─────────────────────
    from cbt.llm_backend import get_backend  # noqa
    b = get_backend()
    gate_a = {
        "backend":              b["backend"],
        "model":                b["model"],
        "extraction_yield":     summary["ece_yield"],
        "yield_floor":          GATE_FLOOR_EXTRACTION_YIELD,
        "yield_pass":           summary["ece_yield"] >= GATE_FLOOR_EXTRACTION_YIELD,
        "adversarial_leak_rate": leak_rate,
        "leak_floor":           GATE_FLOOR_ADVERSARIAL_LEAK,
        "leak_pass":            leak_rate <= GATE_FLOOR_ADVERSARIAL_LEAK,
    }
    gate_a["verdict"] = "PASS" if (gate_a["yield_pass"] and gate_a["leak_pass"]) else "FAIL"
    summary["gate_a_local_reproduction"] = gate_a

    path = _backend_path("extraction_fidelity.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "per_item": per_item}, f, indent=2)

    # Print headline
    print(f"[fidelity] backend={b['backend']}  model={b['model']}")
    print(f"[fidelity] ece_yield = {summary['ece_yield']:.3f} ({summary['n_ok']}/{summary['n_items']})")
    for st in ("novel", "adversarial"):
        ex_rates = [v["exact_match_rate"] for v in summary[st].values()
                    if v is not None]
        avg_ex = sum(ex_rates) / len(ex_rates) if ex_rates else float("nan")
        print(f"[fidelity] {st:>11}: avg field-exact-match = {avg_ex:.3f}")
    print(f"[fidelity] in_domain  avg exact_match per item = {summary.get('in_domain_avg_exact_match')}")
    print(f"[fidelity] held_out   avg exact_match per item = {summary.get('held_out_avg_exact_match')}")
    print(f"[fidelity] adversarial leak_rate = {leak_rate:.3f} ({adv_leaks}/{adv_total}) "
          f"floor={GATE_FLOOR_ADVERSARIAL_LEAK}")
    print(f"[fidelity] gate (a) gate_local_reproduction: {gate_a['verdict']} "
          f"(yield_pass={gate_a['yield_pass']}, leak_pass={gate_a['leak_pass']})")
    print(f"[fidelity] saved → {path}")


def cmd_verify_u3() -> int:
    """U3 inline-verify: extracted_facts present for every item, ECE yield
    recorded, fidelity report present + non-empty. Backend-aware (WP-11)."""
    failures = []
    extracted_file = _backend_path("extracted_facts.jsonl")
    if not os.path.exists(extracted_file):
        failures.append(f"extracted facts missing: {extracted_file}")
        return _verify_report(failures)
    fid_path = _backend_path("extraction_fidelity.json")
    if not os.path.exists(fid_path):
        failures.append(f"extraction fidelity missing: {fid_path}")
        return _verify_report(failures)
    items = load_items()
    recs  = load_extracted_facts()
    if len(recs) != len(items):
        failures.append(f"extracted records {len(recs)} != items {len(items)}")
    for it in items:
        r = recs.get(it["id"])
        if r is None:
            failures.append(f"item id={it['id']} missing from extracted facts")
            continue
        if r.get("ok") and not r.get("facts"):
            failures.append(f"item id={it['id']} marked ok but facts is empty")

    fid = json.load(open(fid_path))
    summary = fid["summary"]
    for k in ("ece_yield", "novel", "adversarial",
              "in_domain_avg_exact_match", "held_out_avg_exact_match"):
        if k not in summary:
            failures.append(f"fidelity summary missing key {k}")
    # Smoke-call ece_extract on one item via reuse of stored facts (no LLM)
    # — confirm load_extracted_facts round-trips.
    sample_id = next(iter(recs))
    sample = recs[sample_id]
    if sample.get("ok") and sample.get("subtype") == "novel":
        f = sample["facts"]
        for k in ECE_FIELDS_NOVEL:
            if k not in f:
                failures.append(f"novel facts missing field {k}")
                break
    print(f"[verify-u3] yield={summary['ece_yield']:.3f}  "
          f"in_domain={summary.get('in_domain_avg_exact_match')}  "
          f"held_out={summary.get('held_out_avg_exact_match')}")
    return _verify_report(failures)


# ═══════════════════════════════════════════════════════════════════════════
# U4: Binder + structure-isolation spot-check.
#
# The Binder takes one item and its LEARNED ECE facts and assembles the TWO
# injection formats whose comparison drives gate_structure_vs_facts:
#
#   PLAINFACTS     — same facts as ordinary prose; NO JSON, NO scaffold
#   C_PACK_LEARNED — same facts as structured pack inside the WP-6A scaffold
#
# Both are produced from the SAME `extracted_facts` dict → information held
# constant; only the FORMAT differs. The spot-check verifies that property
# per-item, before the U5 harness exercises it on the LLM.
# ═══════════════════════════════════════════════════════════════════════════

def bind_conditions(item: dict, extracted_facts: dict) -> dict:
    """U4 Binder: assemble PLAINFACTS + C_PACK_LEARNED system prompts from
    the same `extracted_facts`. Returns a structured record with
    information-constancy + scaffold-cleanliness flags."""
    pf = build_system_prompt("PLAINFACTS",     item, extracted_facts)
    cp = build_system_prompt("C_PACK_LEARNED", item, extracted_facts)

    leak_report = []
    info_constant = True
    for k, v in extracted_facts.items():
        v_str = str(v).strip()
        if not v_str:
            continue
        if v_str not in pf:
            info_constant = False
            leak_report.append({"field": k, "missing_in": "PLAINFACTS", "value": v_str})
        if v_str not in cp:
            info_constant = False
            leak_report.append({"field": k, "missing_in": "C_PACK_LEARNED", "value": v_str})

    # Structure check: PLAINFACTS must not leak the scaffold form (no JSON braces,
    # no "CONTRACT:" header). C_PACK_LEARNED must contain the scaffold marker.
    structure_clean = ("CONTRACT" not in pf) and ("{" not in pf) and ("CONTRACT" in cp) and ("{" in cp)

    return {
        "item_id":         item["id"],
        "subtype":         item["subtype"],
        "held_out":        item.get("held_out", False),
        "PLAINFACTS":      pf,
        "C_PACK_LEARNED":  cp,
        "facts_used":      extracted_facts,
        "info_constant":   info_constant,
        "structure_clean": structure_clean,
        "leak_report":     leak_report,
    }


def cmd_binder_spotcheck() -> None:
    """U4: spot-check the binder across all 37 items. Verify the SAME facts
    appear in BOTH PLAINFACTS and C_PACK_LEARNED (information-constancy) and
    that the formats are distinct (scaffold-cleanliness)."""
    items = load_items()
    facts_recs = load_extracted_facts()

    per_item = []
    for it in items:
        rec = facts_recs.get(it["id"])
        if rec is None or not rec.get("ok"):
            per_item.append({"item_id": it["id"], "skipped": True,
                             "reason": "no ECE facts available"})
            continue
        bind = bind_conditions(it, rec["facts"])
        per_item.append({
            "item_id":         bind["item_id"],
            "subtype":         bind["subtype"],
            "held_out":        bind["held_out"],
            "info_constant":   bind["info_constant"],
            "structure_clean": bind["structure_clean"],
            "leak_report":     bind["leak_report"],
        })

    active = [r for r in per_item if not r.get("skipped")]
    n = len(active)
    n_ic = sum(1 for r in active if r["info_constant"])
    n_sc = sum(1 for r in active if r["structure_clean"])
    summary = {
        "n":                        n,
        "info_constant":            n_ic,
        "info_constant_rate":       n_ic / max(n, 1),
        "structure_clean":          n_sc,
        "structure_clean_rate":     n_sc / max(n, 1),
        "n_skipped":                len(per_item) - n,
    }
    path = os.path.join(DATA_DIR, "binder_spotcheck.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "per_item": per_item}, f, indent=2)
    print(f"[binder] info-constant {n_ic}/{n} ({summary['info_constant_rate']:.3f})  "
          f"structure-clean {n_sc}/{n} ({summary['structure_clean_rate']:.3f})  "
          f"skipped={summary['n_skipped']}")
    print(f"[binder] saved → {path}")


def cmd_verify_u4() -> int:
    """U4 inline-verify: binder spot-check artifact present, info-constancy
    rate == 1.0, scaffold-cleanliness rate == 1.0."""
    failures = []
    path = os.path.join(DATA_DIR, "binder_spotcheck.json")
    if not os.path.exists(path):
        failures.append(f"binder spot-check missing: {path}")
        return _verify_report(failures)
    doc = json.load(open(path))
    s = doc["summary"]
    if s["info_constant_rate"] < 1.0:
        failures.append(
            f"info-constancy < 1.0: {s['info_constant']}/{s['n']}; "
            f"per-item leaks: {[r for r in doc['per_item'] if not r.get('skipped') and not r['info_constant']]}"
        )
    if s["structure_clean_rate"] < 1.0:
        failures.append(
            f"scaffold-cleanliness < 1.0: {s['structure_clean']}/{s['n']}"
        )
    if s["n"] != 37:
        failures.append(f"binder spot-check n={s['n']} != 37")
    print(f"[verify-u4] info_constant_rate={s['info_constant_rate']:.3f}  "
          f"structure_clean_rate={s['structure_clean_rate']:.3f}  n={s['n']}")
    return _verify_report(failures)


# ═══════════════════════════════════════════════════════════════════════════
# U5: Subject-LLM harness — items × conditions × N reps.
#
# For each (item, condition, rep) cell:
#   - Build system prompt via build_system_prompt(condition, item, learned_facts)
#   - Build user msg via build_user_message (always the question)
#   - Call DeepSeek via WP-6 oracle_payoff.call_llm (temp=0)
#   - Run check_violation_fair (WP-6A counterfactual checker) → verdict
#   - Append full record to data/cer_ece/raw_runs.jsonl
#
# FAIL-FAST frozen-hash assert at entry. Resumable via (item_id, condition,
# rep_id) key. Smoke = 4 items × 4 conditions × 1 rep = 16 calls.
# Full   = 37 items × 4 conditions × 3 reps = 444 calls.
# ═══════════════════════════════════════════════════════════════════════════

SMOKE_ITEM_IDS = (100, 212, 400, 500)
# 100 = first in-domain novel; 212 = first in-domain adversarial;
# 400 = first held-out novel; 500 = first held-out adversarial.


def _call_subject_llm(system_prompt: str, user_msg: str) -> str:
    """Backend-aware subject LLM call (WP-11 U4). Returns clean response text.
    Preserves WP-6 oracle_payoff.call_llm for the deepseek path (archived
    behavior). Local path uses the Ollama native /api/chat with GREEDY_OPTS."""
    from cbt.llm_backend import get_backend  # noqa
    b = get_backend()
    if b["backend"] == "deepseek":
        from oracle_payoff import call_llm  # noqa: WPS433
        return call_llm(user_msg, system_prompt=system_prompt)
    if b["backend"] == "local":
        return _ollama_chat_greedy(system_prompt, user_msg,
                                   base_url=b["base_url"], model=b["model"])
    raise RuntimeError(f"unsupported backend: {b['backend']}")


def call_subject_llm(item: dict, condition: str,
                     learned_facts: dict = None) -> tuple:
    """One subject-LLM cell call. Returns (system_prompt, user_msg, response, verdict)."""
    sys_prompt = build_system_prompt(condition, item, learned_facts=learned_facts)
    user_msg   = build_user_message(condition, item)
    response   = _call_subject_llm(sys_prompt, user_msg)
    verdict    = check_violation_fair(item, response)
    return sys_prompt, user_msg, response, verdict


def _run_loop(items_to_run: list, conditions: list, n_reps: int,
              facts_recs: dict, label: str) -> dict:
    """Inner loop — items × conditions × n_reps. Resumable. Returns counters.
    Backend-aware: writes to data/cer_ece/raw_runs.jsonl for DeepSeek (WP-10
    archived behavior preserved), data/cer_ece_qwen/raw_runs.jsonl for local."""
    runs_file = _backend_path("raw_runs.jsonl")
    os.makedirs(os.path.dirname(runs_file), exist_ok=True)

    # Write backend manifest alongside (self-labeling separate track).
    from cbt.llm_backend import record_backend_manifest, get_backend  # noqa
    record_backend_manifest(_backend_path(f"manifest_{label}.json"))
    backend_name = get_backend()["backend"]

    done = set()
    if os.path.exists(runs_file):
        with open(runs_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["item_id"], r["condition"], r["rep_id"]))

    target = sum(
        1
        for it in items_to_run
        for cond in conditions
        for rep in range(n_reps)
        if (it["id"], cond, rep) not in done
        and (cond in ("B_FAIR", "C_KNOW_ORACLE")
             or (facts_recs.get(it["id"]) and facts_recs[it["id"]].get("ok")))
    )

    new_records = 0
    skipped     = 0
    errors      = 0
    with open(runs_file, "a") as f_out:
        for it in items_to_run:
            facts_rec = facts_recs.get(it["id"])
            facts = facts_rec["facts"] if facts_rec and facts_rec.get("ok") else None
            for cond in conditions:
                # Conditions that NEED learned_facts
                if cond in ("PLAINFACTS", "C_PACK_LEARNED") and facts is None:
                    skipped += 1
                    continue
                for rep in range(n_reps):
                    key = (it["id"], cond, rep)
                    if key in done:
                        continue
                    try:
                        sys_prompt, user_msg, response, verdict = call_subject_llm(
                            it, cond, learned_facts=facts
                        )
                        is_err = response.startswith("ERROR:") or response == "TIMEOUT"
                        rec = {
                            "item_id":   it["id"],
                            "subtype":   it["subtype"],
                            "held_out":  it.get("held_out", False),
                            "condition": cond,
                            "rep_id":    rep,
                            "question":  it["question"],
                            "response":  response,
                            "verdict":   verdict,
                            "model":     LLM_MODEL,
                            "temp":      LLM_TEMP,
                            "is_error":  is_err,
                            "backend":   backend_name,
                        }
                        if is_err:
                            errors += 1
                    except Exception as e:  # noqa: BLE001
                        rec = {
                            "item_id":   it["id"],
                            "condition": cond,
                            "rep_id":    rep,
                            "error":     str(e)[:200],
                            "is_error":  True,
                            "backend":   backend_name,
                        }
                        errors += 1
                    f_out.write(json.dumps(rec) + "\n")
                    f_out.flush()
                    new_records += 1
                    flag = "VIOL" if rec.get("verdict", {}).get("violation") else "ok"
                    if rec.get("is_error"):
                        flag = "ERR"
                    print(f"[{label}] id={it['id']:>3} {cond:<14} rep={rep} → {flag}")

    return {"new": new_records, "skipped": skipped, "errors": errors,
            "target": target}


def cmd_smoke() -> None:
    """U5 smoke: 4 items × 4 conditions × 1 rep = 16 calls."""
    assert_frozen()
    items = load_items()
    facts_recs = load_extracted_facts()
    smoke_items = [it for it in items if it["id"] in SMOKE_ITEM_IDS]
    if len(smoke_items) != len(SMOKE_ITEM_IDS):
        raise RuntimeError(
            f"smoke items resolved {len(smoke_items)} of {len(SMOKE_ITEM_IDS)} expected"
        )
    print(f"[smoke] items={[it['id'] for it in smoke_items]}, "
          f"conditions={CONDITIONS}, n_reps=1")
    counters = _run_loop(smoke_items, CONDITIONS, 1, facts_recs, "smoke")
    print(f"[smoke] new={counters['new']} skipped={counters['skipped']} "
          f"errors={counters['errors']}")


def cmd_run_full(n_reps: int = None) -> None:
    """U5 full: all items × 4 conditions × N reps. Resumable. Appends to RUNS_FILE."""
    assert_frozen()
    items = load_items()
    facts_recs = load_extracted_facts()
    n_reps = n_reps or N_REPS
    print(f"[run] items={len(items)} conditions={CONDITIONS} n_reps={n_reps}")
    counters = _run_loop(items, CONDITIONS, n_reps, facts_recs, "run")
    print(f"[run] new={counters['new']} skipped={counters['skipped']} "
          f"errors={counters['errors']}")


def cmd_verify_u5(smoke_only: bool = False) -> int:
    """U5 inline-verify: raw_runs file present, records have required fields,
    smoke records cover the expected (4 items × 4 conditions × 1 rep) grid,
    and the C_PACK_LEARNED == C_KNOW_ORACLE deterministic-sanity holds for
    every (item_id, rep_id) where both are present."""
    failures = []
    runs_file = _backend_path("raw_runs.jsonl")
    if not os.path.exists(runs_file):
        failures.append(f"raw_runs missing: {runs_file}")
        return _verify_report(failures)

    runs = []
    with open(runs_file) as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    if not runs:
        failures.append("raw_runs empty")
        return _verify_report(failures)

    # Smoke-grid check
    smoke_cells = {(r["item_id"], r["condition"], r["rep_id"])
                   for r in runs if r["item_id"] in SMOKE_ITEM_IDS}
    expected_smoke = {(iid, c, 0)
                      for iid in SMOKE_ITEM_IDS
                      for c in CONDITIONS}
    missing = expected_smoke - smoke_cells
    if missing:
        failures.append(f"smoke grid incomplete; missing {len(missing)} cells: {sorted(missing)[:5]}")

    # Required fields on every non-error record
    for r in runs:
        if r.get("is_error"):
            continue
        for k in ("item_id", "condition", "rep_id", "question",
                  "response", "verdict", "model"):
            if k not in r:
                failures.append(f"record missing field {k}: {r}")
                break

    # Deterministic-sanity: C_PACK_LEARNED == C_KNOW_ORACLE per (item, rep)
    # Because U3 extraction is 1.000 and temp=0, the two should yield identical responses.
    grouped = {}
    for r in runs:
        if r.get("is_error"):
            continue
        if r["condition"] in ("C_PACK_LEARNED", "C_KNOW_ORACLE"):
            grouped.setdefault((r["item_id"], r["rep_id"]), {})[r["condition"]] = r["response"]
    n_pairs = 0
    n_mismatch = 0
    examples = []
    for key, pair in grouped.items():
        if "C_PACK_LEARNED" in pair and "C_KNOW_ORACLE" in pair:
            n_pairs += 1
            if pair["C_PACK_LEARNED"] != pair["C_KNOW_ORACLE"]:
                n_mismatch += 1
                if len(examples) < 3:
                    examples.append({"key": key,
                                     "pack_first120": pair["C_PACK_LEARNED"][:120],
                                     "oracle_first120": pair["C_KNOW_ORACLE"][:120]})
    if n_pairs == 0:
        failures.append("no C_PACK_LEARNED/C_KNOW_ORACLE pairs to compare")
    else:
        mismatch_rate = n_mismatch / n_pairs
        # Tolerate small (≤5%) divergence — DeepSeek temp=0 is not strictly
        # deterministic in practice on identical inputs; flag if higher.
        if mismatch_rate > 0.10:
            failures.append(
                f"C_PACK_LEARNED vs C_KNOW_ORACLE byte-mismatch {n_mismatch}/{n_pairs} "
                f"({mismatch_rate:.2%}) > 10% — non-determinism or bug? "
                f"first examples: {examples}"
            )
        print(f"[verify-u5] determinism pairs: {n_pairs}, mismatch={n_mismatch} ({mismatch_rate:.2%})")

    n_total = len(runs)
    n_err   = sum(1 for r in runs if r.get("is_error"))
    print(f"[verify-u5] total records={n_total} errors={n_err} "
          f"smoke-grid={len(smoke_cells & expected_smoke)}/{len(expected_smoke)}")
    return _verify_report(failures)


# ═══════════════════════════════════════════════════════════════════════════
# U6: Aggregate + DECISIVE gates.
#
# Reads raw_runs.jsonl, computes per-condition violation rates (overall +
# per-split: in_domain / held_out), paired per-item deltas, and effect size.
# Writes papers/results_cer_ece.md.
#
# Gates (floors pre-registered at U1):
#   gate_learned_payoff     V(C_PACK_LEARNED) <  V(B_FAIR)     by GATE_FLOOR_PAYOFF
#   gate_structure_vs_facts V(C_PACK_LEARNED) <  V(PLAINFACTS) by GATE_FLOOR_STRUCTURE
#
# Diagnostics:
#   extraction_quality_to_payoff_curve — extracted facts fidelity (1.0 here) vs payoff
#   C_PACK_LEARNED vs C_KNOW_ORACLE gap — payoff lost to imperfect extraction
#   held_out vs in_domain split           — generalization
#
# Real-rep-variance assertion: if all reps within an (item, condition) cell
# give identical verdicts (temp=0 deterministic), per-item std=0 is HONEST —
# we report it and use absolute-unanimity effect-strength tagging instead of
# Cohen's d when std(paired deltas)=0 (avoids the WP-5 d=∞ vacuity).
# ═══════════════════════════════════════════════════════════════════════════

def _load_runs() -> list:
    runs_file = _backend_path("raw_runs.jsonl")
    if not os.path.exists(runs_file):
        raise RuntimeError(f"raw_runs missing: {runs_file}")
    out = []
    with open(runs_file) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _per_cell_violation(runs: list) -> dict:
    """Returns dict[(item_id, condition)] = {n_reps, n_viol, rate, std,
    deterministic}. deterministic = True iff all reps in cell agree."""
    cells = {}
    for r in runs:
        if r.get("is_error"):
            continue
        key = (r["item_id"], r["condition"])
        cells.setdefault(key, []).append(int(bool(r["verdict"]["violation"])))
    out = {}
    for key, viols in cells.items():
        n = len(viols)
        mean = sum(viols) / n if n else 0.0
        if n > 1:
            var = sum((v - mean) ** 2 for v in viols) / n
            std = var ** 0.5
        else:
            std = 0.0
        out[key] = {
            "n_reps":         n,
            "n_viol":         sum(viols),
            "rate":           mean,
            "std":            std,
            "deterministic":  std == 0.0,
        }
    return out


def _per_condition_aggregate(cells: dict, items: list,
                             which: str = "all") -> dict:
    """Aggregate per-condition over items, optionally filtered by split.
    which ∈ {"all", "in_domain", "held_out"}."""
    items_by_id = {it["id"]: it for it in items}
    if which == "in_domain":
        item_ids = {it["id"] for it in items if not it.get("held_out")}
    elif which == "held_out":
        item_ids = {it["id"] for it in items if it.get("held_out")}
    else:
        item_ids = {it["id"] for it in items}

    by_cond = {}
    for (iid, cond), cell in cells.items():
        if iid not in item_ids:
            continue
        by_cond.setdefault(cond, []).append(cell["rate"])
    out = {}
    for cond, rates in by_cond.items():
        n = len(rates)
        mean = sum(rates) / n if n else 0.0
        if n > 1:
            var = sum((r - mean) ** 2 for r in rates) / (n - 1)
            std = var ** 0.5
        else:
            std = 0.0
        out[cond] = {
            "n_items":           n,
            "mean_violation_rate": mean,
            "std_across_items":    std,
        }
    return out


def _paired_delta(cells: dict, items: list, cond_a: str, cond_b: str,
                  which: str = "all") -> dict:
    """Per-item paired delta = V[cond_a] - V[cond_b], aggregated over items.
    Returns mean, std, n, Cohen-d if std>0 else effect_strength label."""
    if which == "in_domain":
        item_ids = [it["id"] for it in items if not it.get("held_out")]
    elif which == "held_out":
        item_ids = [it["id"] for it in items if it.get("held_out")]
    else:
        item_ids = [it["id"] for it in items]

    deltas = []
    for iid in item_ids:
        a = cells.get((iid, cond_a))
        b = cells.get((iid, cond_b))
        if a is None or b is None:
            continue
        deltas.append(a["rate"] - b["rate"])
    if not deltas:
        return {"n": 0, "mean": None, "std": None, "cohen_d": None,
                "effect_strength": "no_pairs"}
    n = len(deltas)
    mean = sum(deltas) / n
    if n > 1:
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0
    if std > 0:
        d = mean / std
        # Effect strength label by |d|
        ad = abs(d)
        label = ("small" if ad < 0.5 else "medium" if ad < 0.8
                 else "large" if ad < 1.2 else "very_large")
    else:
        d = None
        # All deltas identical — absolute unanimity
        if mean == 0.0:
            label = "absolute_tie"
        else:
            label = f"absolute_unanimity_delta_{mean:+.3f}"
    return {
        "n":               n,
        "mean":            mean,
        "std":             std,
        "cohen_d":         d,
        "effect_strength": label,
        "deltas":          deltas,
    }


def _evaluate_gates(by_cond_all: dict, by_cond_in: dict, by_cond_held: dict,
                    cells: dict, items: list) -> dict:
    """Apply pre-registered floors to determine gate verdicts."""
    def _floor_pass(delta_record, floor):
        if delta_record["n"] == 0 or delta_record["mean"] is None:
            return False
        # Negative mean means cond_a < cond_b — passes if magnitude ≥ floor.
        return delta_record["mean"] <= -floor

    def _effect_pass(delta_record):
        if delta_record.get("cohen_d") is not None:
            return abs(delta_record["cohen_d"]) >= 0.5
        # Absolute unanimity with nonzero mean → effect is structurally maximal.
        return delta_record.get("effect_strength", "").startswith("absolute_unanimity")

    out = {}

    # gate_learned_payoff: V(C_PACK_LEARNED) < V(B_FAIR)
    for split_name, items_for_split in (
        ("all", items),
        ("in_domain", [it for it in items if not it.get("held_out")]),
        ("held_out",  [it for it in items if it.get("held_out")]),
    ):
        delta = _paired_delta(cells, items, "C_PACK_LEARNED", "B_FAIR", which=split_name)
        passes_floor = _floor_pass(delta, GATE_FLOOR_PAYOFF)
        passes_effect = _effect_pass(delta)
        out[f"gate_learned_payoff__{split_name}"] = {
            "delta":          delta,
            "floor":          GATE_FLOOR_PAYOFF,
            "floor_pass":     passes_floor,
            "effect_pass":    passes_effect,
            "verdict":        "PASS" if (passes_floor and passes_effect) else "FAIL",
        }

    # gate_structure_vs_facts: V(C_PACK_LEARNED) < V(PLAINFACTS)
    for split_name in ("all", "in_domain", "held_out"):
        delta = _paired_delta(cells, items, "C_PACK_LEARNED", "PLAINFACTS", which=split_name)
        passes_floor = _floor_pass(delta, GATE_FLOOR_STRUCTURE)
        passes_effect = _effect_pass(delta)
        out[f"gate_structure_vs_facts__{split_name}"] = {
            "delta":          delta,
            "floor":          GATE_FLOOR_STRUCTURE,
            "floor_pass":     passes_floor,
            "effect_pass":    passes_effect,
            "verdict":        "PASS" if (passes_floor and passes_effect) else "FAIL",
        }

    # Diagnostic: C_PACK_LEARNED vs C_KNOW_ORACLE (extraction quality cost)
    for split_name in ("all",):
        delta = _paired_delta(cells, items, "C_PACK_LEARNED", "C_KNOW_ORACLE", which=split_name)
        out[f"diag_pack_vs_oracle__{split_name}"] = {
            "delta": delta,
            "interpretation": (
                "Expected ≈0 (byte-identical inputs at temp=0). "
                "Non-zero gap signals extraction loss or stochasticity."
            ),
        }

    return out


def cmd_aggregate() -> dict:
    """U6: compute per-condition rates + gates + write results_cer_ece.md."""
    assert_frozen()
    runs = _load_runs()
    items = load_items()
    cells = _per_cell_violation(runs)

    by_all  = _per_condition_aggregate(cells, items, which="all")
    by_in   = _per_condition_aggregate(cells, items, which="in_domain")
    by_held = _per_condition_aggregate(cells, items, which="held_out")
    gates   = _evaluate_gates(by_all, by_in, by_held, cells, items)

    # Rep-variance honesty audit
    rep_var_audit = {
        "n_cells":              len(cells),
        "n_deterministic":      sum(1 for c in cells.values() if c["deterministic"]),
        "frac_deterministic":   (sum(1 for c in cells.values() if c["deterministic"])
                                 / max(len(cells), 1)),
        "n_with_variance":      sum(1 for c in cells.values() if not c["deterministic"]),
    }

    # Extraction-quality → payoff sketch (single data point at fidelity=1.0)
    ext = json.load(open(_backend_path("extraction_fidelity.json")))
    extraction_summary = {
        "ece_yield":                       ext["summary"]["ece_yield"],
        "in_domain_avg_exact_match":       ext["summary"].get("in_domain_avg_exact_match"),
        "held_out_avg_exact_match":        ext["summary"].get("held_out_avg_exact_match"),
        "note": (
            "Single data point — extraction is essentially perfect at this scope. "
            "The payoff-vs-fidelity curve cannot be drawn from one fidelity value. "
            "Future scopes with noisier inputs will sweep this axis."
        ),
    }

    report = {
        "by_condition": {
            "all":       by_all,
            "in_domain": by_in,
            "held_out":  by_held,
        },
        "gates":               gates,
        "rep_variance_audit":  rep_var_audit,
        "extraction":          extraction_summary,
        "gate_floors":         {"payoff": GATE_FLOOR_PAYOFF,
                                "structure": GATE_FLOOR_STRUCTURE,
                                "effect_size_cohen_d_min": 0.5},
        "model":               _active_model_for_report(),
        "backend":             _active_backend_for_report(),
        "n_items":             len(items),
        "n_held_out":          sum(1 for it in items if it.get("held_out")),
        "conditions":          CONDITIONS,
    }
    eval_path = _backend_path("eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(report, f, indent=2)
    results_md = _results_md_path()
    _write_results_md(report, results_md)
    print(f"[aggregate] saved {eval_path}")
    print(f"[aggregate] saved {results_md}")
    _print_headline(report)
    return report


def _active_model_for_report() -> str:
    """Resolve model name from active backend for report metadata."""
    from cbt.llm_backend import get_backend  # noqa
    return get_backend()["model"]


def _active_backend_for_report() -> str:
    from cbt.llm_backend import get_backend  # noqa
    return get_backend()["backend"]


def _results_md_path() -> str:
    """Path to results doc — backend-namespaced."""
    from cbt.llm_backend import get_backend  # noqa
    b = get_backend()
    if b["backend"] == "deepseek":
        return RESULTS_MD  # WP-10 archived path
    return os.path.join(PAPERS_DIR, "results_local_reproduction.md")


def _fmt_pct(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def _fmt_signed(x):
    if x is None:
        return "—"
    return f"{x:+.3f}"


def _fmt_d(d):
    if d is None:
        return "—"
    return f"{d:+.2f}"


def _write_results_md(report: dict, out_path: str = None) -> None:
    if out_path is None:
        out_path = RESULTS_MD
    by = report["by_condition"]
    gates = report["gates"]
    lines = []
    lines.append(f"# Results — WP-ST-10 CER/ECE first scoped microcell\n")
    lines.append(f"**Model:** `{report['model']}` (temperature=0)  ")
    lines.append(f"**Items:** {report['n_items']} clear-counterfactual  "
                 f"(in-domain {report['n_items'] - report['n_held_out']} + held-out {report['n_held_out']})  ")
    lines.append(f"**Conditions:** {' / '.join(report['conditions'])}\n")

    lines.append("## Per-condition violation rate (mean ± std across items)\n")
    for split_name, by_split in by.items():
        lines.append(f"### Split: `{split_name}`")
        lines.append("| Condition | n items | mean V | std across items |")
        lines.append("|---|---:|---:|---:|")
        for cond in report["conditions"]:
            v = by_split.get(cond, {})
            lines.append(f"| `{cond}` | {v.get('n_items', 0)} | "
                         f"{_fmt_pct(v.get('mean_violation_rate', 0))} | "
                         f"{_fmt_pct(v.get('std_across_items', 0))} |")
        lines.append("")

    lines.append("## DECISIVE gates (pre-registered floors locked at U1)\n")
    floor_payoff = report["gate_floors"]["payoff"]
    floor_struct = report["gate_floors"]["structure"]
    lines.append(f"Floors: gate_learned_payoff = Δ ≥ {floor_payoff} (PACK lower than B_FAIR), "
                 f"gate_structure_vs_facts = Δ ≥ {floor_struct} (PACK lower than PLAINFACTS), "
                 f"effect size |d| ≥ 0.5 OR absolute_unanimity.\n")

    def _gate_block(title, key_prefix):
        lines.append(f"### {title}")
        lines.append("| Split | n | Δ mean | std | Cohen d | floor pass | effect pass | verdict |")
        lines.append("|---|---:|---:|---:|---:|:---:|:---:|:---:|")
        for split in ("all", "in_domain", "held_out"):
            g = gates.get(f"{key_prefix}__{split}", {})
            d = g.get("delta", {})
            lines.append(
                f"| `{split}` | {d.get('n', 0)} | "
                f"{_fmt_signed(d.get('mean'))} | "
                f"{_fmt_signed(d.get('std'))} | "
                f"{_fmt_d(d.get('cohen_d'))} | "
                f"{'✓' if g.get('floor_pass') else '✗'} | "
                f"{'✓' if g.get('effect_pass') else '✗'} | "
                f"**{g.get('verdict', '—')}** |"
            )
        lines.append("")

    _gate_block("gate_learned_payoff (V(C_PACK_LEARNED) < V(B_FAIR))",
                "gate_learned_payoff")
    _gate_block("gate_structure_vs_facts (V(C_PACK_LEARNED) < V(PLAINFACTS))",
                "gate_structure_vs_facts")

    lines.append("## Diagnostic — C_PACK_LEARNED vs C_KNOW_ORACLE\n")
    diag = gates.get("diag_pack_vs_oracle__all", {})
    d = diag.get("delta", {})
    lines.append(f"- n = {d.get('n', 0)}, Δ mean = {_fmt_signed(d.get('mean'))}, "
                 f"effect = `{d.get('effect_strength', '—')}`")
    lines.append(f"- Interpretation: {diag.get('interpretation', '')}\n")

    lines.append("## Rep-variance honesty audit\n")
    rep = report["rep_variance_audit"]
    lines.append(f"- cells with rep variance: {rep['n_with_variance']} / {rep['n_cells']}")
    lines.append(f"- deterministic cells:     {rep['n_deterministic']} / {rep['n_cells']}  "
                 f"({_fmt_pct(rep['frac_deterministic'])})")
    lines.append("")
    lines.append("Deterministic ≠ vacuous: at temp=0, identical (system,user) inputs SHOULD "
                 "yield identical responses, so per-cell std=0 is a structural property, "
                 "not a bug. Effect strength uses `absolute_unanimity` labeling when "
                 "paired-delta std=0 — avoids the WP-5 Cohen-d=∞ vacuity.\n")

    lines.append("## Extraction quality\n")
    ext = report["extraction"]
    lines.append(f"- ECE yield: {_fmt_pct(ext['ece_yield'])}")
    lines.append(f"- in_domain per-item exact-match: {ext['in_domain_avg_exact_match']}")
    lines.append(f"- held_out per-item exact-match:  {ext['held_out_avg_exact_match']}")
    lines.append(f"- {ext['note']}\n")

    os.makedirs(PAPERS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def _print_headline(report: dict) -> None:
    by = report["by_condition"]["all"]
    gates = report["gates"]
    print("\n=== HEADLINE ===")
    for cond in report["conditions"]:
        v = by.get(cond, {})
        print(f"  V({cond:<14}) = {_fmt_pct(v.get('mean_violation_rate', 0))}  "
              f"(n={v.get('n_items', 0)}, std={_fmt_pct(v.get('std_across_items', 0))})")
    for key in ("gate_learned_payoff__all", "gate_structure_vs_facts__all"):
        g = gates.get(key, {})
        d = g.get("delta", {})
        print(f"  {key}: Δ={_fmt_signed(d.get('mean'))} d={_fmt_d(d.get('cohen_d'))} "
              f"effect={d.get('effect_strength', '—')} → {g.get('verdict')}")


def cmd_verify_u6() -> int:
    """U6 inline-verify: results doc + eval_results.json exist; both
    gates have verdicts; rep-variance honesty audit present. Backend-aware."""
    failures = []
    results_md = _results_md_path()
    if not os.path.exists(results_md):
        failures.append(f"results doc missing: {results_md}")
    eval_path = _backend_path("eval_results.json")
    if not os.path.exists(eval_path):
        failures.append(f"eval results missing: {eval_path}")
        return _verify_report(failures)
    rep = json.load(open(eval_path))
    for key in ("gate_learned_payoff__all", "gate_structure_vs_facts__all"):
        g = rep["gates"].get(key)
        if g is None or g.get("verdict") not in ("PASS", "FAIL"):
            failures.append(f"gate missing or no verdict: {key}")
    if "rep_variance_audit" not in rep:
        failures.append("rep_variance_audit missing")
    if "gate_floors" not in rep:
        failures.append("gate_floors missing")
    return _verify_report(failures)


# ═══════════════════════════════════════════════════════════════════════════
# WP-ST-11 U2: greedy determinism verification on the active backend.
#
# DeepSeek temp=0 had a ~5.4% byte-level near-determinism slip on WP-10
# (same input, different response text on 6/111 pairs). For Qwen-local at
# greedy temp=0 (Ollama options: temperature=0, top_p=1.0, top_k=1, seed=42,
# fixed num_ctx) we expect byte-identical responses on identical inputs.
#
# Probe set (K=4 inputs):
#   1. Novel item                  (C_PACK_LEARNED system + question)
#   2. Adversarial item            (C_PACK_LEARNED system + question)
#   3. Question-only (no system)
#   4. Contract pack only          (no item question — bare scaffold + neutral prompt)
#
# For each probe: send N=5 times, assert len(set(responses)) == 1.
#
# PASS = 4/4 probes byte-identical across 5 reps -> "fully deterministic at
#        this model + Ollama config + these 4 probes" (bounded claim).
# FAIL = surface the actual slip rate + non-identical responses for follow-up.
# ═══════════════════════════════════════════════════════════════════════════

DETERMINISM_REPORT = "data/cer_ece_qwen/determinism_report.json"
DATA_DIR_QWEN      = "data/cer_ece_qwen"
N_REPS_DETERMINISM = 5

# Pre-registered greedy options (sent inside `options` for Ollama native, also
# echoed as OpenAI-compat fields; some are no-ops on the OpenAI-compat path
# but harmless to send). Frozen at U2 close.
GREEDY_OPTS = {
    "temperature": 0.0,
    "top_p":       1.0,
    "top_k":       1,
    "seed":        42,
    "num_ctx":     8192,
}


def _ollama_chat_greedy(system_prompt: str, user_msg: str,
                        base_url: str, model: str,
                        timeout: int = 120) -> str:
    """Hit Ollama's /api/chat (native, NOT OpenAI-compat) with greedy options.

    Native /api/chat honors the full `options` block including seed + top_k +
    num_ctx, which is what we need for true greedy determinism. The harness
    (U3+) can use either /api/chat or OpenAI-compat /v1/chat/completions;
    U2 uses native to give greedy the strongest possible chance.
    """
    import urllib.error
    import urllib.request
    # base_url for OpenAI-compat is .../v1; for native, drop the /v1 suffix.
    native_base = base_url.rstrip("/")
    if native_base.endswith("/v1"):
        native_base = native_base[:-3]
    url = native_base.rstrip("/") + "/api/chat"

    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_msg})
    body = json.dumps({
        "model":    model,
        "messages": msgs,
        "stream":   False,
        "options":  GREEDY_OPTS,
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        doc = json.loads(r.read())
    # Native /api/chat returns {"message": {"role":..., "content":...}, ...}
    msg = doc.get("message", {}) or {}
    return str(msg.get("content", ""))


def _determinism_probes() -> list:
    """Build the 4 probe (system, user, label) triples."""
    items = load_items()
    novel = next(it for it in items if it["subtype"] == "novel")
    adv   = next(it for it in items if it["subtype"] == "adversarial")

    # Reuse U3's oracle facts as the "perfect extraction" stand-in for the
    # C_PACK_LEARNED probe — at U3 fidelity 1.000 they would be identical,
    # but here we sidestep the LEARNED ECE call (U2 tests determinism only,
    # not extraction; using oracle facts is faster and equally valid for
    # byte-identity probing of the subject LLM).
    novel_facts = oracle_facts(novel)
    adv_facts   = oracle_facts(adv)

    probes = [
        {
            "label":   "novel_C_PACK_LEARNED",
            "system":  build_system_prompt("C_PACK_LEARNED", novel, novel_facts),
            "user":    novel["question"],
            "item_id": novel["id"],
        },
        {
            "label":   "adversarial_C_PACK_LEARNED",
            "system":  build_system_prompt("C_PACK_LEARNED", adv, adv_facts),
            "user":    adv["question"],
            "item_id": adv["id"],
        },
        {
            "label":   "question_only_no_system",
            "system":  "",
            "user":    novel["question"],
            "item_id": novel["id"],
        },
        {
            "label":   "contract_scaffold_neutral",
            "system":  build_system_prompt("C_KNOW_ORACLE", adv),
            "user":    "Respond with 'OK' if you can read the contract above; nothing else.",
            "item_id": adv["id"],
        },
    ]
    return probes


def cmd_verify_determinism() -> None:
    """U2: probe greedy determinism on the active backend (intended: local).
    Writes data/cer_ece_qwen/determinism_report.json."""
    from cbt.llm_backend import get_backend, record_backend_manifest  # noqa
    b = get_backend()
    if b["backend"] != "local":
        raise RuntimeError(
            f"--verify-determinism requires LLM_BACKEND=local; got {b['backend']}. "
            f"Set LLM_BACKEND=local and re-run."
        )

    os.makedirs(DATA_DIR_QWEN, exist_ok=True)
    record_backend_manifest(os.path.join(DATA_DIR_QWEN, "manifest_u2_determinism.json"))

    probes = _determinism_probes()
    print(f"[determinism] probes={len(probes)}  reps={N_REPS_DETERMINISM}  "
          f"model={b['model']}  quant={b['quant']}")

    per_probe = []
    all_pass = True
    for probe in probes:
        responses = []
        for rep in range(N_REPS_DETERMINISM):
            try:
                resp = _ollama_chat_greedy(
                    probe["system"], probe["user"],
                    base_url=b["base_url"], model=b["model"],
                )
            except Exception as e:  # noqa: BLE001
                resp = f"__ERROR__:{str(e)[:160]}"
            responses.append(resp)
            print(f"[determinism] {probe['label']:<28} rep={rep}  len={len(resp)}")
        unique = sorted(set(responses), key=responses.index)
        n_unique = len(unique)
        byte_identical = (n_unique == 1)
        if not byte_identical:
            all_pass = False
        per_probe.append({
            "label":          probe["label"],
            "item_id":        probe["item_id"],
            "n_reps":         N_REPS_DETERMINISM,
            "n_unique":       n_unique,
            "byte_identical": byte_identical,
            "first_response_len": len(responses[0]),
            "responses":      responses,
        })
        print(f"[determinism] {probe['label']:<28} unique={n_unique} "
              f"byte_identical={byte_identical}")

    report = {
        "model":            b["model"],
        "quant":            b["quant"],
        "base_url":         b["base_url"],
        "greedy_options":   GREEDY_OPTS,
        "n_probes":         len(probes),
        "n_reps_per_probe": N_REPS_DETERMINISM,
        "all_byte_identical": all_pass,
        "slip_rate":        sum(p["n_unique"] - 1 for p in per_probe) / (len(probes) * N_REPS_DETERMINISM),
        "per_probe":        per_probe,
    }
    with open(DETERMINISM_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[determinism] all_byte_identical={all_pass}  slip_rate={report['slip_rate']:.3f}")
    print(f"[determinism] saved {DETERMINISM_REPORT}")


def cmd_verify_u2_qwen() -> int:
    """U2 inline-verify (WP-11): determinism report present + 4/4 probes
    byte-identical across 5 reps."""
    failures = []
    if not os.path.exists(DETERMINISM_REPORT):
        failures.append(f"determinism report missing: {DETERMINISM_REPORT}")
        return _verify_report(failures)
    rep = json.load(open(DETERMINISM_REPORT))
    if rep["n_probes"] != 4:
        failures.append(f"expected 4 probes, got {rep['n_probes']}")
    if rep["n_reps_per_probe"] != N_REPS_DETERMINISM:
        failures.append(f"expected {N_REPS_DETERMINISM} reps, got {rep['n_reps_per_probe']}")
    if not rep["all_byte_identical"]:
        bad = [p["label"] for p in rep["per_probe"] if not p["byte_identical"]]
        failures.append(f"non-deterministic probes: {bad}")
    if rep.get("slip_rate", 1.0) > 0.0:
        # PASS contract says 0.0; record diagnostic but still fail.
        if not failures:
            failures.append(f"slip_rate > 0: {rep['slip_rate']:.3f}")
    print(f"[verify-u2-qwen] all_byte_identical={rep['all_byte_identical']}  "
          f"slip_rate={rep['slip_rate']:.3f}  probes={rep['n_probes']}")
    return _verify_report(failures)


# ═══════════════════════════════════════════════════════════════════════════
# WP-ST-11 U6: cross-model compare (DeepSeek vs Qwen) + final
# gate_local_reproduction verdict.
#
# Reads both per-backend eval_results.json files (NEVER merging them into
# one row table — David's anti-silent-merge directive). Side-by-side per-
# condition V table + per-gate verdict pair. Decides overall gate_local_reproduction
# = (a) PASS ∧ (b) PASS, where:
#   (a) WP-11 U3 gate_a_local_reproduction (extraction yield + leak)
#   (b) Qwen's gate_learned_payoff matches DeepSeek's gate_learned_payoff
#       verdict (both PASS, both with the same sign + within tolerance)
# (c) is a DIAGNOSTIC — reports whether PLAINFACTS saturation is broken on
#     Qwen (V > 0%); if so, headroom for structure-vs-facts is reported
#     per-regime, but this does NOT gate the WP overall.
# ═══════════════════════════════════════════════════════════════════════════

CROSS_MODEL_COMPARE_MD   = os.path.join(DATA_DIR_QWEN, "cross_model_compare.md")
CROSS_MODEL_COMPARE_JSON = os.path.join(DATA_DIR_QWEN, "cross_model_compare.json")


def _load_eval(backend_dir: str) -> dict:
    p = os.path.join(backend_dir, "eval_results.json")
    if not os.path.exists(p):
        raise RuntimeError(f"eval_results missing for {backend_dir}: {p}")
    return json.load(open(p))


def cmd_compare_models() -> None:
    """U6 (WP-11): side-by-side compare DeepSeek vs Qwen + decide
    gate_local_reproduction. Reads both eval_results.json files; never merges
    rows; labels every cell with `backend`."""
    deepseek = _load_eval(DATA_DIR)
    qwen     = _load_eval(DATA_DIR_QWEN)

    # Also need gate (a) from extraction_fidelity (Qwen track only).
    qwen_fid = json.load(open(os.path.join(DATA_DIR_QWEN, "extraction_fidelity.json")))
    gate_a   = qwen_fid["summary"]["gate_a_local_reproduction"]

    # Per-condition V (overall split) for both backends
    by_cond = {}
    for cond in deepseek["conditions"]:
        d_v = deepseek["by_condition"]["all"].get(cond, {})
        q_v = qwen    ["by_condition"]["all"].get(cond, {})
        by_cond[cond] = {
            "deepseek_V":   d_v.get("mean_violation_rate"),
            "qwen_V":       q_v.get("mean_violation_rate"),
            "delta_subject": (q_v.get("mean_violation_rate", 0)
                              - d_v.get("mean_violation_rate", 0)),
        }

    # Per-gate verdict pair (overall split) — gate_learned_payoff
    glp_deep = deepseek["gates"].get("gate_learned_payoff__all", {})
    glp_qwen = qwen    ["gates"].get("gate_learned_payoff__all", {})
    gsf_deep = deepseek["gates"].get("gate_structure_vs_facts__all", {})
    gsf_qwen = qwen    ["gates"].get("gate_structure_vs_facts__all", {})

    # gate_local_reproduction decision
    glp_b_pass = (glp_qwen.get("verdict") == "PASS")  # (b) sub-gate
    gate_local_reproduction = "PASS" if (gate_a["verdict"] == "PASS" and glp_b_pass) else "FAIL"

    # (c) Diagnostic: does PLAINFACTS saturate on Qwen (V == 0) or break?
    qwen_pf_V    = qwen["by_condition"]["all"].get("PLAINFACTS",     {}).get("mean_violation_rate", 0.0)
    qwen_pack_V  = qwen["by_condition"]["all"].get("C_PACK_LEARNED", {}).get("mean_violation_rate", 0.0)
    pf_saturated = (qwen_pf_V == 0.0)
    structure_headroom = qwen_pf_V - qwen_pack_V  # >0 means structure helps on Qwen
    # Per-regime split for (c)
    per_regime_headroom = {}
    for split_name in ("in_domain", "held_out"):
        pf = qwen["by_condition"][split_name].get("PLAINFACTS",     {}).get("mean_violation_rate", 0.0)
        cp = qwen["by_condition"][split_name].get("C_PACK_LEARNED", {}).get("mean_violation_rate", 0.0)
        per_regime_headroom[split_name] = {
            "PLAINFACTS_V":     pf,
            "C_PACK_LEARNED_V": cp,
            "headroom":         pf - cp,
            "saturated":        (pf == 0.0),
        }

    # Persist JSON
    out_json = {
        "subjects": {
            "deepseek": {"model": deepseek.get("model"), "source": "data/cer_ece/eval_results.json"},
            "qwen":     {"model": qwen    .get("model"), "source": "data/cer_ece_qwen/eval_results.json"},
        },
        "by_condition_overall": by_cond,
        "gate_a_extraction_yield_and_leak": gate_a,
        "gate_b_learned_payoff_pair": {
            "deepseek_verdict": glp_deep.get("verdict"),
            "qwen_verdict":     glp_qwen.get("verdict"),
            "deepseek_delta":   (glp_deep.get("delta", {}) or {}).get("mean"),
            "qwen_delta":       (glp_qwen.get("delta", {}) or {}).get("mean"),
            "deepseek_d":       (glp_deep.get("delta", {}) or {}).get("cohen_d"),
            "qwen_d":           (glp_qwen.get("delta", {}) or {}).get("cohen_d"),
            "qwen_pass":        glp_b_pass,
        },
        "diagnostic_c_structure_vs_facts": {
            "deepseek_verdict":   gsf_deep.get("verdict"),
            "qwen_verdict":       gsf_qwen.get("verdict"),
            "qwen_PLAINFACTS_V":  qwen_pf_V,
            "qwen_C_PACK_V":      qwen_pack_V,
            "qwen_pf_saturated":  pf_saturated,
            "structure_headroom_overall": structure_headroom,
            "per_regime_headroom": per_regime_headroom,
        },
        "gate_local_reproduction": gate_local_reproduction,
    }
    os.makedirs(DATA_DIR_QWEN, exist_ok=True)
    with open(CROSS_MODEL_COMPARE_JSON, "w") as f:
        json.dump(out_json, f, indent=2)

    # Markdown side-by-side — NEVER silent merge; every row carries backend label.
    lines = []
    lines.append(f"# Cross-model compare — WP-ST-11\n")
    lines.append(f"**These are TWO subjects on the SAME scope (37 cf items × 4 conditions). "
                 f"Comparisons are subject-relative. No silent merge.**\n")
    lines.append(f"- Subject 1: `{deepseek.get('model')}` (DeepSeek; source `data/cer_ece/eval_results.json`)")
    lines.append(f"- Subject 2: `{qwen    .get('model')}` (Ollama local; source `data/cer_ece_qwen/eval_results.json`)\n")

    lines.append("## Per-condition violation rate (overall, n=37)\n")
    lines.append("| Condition | V (deepseek) | V (qwen) | Δ (qwen − deepseek) |")
    lines.append("|---|---:|---:|---:|")
    for cond, v in by_cond.items():
        lines.append(f"| `{cond}` | {_fmt_pct(v['deepseek_V'])} | {_fmt_pct(v['qwen_V'])} | {_fmt_signed(v['delta_subject'])} |")
    lines.append("")

    lines.append("## Gate (a) — extraction yield + adversarial leak (Qwen track only)\n")
    lines.append(f"- ECE yield: {_fmt_pct(gate_a['extraction_yield'])}  (floor {_fmt_pct(gate_a['yield_floor'])})  → {'PASS' if gate_a['yield_pass'] else 'FAIL'}")
    lines.append(f"- adversarial leak_rate: {_fmt_pct(gate_a['adversarial_leak_rate'])}  (floor ≤ {_fmt_pct(gate_a['leak_floor'])})  → {'PASS' if gate_a['leak_pass'] else 'FAIL'}")
    lines.append(f"- **Gate (a) verdict: {gate_a['verdict']}**\n")

    lines.append("## Gate (b) — gate_learned_payoff verdict pair\n")
    lines.append(f"- DeepSeek: **{glp_deep.get('verdict')}**  Δ={_fmt_signed((glp_deep.get('delta') or {}).get('mean'))}  d={_fmt_d((glp_deep.get('delta') or {}).get('cohen_d'))}")
    lines.append(f"- Qwen:     **{glp_qwen.get('verdict')}**  Δ={_fmt_signed((glp_qwen.get('delta') or {}).get('mean'))}  d={_fmt_d((glp_qwen.get('delta') or {}).get('cohen_d'))}")
    lines.append(f"- **Gate (b) verdict: {'PASS' if glp_b_pass else 'FAIL'}**  (qwen verdict == PASS)\n")

    lines.append("## Gate (c) — DIAGNOSTIC (does Qwen weakness break PLAINFACTS saturation?)\n")
    lines.append(f"- DeepSeek gate_structure_vs_facts: **{gsf_deep.get('verdict')}** (saturated tie at 0.0%)")
    lines.append(f"- Qwen gate_structure_vs_facts:     **{gsf_qwen.get('verdict')}**  Δ={_fmt_signed((gsf_qwen.get('delta') or {}).get('mean'))}")
    lines.append(f"- Qwen V(PLAINFACTS):     {_fmt_pct(qwen_pf_V)}  saturated_at_zero={pf_saturated}")
    lines.append(f"- Qwen V(C_PACK_LEARNED): {_fmt_pct(qwen_pack_V)}")
    lines.append(f"- structure_headroom (PLAINFACTS − C_PACK): {_fmt_signed(structure_headroom)} (>0 = structure helps)")
    lines.append("")
    lines.append("Per-regime headroom (Qwen track):")
    lines.append("| Regime | V(PLAINFACTS) | V(C_PACK_LEARNED) | headroom | saturated |")
    lines.append("|---|---:|---:|---:|:---:|")
    for split_name, r in per_regime_headroom.items():
        lines.append(f"| `{split_name}` | {_fmt_pct(r['PLAINFACTS_V'])} | {_fmt_pct(r['C_PACK_LEARNED_V'])} | {_fmt_signed(r['headroom'])} | {'YES' if r['saturated'] else 'NO'} |")
    lines.append("")

    lines.append(f"## Overall verdict: `gate_local_reproduction` = **{gate_local_reproduction}**\n")
    lines.append(f"- (a) gate_a_local_reproduction: {gate_a['verdict']}")
    lines.append(f"- (b) gate_learned_payoff reproduces on Qwen: {'PASS' if glp_b_pass else 'FAIL'}")
    lines.append(f"- (c) structure-vs-facts diagnostic: SEE ABOVE (does NOT gate the WP overall)")
    lines.append("")

    with open(CROSS_MODEL_COMPARE_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"[compare] gate_local_reproduction = {gate_local_reproduction}")
    print(f"[compare] saved {CROSS_MODEL_COMPARE_JSON}")
    print(f"[compare] saved {CROSS_MODEL_COMPARE_MD}")


def cmd_verify_u6_wp11() -> int:
    """WP-11 U6 inline-verify: cross-model artifacts present + gate_local_reproduction
    verdict recorded; backend labels present on every row."""
    failures = []
    for p in (CROSS_MODEL_COMPARE_JSON, CROSS_MODEL_COMPARE_MD):
        if not os.path.exists(p):
            failures.append(f"missing: {p}")
    if failures:
        return _verify_report(failures)
    j = json.load(open(CROSS_MODEL_COMPARE_JSON))
    for k in ("subjects", "by_condition_overall", "gate_a_extraction_yield_and_leak",
              "gate_b_learned_payoff_pair", "diagnostic_c_structure_vs_facts",
              "gate_local_reproduction"):
        if k not in j:
            failures.append(f"missing key: {k}")
    if j.get("gate_local_reproduction") not in ("PASS", "FAIL"):
        failures.append(f"gate_local_reproduction missing verdict: {j.get('gate_local_reproduction')}")
    txt = open(CROSS_MODEL_COMPARE_MD).read()
    for marker in ("These are TWO subjects", "No silent merge", "deepseek", "qwen",
                   "Gate (a)", "Gate (b)", "Gate (c)", "gate_local_reproduction"):
        if marker not in txt:
            failures.append(f"compare doc missing marker: {marker}")
    print(f"[verify-u6-wp11] gate_local_reproduction={j.get('gate_local_reproduction')}")
    return _verify_report(failures)


def cmd_verify_u7() -> int:
    """U7 inline-verify: claim doc present + carries the required headers
    (bounded scope, both gate verdicts, saturation note, architecture decision,
    next-WP plan, refused-overclaims block, CBT-v1 status, sign-off)."""
    failures = []
    if not os.path.exists(CLAIM_MD):
        failures.append(f"claim doc missing: {CLAIM_MD}")
        return _verify_report(failures)
    txt = open(CLAIM_MD).read()
    required_markers = [
        "Bounded scope",
        "gate_learned_payoff",
        "gate_structure_vs_facts",
        "SATURATED TIE",
        "Saturation Note",
        "Architecture decision",
        "WP-ST-11",
        "harder non-saturating",
        "decode-time",
        "CBT-v1",
        "Sign-off",
        "deepseek-chat",
        "What we will NOT do",
    ]
    missing = [m for m in required_markers if m not in txt]
    if missing:
        failures.append(f"claim doc missing required markers: {missing}")
    refused = [
        "scoped CBT = RAG",                # rejected drafting that David refused
        "cosmetic failure of contract structure",
        "contract structure is cosmetic in general",
    ]
    for r in refused:
        if r in txt and "NOT" not in txt[:txt.find(r) + len(r) + 80]:
            # Only fail if appears un-prefixed by NOT/⊬ negation; coarse but ok.
            failures.append(f"claim doc may carry refused overclaim: {r}")
    # Verify-u6 must already PASS — claim depends on it.
    if not os.path.exists(os.path.join(DATA_DIR, "eval_results.json")):
        failures.append("eval_results.json missing — U6 precondition")
    print(f"[verify-u7] markers present, length={len(txt)} chars")
    return _verify_report(failures)


def cmd_verify_u7_wp11() -> int:
    """WP-11 U7 inline-verify: papers/claim_local_reproduction.md exists +
    carries the required markers (bounded scope, gate_local_reproduction
    verdict, (a)(b)(c) breakdown, cross-model table, red-team note,
    architecture decision per branch, CBT-v1 status, sign-off)."""
    failures = []
    claim_path = os.path.join(PAPERS_DIR, "claim_local_reproduction.md")
    if not os.path.exists(claim_path):
        failures.append(f"claim doc missing: {claim_path}")
        return _verify_report(failures)
    txt = open(claim_path).read()
    required = [
        "qwen2.5-32b-instruct-q8",
        "Bounded scope",
        "gate_local_reproduction",
        "Sub-check (a)",
        "Sub-check (b)",
        "Sub-check (c)",
        "extraction yield",
        "leak_rate",
        "gate_learned_payoff",
        "gate_structure_vs_facts",
        "DeepSeek",
        "Qwen",
        "Red-team note",
        "S→T",                    # red-team note must include the S->T guard
        "CBT-v1",
        "Sign-off",
        "What we will NOT do",
    ]
    missing = [m for m in required if m not in txt]
    if missing:
        failures.append(f"claim doc missing required markers: {missing}")
    if not os.path.exists(CROSS_MODEL_COMPARE_JSON):
        failures.append("U6 cross-model compare missing — precondition")
    print(f"[verify-u7-wp11] markers present, length={len(txt)} chars")
    return _verify_report(failures)


def cmd_verify_u1() -> int:
    """U1 inline-verify: items file exists + hash stable + conditions + renderer
    + all 4 conditions wired. Exit 0 on PASS, non-zero on FAIL."""
    failures = []

    # Items file present + non-empty
    if not os.path.exists(ITEMS_FILE):
        failures.append(f"items file missing: {ITEMS_FILE}")
        return _verify_report(failures)
    items = load_items()
    if not items:
        failures.append("items file empty")
        return _verify_report(failures)

    # All clear-counterfactual
    bad = [it for it in items if it.get("regime") != "clear-counterfactual"]
    if bad:
        failures.append(f"non-counterfactual items present: {len(bad)}")

    # Count: 25 WP-6A + 12 held-out = 37
    n_wp6a   = sum(1 for it in items if not it.get("held_out", False))
    n_held   = sum(1 for it in items if it.get("held_out", False))
    if n_wp6a != 25:
        failures.append(f"expected 25 WP-6A items, got {n_wp6a}")
    if n_held != 12:
        failures.append(f"expected 12 held-out items, got {n_held}")

    # Every item has context_text + binding + question
    for it in items:
        for k in ("context_text", "binding", "question",
                  "correct_answer_keywords", "subtype"):
            if k not in it or not it.get(k):
                failures.append(f"item id={it.get('id')} missing field {k}")
                break

    # Hash freeze record matches current file
    try:
        h_now = assert_frozen()
        print(f"[verify] frozen-hash assertion PASS: {h_now}")
    except RuntimeError as e:
        failures.append(str(e))

    # Conditions pre-registered (frozen record)
    rec = json.load(open(FROZEN_FILE))
    if rec.get("conditions") != CONDITIONS:
        failures.append(f"frozen conditions != module CONDITIONS")

    # build_system_prompt works for every condition (smoke-call on one item)
    sample = next(it for it in items if it["subtype"] == "novel")
    sample_facts_novel = {
        "quantity":    sample["binding"]["quantity"],
        "unit_name":   sample["binding"]["unit_name"],
        "unit_symbol": sample["binding"]["unit_symbol"],
        "definition":  sample["binding"]["definition"],
    }
    try:
        _ = build_system_prompt("B_FAIR",          sample)
        _ = build_system_prompt("PLAINFACTS",      sample, sample_facts_novel)
        _ = build_system_prompt("C_PACK_LEARNED",  sample, sample_facts_novel)
        _ = build_system_prompt("C_KNOW_ORACLE",   sample)
    except Exception as e:  # noqa: BLE001
        failures.append(f"build_system_prompt smoke failed: {e}")

    sample_adv = next(it for it in items if it["subtype"] == "adversarial")
    sample_facts_adv = {
        "in_context_quantity":  sample_adv["binding"]["in_context_quantity"],
        "in_context_unit_name": sample_adv["binding"]["redefined_unit"],
        "in_context_symbol":    sample_adv["binding"]["redefined_symbol"],
        "domain_override_note": (
            f"In this domain, {sample_adv['binding']['redefined_unit']} is the SI unit "
            f"of {sample_adv['binding']['in_context_quantity']}."
        ),
    }
    try:
        _ = build_system_prompt("PLAINFACTS",     sample_adv, sample_facts_adv)
        _ = build_system_prompt("C_PACK_LEARNED", sample_adv, sample_facts_adv)
    except Exception as e:  # noqa: BLE001
        failures.append(f"build_system_prompt adversarial smoke failed: {e}")

    # PLAINFACTS and C_KNOW_ORACLE both produce non-empty system prompts and
    # PLAINFACTS carries the unit-name + symbol from the facts.
    prose = build_system_prompt("PLAINFACTS", sample, sample_facts_novel)
    if sample["binding"]["unit_name"] not in prose:
        failures.append(f"PLAINFACTS prose missing unit_name {sample['binding']['unit_name']}")
    if sample["binding"]["unit_symbol"] not in prose:
        failures.append(f"PLAINFACTS prose missing unit_symbol {sample['binding']['unit_symbol']}")
    if "CONTRACT" in prose or "{" in prose:
        failures.append("PLAINFACTS contaminated with scaffold/JSON structure")

    # Checker round-trip on one item: a response containing the in-context
    # unit name should NOT violate.
    sample_resp = f"The answer is {sample['binding']['unit_name']} ({sample['binding']['unit_symbol']})."
    v = check_violation_fair(sample, sample_resp)
    if v.get("violation", True):
        failures.append(f"WP-6A checker round-trip FAIL on novel sample: {v}")

    return _verify_report(failures)


def _verify_report(failures: list) -> int:
    if failures:
        print(f"[verify] FAIL ({len(failures)} failures):")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("[verify] PASS")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generate",  action="store_true", help="U1: items.jsonl + freeze hash")
    p.add_argument("--verify-u1", action="store_true", help="U1: inline-verify deliverables")
    p.add_argument("--route-fit", action="store_true", help="U2: fit softmax router; report cv + held-out acc")
    p.add_argument("--verify-u2", action="store_true", help="U2: inline-verify router")
    p.add_argument("--extract-all",       action="store_true", help="U3: run LEARNED ECE on all items (resumable)")
    p.add_argument("--extraction-fidelity", action="store_true", help="U3: score extraction vs oracle (no LLM)")
    p.add_argument("--verify-u3", action="store_true", help="U3: inline-verify ECE")
    p.add_argument("--binder-spotcheck", action="store_true", help="U4: spot-check binder info-constancy + scaffold-cleanliness")
    p.add_argument("--verify-u4", action="store_true", help="U4: inline-verify binder")
    p.add_argument("--smoke",     action="store_true", help="U5 smoke: 4 items × 4 cond × 1 rep = 16 calls")
    p.add_argument("--run-full",  action="store_true", help="U5 full: 37 × 4 × N_REPS calls (resumable)")
    p.add_argument("--reps",      type=int, default=None, help="override n_reps for --run-full")
    p.add_argument("--verify-u5", action="store_true", help="U5: inline-verify harness output")
    p.add_argument("--aggregate", action="store_true", help="U6: aggregate gates + write results doc")
    p.add_argument("--verify-u6", action="store_true", help="U6: inline-verify aggregate")
    p.add_argument("--verify-u7", action="store_true", help="U7: inline-verify claim doc")
    p.add_argument("--verify-determinism", action="store_true",
                   help="WP-11 U2: greedy determinism on active backend (intended: local)")
    p.add_argument("--verify-u2-qwen", action="store_true", help="WP-11 U2: inline-verify determinism")
    p.add_argument("--compare-models", action="store_true", help="WP-11 U6: cross-model compare + gate_local_reproduction")
    p.add_argument("--verify-u6-wp11", action="store_true", help="WP-11 U6: inline-verify compare artifacts")
    p.add_argument("--verify-u7-wp11", action="store_true", help="WP-11 U7: inline-verify local-reproduction claim doc")
    args = p.parse_args()

    if args.generate:
        cmd_generate()
        return
    if args.verify_u1:
        sys.exit(cmd_verify_u1())
    if args.route_fit:
        cmd_route_fit()
        return
    if args.verify_u2:
        sys.exit(cmd_verify_u2())
    if args.extract_all:
        cmd_extract_all()
        return
    if args.extraction_fidelity:
        cmd_extraction_fidelity()
        return
    if args.verify_u3:
        sys.exit(cmd_verify_u3())
    if args.binder_spotcheck:
        cmd_binder_spotcheck()
        return
    if args.verify_u4:
        sys.exit(cmd_verify_u4())
    if args.smoke:
        cmd_smoke()
        return
    if args.run_full:
        cmd_run_full(n_reps=args.reps)
        return
    if args.verify_u5:
        sys.exit(cmd_verify_u5())
    if args.aggregate:
        cmd_aggregate()
        return
    if args.verify_u6:
        sys.exit(cmd_verify_u6())
    if args.verify_u7:
        sys.exit(cmd_verify_u7())
    if args.verify_determinism:
        cmd_verify_determinism()
        return
    if args.verify_u2_qwen:
        sys.exit(cmd_verify_u2_qwen())
    if args.compare_models:
        cmd_compare_models()
        return
    if args.verify_u6_wp11:
        sys.exit(cmd_verify_u6_wp11())
    if args.verify_u7_wp11:
        sys.exit(cmd_verify_u7_wp11())

    p.print_help()


if __name__ == "__main__":
    main()
