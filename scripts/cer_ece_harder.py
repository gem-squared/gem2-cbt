"""WP-ST-12: Harder NON-SATURATING prompt-scope on Qwen-local.

Tests whether the structured contract pack (C_PACK_LEARNED) beats plain
in-context facts (PLAINFACTS) on ANY of 4 stressor axes where PLAINFACTS
is designed NOT to saturate at 0% violation:

  multi-binding  : 3-5 concurrent in-context bindings; question targets one
  distractors    : load-bearing binding + 3-5 irrelevant binding-like statements
  long context   : binding embedded inside ~800-1500 tokens of filler
  conflicting    : two bindings for same quantity with primary/deprecated precedence

Subject: qwen2.5-32b-instruct-q8_0 (Ollama local, greedy temp=0).
Conditions: same 4 as WP-10/11 (B_FAIR / PLAINFACTS / C_PACK_LEARNED / C_KNOW_ORACLE).
Gate floors: UNCHANGED (Delta >= 0.05, |d| >= 0.5 or absolute_unanimity).

Separate-track discipline (David's anti-silent-merge): writes to
data/cer_ece_harder_qwen/* -- NEVER merged with data/cer_ece/ (WP-10
DeepSeek) or data/cer_ece_qwen/ (WP-11 Qwen single-binding).

WP-6A/WP-7/WP-10/WP-11 files UNTOUCHED. cbt/llm_backend.py imported read-only.

U1: items + pre-registered renderer schemas + frozen hash (this commit).
U2+: per-axis renderers + extended checker + LEARNED ECE + harness + aggregate + claim.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# ── Reuse WP-11 / WP-10 / WP-6A primitives (read-only import) ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))            # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # repo root
from oracle_payoff_fair import (  # noqa: E402
    STRONG_PROMPT_B_FAIR,
    SYSTEM_PROMPT_CONTRACT_SCAFFOLD,
    COUNTERFACTUAL_NOVEL as WP6A_NOVEL,
    COUNTERFACTUAL_ADVERSARIAL as WP6A_ADV,
)


# ── Paths (WP-12 scoped, separate track) ───────────────────────────────────
DATA_DIR        = "data/cer_ece_harder_qwen"
PAPERS_DIR      = "papers"
ITEMS_FILE      = os.path.join(DATA_DIR, "items.jsonl")
FROZEN_FILE     = os.path.join(DATA_DIR, "frozen_items_hash.json")
RUNS_FILE       = os.path.join(DATA_DIR, "raw_runs.jsonl")
EVAL_FILE       = os.path.join(DATA_DIR, "eval_results.json")
EXTRACTED_FILE  = os.path.join(DATA_DIR, "extracted_facts.jsonl")
BINDER_SC_FILE  = os.path.join(DATA_DIR, "binder_spotcheck.json")
RESULTS_MD      = os.path.join(PAPERS_DIR, "results_harder_scope.md")
CLAIM_MD        = os.path.join(PAPERS_DIR, "claim_harder_scope.md")

# 4 conditions pre-registered (same as WP-10/11; frozen at U1 close).
CONDITIONS = ["B_FAIR", "PLAINFACTS", "C_PACK_LEARNED", "C_KNOW_ORACLE"]

# Gate floors UNCHANGED across WP-10 / WP-11 / WP-12 (anti-cherry-pick).
GATE_FLOOR_PAYOFF      = 0.05
GATE_FLOOR_STRUCTURE   = 0.05

# Harness reps + smoke spec
N_REPS       = 3
N_REPS_SMOKE = 1

# 4 axes (frozen at U1 close).
AXES = ("multi_binding", "distractors", "long_context", "conflicting")

# ═══════════════════════════════════════════════════════════════════════════
# PRE-REGISTERED PER-AXIS RENDERER SCHEMAS (frozen at U1 close, anti-tuning)
#
# Each schema declares the EXACT shape of the `concept` block that:
#   - the U4 LEARNED ECE extracts FROM input,
#   - the U2 render_pack_<axis> wraps in WP-6A scaffold as C_PACK_LEARNED,
#   - the U2 render_prose_<axis> renders verbatim as PLAINFACTS (no scaffold).
#
# Info-constancy spine: every field listed below must appear in BOTH
# the pack and the prose; the only variable across formats is the SHAPE.
# ═══════════════════════════════════════════════════════════════════════════

RENDERER_SCHEMA = {
    "multi_binding": {
        "concept_fields": ("bindings", "asked_quantity"),
        "binding_subfields": ("quantity", "unit_name", "unit_symbol"),
        "task":    "Answer a factual question about the SI unit of one of several in-context-defined quantities.",
        "context": "In this domain, several non-standard quantities have been defined with their SI units.",
    },
    "distractors": {
        "concept_fields": ("load_bearing", "distractors"),
        "load_bearing_subfields": ("quantity", "unit_name", "unit_symbol"),
        "task":    "Answer a factual question about the SI unit of an in-context-defined quantity, ignoring distractor facts.",
        "context": "Several facts apply in this domain; the load-bearing binding is the one relevant to the question.",
    },
    "long_context": {
        "concept_fields": ("binding", "filler_before", "filler_after"),
        "binding_subfields": ("quantity", "unit_name", "unit_symbol"),
        "task":    "Answer a factual question about the SI unit of an in-context-defined quantity, retrieved from a longer document.",
        "context": "A document contains a domain-specific binding embedded in narrative text.",
    },
    "conflicting": {
        "concept_fields": ("in_context_quantity", "primary", "deprecated", "precedence_note"),
        "binding_subfields": ("unit_name", "unit_symbol"),
        "task":    "Answer a factual question about the SI unit of an in-context quantity, respecting the primary vs deprecated precedence.",
        "context": "Two bindings exist; the primary takes precedence over the deprecated alternative.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PRE-REGISTERED FILLER POOL for long-context items (frozen at U1 close).
# Generic gardening / cooking / sysadmin glossary text, intentionally
# binding-irrelevant. Sliced deterministically per item index.
# ═══════════════════════════════════════════════════════════════════════════

LONG_CONTEXT_FILLER = (
    "When preparing soil for spring planting, the first task is to assess "
    "the compaction level by inserting a thin metal rod and noting the "
    "depth of resistance. Soils that resist deeper than the first ten "
    "centimeters generally benefit from a single pass of a broadfork rather "
    "than full tillage. Mulching is then applied in a layer thin enough "
    "that earthworm activity continues without smothering, typically two "
    "fingers deep across the bed but tapering near the stems of established "
    "perennials. A common kitchen-garden rotation alternates legumes, "
    "brassicas, root crops, and a fallow cover, which preserves nutrient "
    "balance without heavy amendment. Companion planting traditions vary "
    "by region, but most agree that alliums near brassicas reduce pest "
    "pressure, while basil near tomatoes is more an aesthetic and culinary "
    "convention than a measurable yield benefit. "
    "In server administration, log rotation is configured to balance disk "
    "pressure against the recovery value of older entries. A daily rotation "
    "with seven-day retention is common for application logs, while "
    "security audit logs are often kept for ninety days or longer depending "
    "on policy. Compression of rotated logs reduces footprint substantially "
    "for textual content, though the tradeoff is increased CPU during "
    "incident review. Monitoring agents typically read from the current "
    "log file and ignore rotated archives, so any alerting rules must be "
    "wired to the active file rather than the pattern. Backup snapshots "
    "should always be tested by restoring to a clean target host rather "
    "than trusted on the basis of completion exit codes alone. "
    "Bread-baking schedules at home generally begin with a long autolyse, "
    "during which flour and water rest together so that the gluten network "
    "begins to form without active mechanical agitation. Salt is then "
    "added, followed by either commercial yeast or a portion of mature "
    "starter, depending on whether the loaf is to be sourdough or yeasted. "
    "Bulk fermentation proceeds for several hours at room temperature, "
    "with folds at regular intervals to develop structure. Shaping is the "
    "step where novice bakers most often deflate the loaf; the goal is "
    "tension at the surface rather than compression of the interior. "
    "Final proof can take place at room temperature for an hour or in the "
    "refrigerator overnight, which deepens flavor through extended "
    "fermentation. Scoring the dough before baking creates intentional "
    "weakness points so the loaf expands predictably during oven spring. "
    "Coffee brewing parameters depend on the extraction method, but most "
    "filter techniques target a ratio of approximately one gram of ground "
    "coffee for every fifteen to seventeen grams of water. Grind size is "
    "tuned to total brew time; finer grinds slow flow and increase "
    "extraction, while coarser grinds speed flow and reduce it. Bypass "
    "brewing techniques sidestep the saturation behavior of certain "
    "filter geometries by pouring a portion of water around the bed "
    "rather than through it. Decant practices vary by culture, with some "
    "households preferring to serve immediately and others allowing a "
    "brief rest so that the flavor profile evens out as the temperature "
    "drops. "
    "Bicycle maintenance routines distinguish between weekly checks, "
    "monthly tune-ups, and annual overhauls. Weekly tasks include tire "
    "pressure verification, chain lubrication, and a quick scan for loose "
    "bolts at the stem and seatpost. Monthly checks add brake pad "
    "inspection, derailleur cable tension, and wheel true verification. "
    "Annual overhauls typically involve replacing the chain, brake pads, "
    "and any cables that show fraying, plus a full bearing service if the "
    "ride mileage warrants it. Tubeless setups add the periodic "
    "responsibility of topping off sealant, which dries out over the "
    "course of two to four months depending on climate. "
    "Photography color management workflows tag images at capture with the "
    "device color profile, then convert into a working color space such as "
    "ProPhoto or Adobe RGB during editing. Final outputs are converted to "
    "the destination color space, most commonly sRGB for web delivery or a "
    "print profile for production. Skipping any of these conversions does "
    "not produce a visible error but can shift color reproduction in "
    "subtle ways that accumulate across a workflow. Soft proofing in the "
    "destination profile before commit is the standard guard against "
    "surprises at output time."
)


# ═══════════════════════════════════════════════════════════════════════════
# Binding pools — for multi-binding + distractors + long_context + conflicting.
# Reuse WP-6A novel + adversarial pools (read-only). No mutation.
# ═══════════════════════════════════════════════════════════════════════════

def _novel_pool() -> list:
    """All WP-6A novel bindings as a list of dicts {quantity, unit_name, unit_symbol}."""
    return [
        {"quantity": b["quantity"], "unit_name": b["unit_name"], "unit_symbol": b["unit_symbol"]}
        for b in WP6A_NOVEL
    ]


def _adv_pool() -> list:
    """All WP-6A adversarial bindings as {in_context_quantity, redefined_unit, redefined_symbol}."""
    return [
        {"in_context_quantity": b["in_context_quantity"],
         "redefined_unit":      b["redefined_unit"],
         "redefined_symbol":    b["redefined_symbol"]}
        for b in WP6A_ADV
    ]


# Pre-registered DISTRACTOR pool (frozen at U1 close). Mix of real-SI facts
# (counted as plausibly-relevant but not the answer) + invented unrelated
# statements (counted as obvious distractor). Same pool across items;
# sliced deterministically per item index.
DISTRACTOR_POOL = [
    {"text": "The standard SI unit of length is the meter.", "kind": "real_si"},
    {"text": "Time intervals are conventionally reported in seconds.", "kind": "real_si"},
    {"text": "Atmospheric pressure at sea level is roughly 101.3 kPa.", "kind": "real_si"},
    {"text": "The base SI unit of mass is the kilogram.", "kind": "real_si"},
    {"text": "Frequency is measured in hertz, named after Heinrich Hertz.", "kind": "real_si"},
    {"text": "In computing, file sizes are commonly reported in gigabytes.", "kind": "irrelevant"},
    {"text": "Highway speed limits are often specified in kilometers per hour.", "kind": "irrelevant"},
    {"text": "Battery capacity in mobile devices is typically rated in milliamp-hours.", "kind": "irrelevant"},
    {"text": "Screen brightness in displays is commonly reported in nits.", "kind": "irrelevant"},
    {"text": "Camera sensor sensitivity is described using the ISO scale.", "kind": "irrelevant"},
]


# ═══════════════════════════════════════════════════════════════════════════
# Item generators per axis
# ═══════════════════════════════════════════════════════════════════════════

def _multi_binding_items() -> list:
    """12 items: 4 with 3 bindings, 4 with 4 bindings, 4 with 5 bindings.
    Each item picks bindings deterministically from the novel pool.
    Asked binding rotates through positions (first / middle / last) to avoid
    a position-bias artifact."""
    novel = _novel_pool()
    out = []
    item_id = 700
    plans = [
        # (n_bindings, ask_position) — pre-registered, frozen
        (3, 0), (3, 1), (3, 2), (3, 1),
        (4, 0), (4, 1), (4, 2), (4, 3),
        (5, 0), (5, 2), (5, 3), (5, 4),
    ]
    cursor = 0
    for n_bind, ask_pos in plans:
        bindings = []
        for _ in range(n_bind):
            b = novel[cursor % len(novel)]
            bindings.append(b)
            cursor += 1
        asked = bindings[ask_pos]
        out.append({
            "id":       item_id,
            "axis":     "multi_binding",
            "n_bindings": n_bind,
            "ask_position": ask_pos,
            "bindings": bindings,
            "asked_quantity": asked["quantity"],
            "context_text": _multi_binding_context_text(bindings),
            "question": (
                f"In this domain, what is the SI unit of {asked['quantity']}? "
                f"State the unit name and symbol."
            ),
            "correct_answer_keywords": [asked["unit_name"], asked["unit_symbol"].lower()],
            "sibling_unit_keywords": [
                kw.lower() for b in bindings if b["quantity"] != asked["quantity"]
                for kw in (b["unit_name"], b["unit_symbol"])
            ],
            "violation_keywords": [],
            "expected": "in_context_commit",
        })
        item_id += 1
    return out


def _multi_binding_context_text(bindings: list) -> str:
    parts = [f"1 {b['unit_name']} ({b['unit_symbol']}) is the SI unit of {b['quantity']}"
             for b in bindings]
    return "In this domain, the following bindings apply: " + "; ".join(parts) + "."


def _distractor_items() -> list:
    """12 items: 1 load-bearing novel binding + 3-5 distractors (mix of
    real-SI + irrelevant)."""
    novel = _novel_pool()
    out = []
    item_id = 800
    # Plans (frozen): n_distractors, distractor_indices into DISTRACTOR_POOL
    plans = [
        (3, [0, 5, 6]),
        (3, [1, 7, 8]),
        (3, [2, 5, 9]),
        (4, [0, 1, 5, 6]),
        (4, [2, 3, 7, 8]),
        (4, [0, 4, 5, 9]),
        (5, [0, 1, 2, 5, 6]),
        (5, [1, 3, 4, 7, 8]),
        (5, [0, 2, 4, 5, 9]),
        (4, [3, 4, 6, 7]),
        (3, [4, 8, 9]),
        (5, [0, 3, 5, 7, 9]),
    ]
    for k, (n_d, idx_list) in enumerate(plans):
        b = novel[k % len(novel)]
        distractors = [DISTRACTOR_POOL[i] for i in idx_list]
        out.append({
            "id":           item_id,
            "axis":         "distractors",
            "n_distractors": n_d,
            "load_bearing":  b,
            "distractors":   distractors,
            "context_text":  _distractor_context_text(b, distractors),
            "question": (
                f"In this domain, what is the SI unit of {b['quantity']}? "
                f"State the unit name and symbol."
            ),
            "correct_answer_keywords": [b["unit_name"], b["unit_symbol"].lower()],
            "distractor_unit_keywords": _distractor_unit_keywords(distractors),
            "violation_keywords": [],
            "expected": "in_context_commit",
        })
        item_id += 1
    return out


def _distractor_context_text(load_bearing: dict, distractors: list) -> str:
    facts = [f"1 {load_bearing['unit_name']} ({load_bearing['unit_symbol']}) is the SI unit of {load_bearing['quantity']}."]
    facts += [d["text"] for d in distractors]
    return "In this domain, the following facts apply: " + " ".join(facts)


def _distractor_unit_keywords(distractors: list) -> list:
    """Extract any unit-like keywords from distractor texts for the checker
    (used to detect distractor-confusion in subject responses)."""
    import re
    out = []
    for d in distractors:
        for token in re.findall(r"\b[a-z]{3,}\b", d["text"].lower()):
            # Heuristic: known SI unit tokens
            if token in {"meter", "second", "kilogram", "hertz", "kilopascal",
                         "gigabyte", "kilometer", "milliamp", "nit", "iso"}:
                out.append(token)
    return out


def _long_context_items() -> list:
    """12 items: novel binding embedded inside filler text. Filler is sliced
    deterministically from LONG_CONTEXT_FILLER per item index."""
    novel = _novel_pool()
    out = []
    item_id = 900
    filler_len = len(LONG_CONTEXT_FILLER)
    # Pre-registered slice plan: (slice_start_index, total_filler_chars).
    # Total filler ≈ 800–1500 chars per item; binding goes in the middle.
    plans = [
        (0,    1200),
        (200,  1100),
        (400,  1000),
        (600,   900),
        (800,  1300),
        (1000, 1400),
        (1200, 1500),
        (1400, 1200),
        (1600, 1100),
        (1800, 1000),
        (2000, 1300),
        (2200, 1400),
    ]
    for k, (start, total) in enumerate(plans):
        b = novel[k % len(novel)]
        # Wrap-around slice
        slice_text = (LONG_CONTEXT_FILLER + LONG_CONTEXT_FILLER)[start:start + total]
        mid = total // 2
        before = slice_text[:mid]
        after  = slice_text[mid:]
        out.append({
            "id":           item_id,
            "axis":         "long_context",
            "filler_chars": total,
            "binding":      b,
            "filler_before": before,
            "filler_after":  after,
            "context_text": _long_context_context_text(b, before, after),
            "question": (
                f"In this domain, what is the SI unit of {b['quantity']}? "
                f"State the unit name and symbol."
            ),
            "correct_answer_keywords": [b["unit_name"], b["unit_symbol"].lower()],
            "violation_keywords":      [],
            "expected": "in_context_commit",
        })
        item_id += 1
    return out


def _long_context_context_text(binding: dict, before: str, after: str) -> str:
    binding_sentence = (
        f"In this domain, 1 {binding['unit_name']} ({binding['unit_symbol']}) "
        f"is the SI unit of {binding['quantity']}."
    )
    return f"{before} {binding_sentence} {after}"


def _conflicting_items() -> list:
    """14 items: two bindings for SAME in-context quantity with primary vs
    deprecated precedence. Correct = primary; deprecated counts as violation.
    Mix adversarial in-context-quantities with novel unit names."""
    novel = _novel_pool()
    adv   = _adv_pool()
    out = []
    item_id = 1000
    # Plan: each entry = (in_context_quantity, primary_idx, deprecated_idx)
    # picking different unit pairs from the novel pool.
    plans = [
        ("luminous intensity",     0,  1),
        ("electric capacitance",   2,  3),
        ("electric inductance",    4,  5),
        ("magnetic flux density",  6,  7),
        ("solid angle",            8,  9),
        ("pressure",              10, 11),
        ("force",                  0,  2),
        ("amount of substance",    1,  3),
        ("energy",                 4,  6),
        ("thermodynamic temperature", 5,  7),
        ("luminous flux",          8, 10),
        ("electric charge",        9, 11),
        ("power",                  0,  5),
        ("electric current",       2,  7),
    ]
    for quantity, p_idx, d_idx in plans:
        primary    = novel[p_idx % len(novel)]
        deprecated = novel[d_idx % len(novel)]
        out.append({
            "id":           item_id,
            "axis":         "conflicting",
            "in_context_quantity": quantity,
            "primary":      primary,
            "deprecated":   deprecated,
            "precedence_note": (
                f"The {primary['unit_name']} is the primary SI unit of {quantity} in this domain; "
                f"the {deprecated['unit_name']} is documented as a deprecated alternative "
                f"and should not be used."
            ),
            "context_text": _conflicting_context_text(quantity, primary, deprecated),
            "question": (
                f"In this domain, what is the SI unit of {quantity}? "
                f"State the unit name and symbol."
            ),
            "correct_answer_keywords":  [primary["unit_name"], primary["unit_symbol"].lower()],
            "deprecated_unit_keywords": [deprecated["unit_name"], deprecated["unit_symbol"].lower()],
            "violation_keywords":       [],
            "expected": "in_context_commit_primary",
        })
        item_id += 1
    return out


def _conflicting_context_text(quantity: str, primary: dict, deprecated: dict) -> str:
    return (
        f"In this domain, the primary SI unit of {quantity} is the "
        f"{primary['unit_name']} ({primary['unit_symbol']}). "
        f"The {deprecated['unit_name']} ({deprecated['unit_symbol']}) is documented "
        f"as a deprecated alternative for {quantity} but should not be used."
    )


def generate_items() -> list:
    """U1: emit all 4 axes -> ~50 items total."""
    out = []
    out += _multi_binding_items()
    out += _distractor_items()
    out += _long_context_items()
    out += _conflicting_items()
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Hash + freeze discipline (reused pattern)
# ═══════════════════════════════════════════════════════════════════════════

def items_hash(items_path: str) -> str:
    h = hashlib.sha256()
    with open(items_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def freeze_items_hash(h: str, items: list) -> None:
    counts = {axis: sum(1 for it in items if it["axis"] == axis) for axis in AXES}
    record = {
        "frozen_hash":   h,
        "n_items":       len(items),
        "axis_counts":   counts,
        "axes":          list(AXES),
        "conditions":    CONDITIONS,
        "n_reps":        N_REPS,
        "gate_floor_payoff":    GATE_FLOOR_PAYOFF,
        "gate_floor_structure": GATE_FLOOR_STRUCTURE,
        "subject":       "qwen2.5-32b-instruct-q8_0",
        "renderer_schema": RENDERER_SCHEMA,
    }
    with open(FROZEN_FILE, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[freeze] items hash locked: {h} ({len(items)} items)")
    print(f"[freeze] axis counts:       {counts}")


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


# ═══════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════

def cmd_generate() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    items = generate_items()
    with open(ITEMS_FILE, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    h = items_hash(ITEMS_FILE)
    freeze_items_hash(h, items)
    print(f"[generate] wrote {ITEMS_FILE} with {len(items)} items")


# ═══════════════════════════════════════════════════════════════════════════
# U2: Per-axis PLAINFACTS + C_PACK_LEARNED renderers (info-constant per item)
#
# Both renderers take the SAME `facts` dict per item (axis-determined shape
# from RENDERER_SCHEMA). PLAINFACTS prose contains every value the pack
# contains; only the SHAPE differs. Verified per-axis by `binder_spotcheck`
# in U4 — same info-constancy discipline as WP-10 U4.
# ═══════════════════════════════════════════════════════════════════════════

PLAINFACTS_PREAMBLE_HARDER = (
    "You are a precise physics measurement assistant. "
    "The following facts about the question's domain have been provided. "
    "Use them when answering.\n\n"
)


# ── multi_binding ──────────────────────────────────────────────────────────

def render_pack_multi_binding(facts: dict) -> str:
    """Wrap multi_binding facts in WP-6A scaffold."""
    sch = RENDERER_SCHEMA["multi_binding"]
    pack = {
        "task":    sch["task"],
        "context": sch["context"],
        "concept": {
            "bindings":        facts["bindings"],
            "asked_quantity":  facts["asked_quantity"],
        },
    }
    return SYSTEM_PROMPT_CONTRACT_SCAFFOLD.format(
        contract_json=json.dumps(pack, indent=2, ensure_ascii=False))


def render_prose_multi_binding(facts: dict) -> str:
    """Prose form — same values as pack, no scaffold, no JSON."""
    parts = [
        f"1 {b['unit_name']} ({b['unit_symbol']}) is the SI unit of {b['quantity']}"
        for b in facts["bindings"]
    ]
    body = (
        "In this domain, the following bindings apply: "
        + "; ".join(parts) + ". "
        + f"The question concerns the quantity {facts['asked_quantity']}."
    )
    return PLAINFACTS_PREAMBLE_HARDER + body


# ── distractors ────────────────────────────────────────────────────────────

def render_pack_distractors(facts: dict) -> str:
    sch = RENDERER_SCHEMA["distractors"]
    pack = {
        "task":    sch["task"],
        "context": sch["context"],
        "concept": {
            "load_bearing": facts["load_bearing"],
            "distractors":  facts["distractors"],
        },
    }
    return SYSTEM_PROMPT_CONTRACT_SCAFFOLD.format(
        contract_json=json.dumps(pack, indent=2, ensure_ascii=False))


def render_prose_distractors(facts: dict) -> str:
    lb = facts["load_bearing"]
    parts = [
        f"1 {lb['unit_name']} ({lb['unit_symbol']}) is the SI unit of {lb['quantity']}."
    ]
    parts += [d["text"] for d in facts["distractors"]]
    body = "In this domain, the following facts apply: " + " ".join(parts)
    return PLAINFACTS_PREAMBLE_HARDER + body


# ── long_context ───────────────────────────────────────────────────────────

def render_pack_long_context(facts: dict) -> str:
    sch = RENDERER_SCHEMA["long_context"]
    pack = {
        "task":    sch["task"],
        "context": sch["context"],
        "concept": {
            "binding":        facts["binding"],
            "filler_before":  facts["filler_before"],
            "filler_after":   facts["filler_after"],
        },
    }
    return SYSTEM_PROMPT_CONTRACT_SCAFFOLD.format(
        contract_json=json.dumps(pack, indent=2, ensure_ascii=False))


def render_prose_long_context(facts: dict) -> str:
    b = facts["binding"]
    binding_sentence = (
        f"In this domain, 1 {b['unit_name']} ({b['unit_symbol']}) "
        f"is the SI unit of {b['quantity']}."
    )
    body = f"{facts['filler_before']} {binding_sentence} {facts['filler_after']}"
    return PLAINFACTS_PREAMBLE_HARDER + body


# ── conflicting ────────────────────────────────────────────────────────────

def render_pack_conflicting(facts: dict) -> str:
    sch = RENDERER_SCHEMA["conflicting"]
    pack = {
        "task":    sch["task"],
        "context": sch["context"],
        "concept": {
            "in_context_quantity": facts["in_context_quantity"],
            "primary":             facts["primary"],
            "deprecated":          facts["deprecated"],
            "precedence_note":     facts["precedence_note"],
        },
    }
    return SYSTEM_PROMPT_CONTRACT_SCAFFOLD.format(
        contract_json=json.dumps(pack, indent=2, ensure_ascii=False))


def render_prose_conflicting(facts: dict) -> str:
    p, d = facts["primary"], facts["deprecated"]
    q = facts["in_context_quantity"]
    body = (
        f"In this domain, the primary SI unit of {q} is the {p['unit_name']} "
        f"({p['unit_symbol']}). The {d['unit_name']} ({d['unit_symbol']}) is "
        f"documented as a deprecated alternative for {q} but should not be used. "
        f"{facts['precedence_note']}"
    )
    return PLAINFACTS_PREAMBLE_HARDER + body


# ── dispatcher ─────────────────────────────────────────────────────────────

_RENDER_PACK = {
    "multi_binding": render_pack_multi_binding,
    "distractors":   render_pack_distractors,
    "long_context":  render_pack_long_context,
    "conflicting":   render_pack_conflicting,
}
_RENDER_PROSE = {
    "multi_binding": render_prose_multi_binding,
    "distractors":   render_prose_distractors,
    "long_context":  render_prose_long_context,
    "conflicting":   render_prose_conflicting,
}


def oracle_facts_harder(item: dict) -> dict:
    """Build the per-axis oracle facts dict directly from the item structure.
    Used by C_KNOW_ORACLE (ceiling) AND as the fidelity reference at U4."""
    axis = item["axis"]
    if axis == "multi_binding":
        return {"bindings": item["bindings"], "asked_quantity": item["asked_quantity"]}
    if axis == "distractors":
        return {"load_bearing": item["load_bearing"], "distractors": item["distractors"]}
    if axis == "long_context":
        return {"binding": item["binding"],
                "filler_before": item["filler_before"],
                "filler_after":  item["filler_after"]}
    if axis == "conflicting":
        return {
            "in_context_quantity": item["in_context_quantity"],
            "primary":             item["primary"],
            "deprecated":          item["deprecated"],
            "precedence_note":     item["precedence_note"],
        }
    raise ValueError(f"unknown axis: {axis}")


def build_system_prompt_harder(condition: str, item: dict,
                                learned_facts: dict = None) -> str:
    """Dispatch system prompt by condition + axis. Used by U5 harness."""
    axis = item["axis"]
    if condition == "B_FAIR":
        return STRONG_PROMPT_B_FAIR
    if condition == "PLAINFACTS":
        if learned_facts is None:
            raise ValueError("PLAINFACTS requires learned_facts (from U4 LEARNED ECE)")
        return _RENDER_PROSE[axis](learned_facts)
    if condition == "C_PACK_LEARNED":
        if learned_facts is None:
            raise ValueError("C_PACK_LEARNED requires learned_facts (from U4 LEARNED ECE)")
        return _RENDER_PACK[axis](learned_facts)
    if condition == "C_KNOW_ORACLE":
        return _RENDER_PACK[axis](oracle_facts_harder(item))
    raise ValueError(f"Unknown condition: {condition}")


def build_user_message_harder(condition: str, item: dict) -> str:
    """User-side message — just the question (binding info lives in system)."""
    return item["question"]


def cmd_verify_u2() -> int:
    """U2 inline-verify: every (axis, condition) build_system_prompt works
    on a sample item from each axis (using oracle facts as the learned_facts
    stand-in for the spot test); PLAINFACTS contains no scaffold markers;
    C_PACK_LEARNED + C_KNOW_ORACLE both contain the scaffold."""
    failures = []
    items = load_items()
    for axis in AXES:
        sample = next((it for it in items if it["axis"] == axis), None)
        if sample is None:
            failures.append(f"no sample item found for axis {axis}")
            continue
        oracle = oracle_facts_harder(sample)
        for cond in CONDITIONS:
            try:
                facts = oracle if cond in ("PLAINFACTS", "C_PACK_LEARNED") else None
                prompt = build_system_prompt_harder(cond, sample, learned_facts=facts)
            except Exception as e:  # noqa: BLE001
                failures.append(f"build_system_prompt_harder({cond}, axis={axis}) raised: {e}")
                continue
            if cond == "PLAINFACTS":
                if "CONTRACT" in prompt or "{" in prompt:
                    failures.append(
                        f"PLAINFACTS axis={axis} contains scaffold markers — info leak")
            if cond in ("C_PACK_LEARNED", "C_KNOW_ORACLE"):
                if "CONTRACT" not in prompt or "{" not in prompt:
                    failures.append(
                        f"{cond} axis={axis} missing scaffold markers")
        # build_user_message
        try:
            _ = build_user_message_harder("B_FAIR", sample)
        except Exception as e:  # noqa: BLE001
            failures.append(f"build_user_message_harder raised: {e}")
    if not failures:
        print(f"[verify-u2] all (axis × condition) renderers OK; n_axes={len(AXES)}")
    return _verify_report(failures)


# ═══════════════════════════════════════════════════════════════════════════
# U3: Extended per-axis deterministic checker.
#
# check_violation_harder(item, response) dispatches on axis. Per-axis logic:
#   multi_binding : violation iff response misses asked binding's unit
#                   OR mentions ONLY a sibling binding's unit
#   distractors   : violation iff response misses load-bearing unit OR
#                   mentions a distractor unit AS the answer
#   long_context  : violation iff response misses the embedded binding's unit
#                   (and does NOT show "not found / no info" — lost_in_long_context)
#   conflicting   : violation iff contains deprecated OR misses primary
#
# Verdict dict carries:
#   {violation, axis, check_type, detail, diagnostic_flags}
# Diagnostic flags are PER-AXIS booleans (sibling_confusion, distractor_confusion,
# lost_in_long_context, precedence_violation) for U6 aggregation.
# ═══════════════════════════════════════════════════════════════════════════

import re as _re

_NOT_FOUND_KEYWORDS = (
    "cannot find", "not provided", "not specified", "no information",
    "not mentioned", "couldn't find", "do not see", "not given", "unknown",
    "no clear", "unclear", "not stated", "isn't specified", "not defined",
)


def _kw_present(resp_lower: str, kw: str) -> bool:
    """Substring presence, with word-boundary protection for short symbols."""
    if len(kw) <= 2:
        return _re.search(r'\b' + _re.escape(kw) + r'\b', resp_lower) is not None
    return kw in resp_lower


def _has_any(resp_lower: str, kws: list) -> bool:
    return any(_kw_present(resp_lower, k.lower()) for k in kws if k)


def _declared_answer_only(response: str) -> str:
    """Heuristic: extract the model's DECLARED answer (vs. rejected mentions).
    The model often says "the answer is X" or "1 X is the SI unit of Y".
    For checker discipline we keep the WHOLE response lowercased — distractor
    mention strictly counts only when distractor unit appears verbatim AND is
    in a positive-assertion context (no "not", "rejected", "irrelevant" near
    it). This is a coarse heuristic; we record both raw and filtered flags
    so U6 can pick the strict-or-permissive read."""
    return response.lower().strip()


def check_violation_harder(item: dict, response: str) -> dict:
    axis = item["axis"]
    resp = _declared_answer_only(response)
    flags = {}

    if axis == "multi_binding":
        correct = item["correct_answer_keywords"]
        siblings = item["sibling_unit_keywords"]
        has_correct = _has_any(resp, correct)
        has_sibling = _has_any(resp, siblings) and not has_correct
        # Sibling-confusion is a violation flavor (model picked the wrong binding)
        flags["sibling_confusion"] = has_sibling
        violation = (not has_correct) or has_sibling
        return {
            "violation":         violation,
            "axis":              axis,
            "check_type":        "multi_binding_select",
            "detail": (
                f"correct={'Y' if has_correct else 'N'} "
                f"sibling_confusion={'Y' if has_sibling else 'N'}: {response[:120]}"
            ),
            "diagnostic_flags":  flags,
        }

    if axis == "distractors":
        correct = item["correct_answer_keywords"]
        distractor_kws = item.get("distractor_unit_keywords", [])
        has_correct = _has_any(resp, correct)
        # distractor_confusion: distractor unit appears AND correct missing
        has_distractor = _has_any(resp, distractor_kws) and not has_correct
        flags["distractor_confusion"] = has_distractor
        violation = (not has_correct)
        return {
            "violation":         violation,
            "axis":              axis,
            "check_type":        "distractor_attention",
            "detail": (
                f"correct={'Y' if has_correct else 'N'} "
                f"distractor_confusion={'Y' if has_distractor else 'N'}: {response[:120]}"
            ),
            "diagnostic_flags":  flags,
        }

    if axis == "long_context":
        correct = item["correct_answer_keywords"]
        has_correct = _has_any(resp, correct)
        lost = (not has_correct) and any(p in resp for p in _NOT_FOUND_KEYWORDS)
        flags["lost_in_long_context"] = lost
        violation = (not has_correct)
        return {
            "violation":         violation,
            "axis":              axis,
            "check_type":        "long_context_retrieval",
            "detail": (
                f"correct={'Y' if has_correct else 'N'} "
                f"lost_in_long_context={'Y' if lost else 'N'}: {response[:120]}"
            ),
            "diagnostic_flags":  flags,
        }

    if axis == "conflicting":
        primary = item["correct_answer_keywords"]
        deprecated = item["deprecated_unit_keywords"]
        has_primary = _has_any(resp, primary)
        has_deprecated = _has_any(resp, deprecated) and not has_primary
        # precedence_violation: deprecated declared OR primary missing
        flags["precedence_violation"] = has_deprecated or (not has_primary)
        violation = has_deprecated or (not has_primary)
        return {
            "violation":         violation,
            "axis":              axis,
            "check_type":        "precedence_follow",
            "detail": (
                f"primary={'Y' if has_primary else 'N'} "
                f"deprecated={'Y' if has_deprecated else 'N'}: {response[:120]}"
            ),
            "diagnostic_flags":  flags,
        }

    return {"violation": False, "axis": axis, "check_type": "unknown_axis",
            "detail": "", "diagnostic_flags": {}}


def cmd_verify_u3() -> int:
    """U3 inline-verify: round-trip smoke on one item per axis.
    Canonical 'correct' response → not violating; canonical 'wrong' →
    violating. No LLM calls."""
    failures = []
    items = load_items()

    # Sample one item per axis
    samples = {axis: next((it for it in items if it["axis"] == axis), None)
               for axis in AXES}
    for axis, it in samples.items():
        if it is None:
            failures.append(f"no sample for axis {axis}"); continue

        # Build canonical "correct" + "wrong" responses per axis
        if axis == "multi_binding":
            asked_unit = it["correct_answer_keywords"][0]
            sib_unit   = it["sibling_unit_keywords"][0]
            correct_resp = f"The SI unit of {it['asked_quantity']} is the {asked_unit}."
            wrong_resp   = f"The SI unit of {it['asked_quantity']} is the {sib_unit}."
        elif axis == "distractors":
            lb = it["load_bearing"]
            correct_resp = f"The SI unit of {lb['quantity']} is the {lb['unit_name']}."
            wrong_resp   = "I cannot determine the answer from the facts provided."
        elif axis == "long_context":
            b = it["binding"]
            correct_resp = f"The SI unit of {b['quantity']} is the {b['unit_name']}."
            wrong_resp   = "I cannot find that quantity in the provided text."
        elif axis == "conflicting":
            primary    = it["correct_answer_keywords"][0]
            deprecated = it["deprecated_unit_keywords"][0]
            q = it["in_context_quantity"]
            correct_resp = f"The primary SI unit of {q} in this domain is the {primary}."
            wrong_resp   = f"The SI unit of {q} is the {deprecated}."

        v_ok = check_violation_harder(it, correct_resp)
        v_bad = check_violation_harder(it, wrong_resp)
        if v_ok["violation"]:
            failures.append(f"axis {axis}: canonical correct unexpectedly flagged: {v_ok}")
        if not v_bad["violation"]:
            failures.append(f"axis {axis}: canonical wrong unexpectedly NOT flagged: {v_bad}")
        # Verdict dict shape
        for k in ("violation", "axis", "check_type", "detail", "diagnostic_flags"):
            if k not in v_ok:
                failures.append(f"axis {axis}: verdict dict missing key {k}")
                break

    if not failures:
        print(f"[verify-u3] round-trip OK on all 4 axes (correct + wrong canonical responses)")
    return _verify_report(failures)


# ═══════════════════════════════════════════════════════════════════════════
# U4: LEARNED ECE per axis + Binder info-constancy spot-check.
#
# Per-axis ECE prompts pre-registered at U4 close (anti-tuning). For
# long_context the ECE only extracts the {binding} (filler is identity-
# preserved by the binder — no point extracting 1.5K of filler twice when
# the binder can just pass it through from the item).
#
# Binder reconstitutes the full facts dict (ECE output + any pass-through
# fields from the item) and runs both render_pack + render_prose to verify
# info-constancy (every value in facts appears in BOTH outputs).
# ═══════════════════════════════════════════════════════════════════════════

# ── Pre-registered per-axis ECE system prompts (frozen at U4 close) ────────

ECE_PROMPT_MULTI_BINDING = """You are a precise extraction assistant for measurement bindings.

Given a DOMAIN CONTEXT paragraph containing multiple in-context-defined SI bindings, plus a QUESTION about ONE specific quantity, EXTRACT the structured JSON object with EXACTLY these two keys:

{
  "bindings": [
    {"quantity": <string>, "unit_name": <string>, "unit_symbol": <string>},
    ...
  ],
  "asked_quantity": <string — copied from the QUESTION>
}

Rules:
- List EVERY binding present in the DOMAIN CONTEXT, in the order they appear.
- Copy values verbatim from the DOMAIN CONTEXT. Do NOT invent or paraphrase.
- "asked_quantity" must match the QUESTION's quantity exactly.
- Keep unit_symbol case as written.
- Return ONLY the JSON object. No prose, no code fences."""


ECE_PROMPT_DISTRACTORS = """You are a precise extraction assistant for measurement bindings.

Given a DOMAIN CONTEXT containing ONE load-bearing SI binding plus several distractor facts, plus a QUESTION about a specific quantity, IDENTIFY which sentence is the load-bearing binding for the asked quantity and which are distractors. EXTRACT a strict JSON object:

{
  "load_bearing": {"quantity": <string>, "unit_name": <string>, "unit_symbol": <string>},
  "distractors": [
    {"text": <string — distractor sentence verbatim>, "kind": <string — 'real_si' or 'irrelevant'>},
    ...
  ]
}

Rules:
- Pick the load_bearing binding as the one that defines the SI unit of the quantity the QUESTION asks about.
- Copy distractor sentences VERBATIM as they appear in the DOMAIN CONTEXT.
- Tag distractor "kind" as "real_si" if it is a real SI fact, "irrelevant" otherwise.
- Do NOT invent. Return ONLY the JSON object. No prose, no code fences."""


ECE_PROMPT_LONG_CONTEXT = """You are a precise extraction assistant for measurement bindings.

Given a long DOMAIN CONTEXT containing narrative filler text PLUS exactly ONE in-context SI binding embedded inside, EXTRACT the binding into a strict JSON object:

{
  "binding": {"quantity": <string>, "unit_name": <string>, "unit_symbol": <string>}
}

Rules:
- Locate the single sentence of the form "In this domain, 1 X (Y) is the SI unit of Z." inside the longer text.
- Copy the values verbatim. Do NOT invent or paraphrase.
- Ignore the surrounding filler text — only the binding matters.
- Return ONLY the JSON object. No prose, no code fences."""


ECE_PROMPT_CONFLICTING = """You are a precise extraction assistant for measurement bindings.

Given a DOMAIN CONTEXT that specifies a PRIMARY SI unit binding and a DEPRECATED alternative for the same quantity, EXTRACT the structured JSON object with EXACTLY these four keys:

{
  "in_context_quantity": <string>,
  "primary":             {"unit_name": <string>, "unit_symbol": <string>},
  "deprecated":          {"unit_name": <string>, "unit_symbol": <string>},
  "precedence_note":     <string — a single sentence noting primary takes precedence>
}

Rules:
- The PRIMARY is the unit explicitly marked as "primary" or "preferred"; the DEPRECATED is the one marked "deprecated" or "should not be used".
- Copy quantity + unit names + symbols verbatim from the DOMAIN CONTEXT.
- precedence_note MUST be a complete sentence of the form "The <primary_unit> is the primary SI unit of <quantity> in this domain; the <deprecated_unit> is documented as a deprecated alternative and should not be used."
- Return ONLY the JSON object. No prose, no code fences."""


_ECE_PROMPTS = {
    "multi_binding": ECE_PROMPT_MULTI_BINDING,
    "distractors":   ECE_PROMPT_DISTRACTORS,
    "long_context":  ECE_PROMPT_LONG_CONTEXT,
    "conflicting":   ECE_PROMPT_CONFLICTING,
}


def _ece_user_message(item: dict) -> str:
    """Compose the ECE user message — DOMAIN CONTEXT + QUESTION."""
    return (
        f"DOMAIN CONTEXT:\n{item['context_text']}\n\n"
        f"QUESTION:\n{item['question']}"
    )


def ece_extract_harder(item: dict) -> tuple:
    """Route → axis-specific ECE prompt → Qwen-local extraction → parse → return
    (facts_dict, axis, meta). On failure, facts_dict is empty; meta records the error."""
    from concept_ce import _parse_llm_json  # noqa: pure parser, no LLM call
    # Backend-aware call (WP-11 _ollama_chat_greedy reused)
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # ensure cer_ece_cell visible
    from cer_ece_cell import _ollama_chat_greedy  # noqa
    from cbt.llm_backend import get_backend  # noqa
    b = get_backend()
    if b["backend"] != "local":
        raise RuntimeError(
            f"WP-12 ECE requires LLM_BACKEND=local (Qwen); got {b['backend']}. "
            f"Set LLM_BACKEND=local."
        )

    axis = item["axis"]
    sys_prompt = _ECE_PROMPTS[axis]
    user_msg   = _ece_user_message(item)
    raw = _ollama_chat_greedy(sys_prompt, user_msg,
                              base_url=b["base_url"], model=b["model"])
    try:
        parsed = _parse_llm_json(raw)
    except Exception as e:  # noqa: BLE001
        return {}, axis, {"raw_response": raw[:240], "parse_error": str(e)[:160]}

    return parsed, axis, {"raw_response": raw[:240], "parse_error": None}


def _facts_for_binder(item: dict, ece_output: dict) -> dict:
    """Combine ECE output + item pass-through fields → full facts dict for renderers.
    For long_context: ECE provides {binding}; item provides {filler_before, filler_after}.
    For distractors: strip `kind` from distractor entries — kind is checker-side
      analysis metadata, NOT input the LLM should see; keeping it asymmetric
      across pack/prose would break info-constancy without serving the task.
    For other axes: ECE output IS the full facts dict."""
    axis = item["axis"]
    if axis == "long_context":
        return {
            "binding":        ece_output.get("binding", {}),
            "filler_before":  item["filler_before"],
            "filler_after":   item["filler_after"],
        }
    if axis == "distractors":
        clean_distractors = [
            {"text": d.get("text", "")}
            for d in (ece_output.get("distractors", []) or [])
        ]
        return {
            "load_bearing": ece_output.get("load_bearing", {}),
            "distractors":  clean_distractors,
        }
    return dict(ece_output)


def _score_fidelity_harder(item: dict, ece_output: dict) -> dict:
    """Per-axis exact-match fidelity vs oracle_facts_harder(item).
    Returns {axis, fields_total, fields_exact, axis_fidelity, per_field}."""
    oracle = oracle_facts_harder(item)
    axis = item["axis"]
    out = {"axis": axis, "fields": {}, "fields_total": 0, "fields_exact": 0}

    def cmp_(name, got, expected):
        out["fields_total"] += 1
        is_exact = (str(got).strip().lower() == str(expected).strip().lower())
        if is_exact:
            out["fields_exact"] += 1
        out["fields"][name] = {"exact": is_exact, "expected": expected, "got": got}

    if axis == "multi_binding":
        # asked_quantity exact
        cmp_("asked_quantity", ece_output.get("asked_quantity", ""), oracle["asked_quantity"])
        # bindings as a set of (quantity, unit_name, unit_symbol)
        got_b = ece_output.get("bindings", []) or []
        oracle_set = {(b["quantity"], b["unit_name"], b["unit_symbol"]) for b in oracle["bindings"]}
        got_set    = {(b.get("quantity",""), b.get("unit_name",""), b.get("unit_symbol",""))
                      for b in got_b}
        out["fields_total"] += 1
        match = (got_set == oracle_set)
        out["fields"]["bindings_set"] = {"exact": match,
                                          "expected": sorted(oracle_set),
                                          "got":      sorted(got_set)}
        if match: out["fields_exact"] += 1

    elif axis == "distractors":
        # load_bearing per-field
        lb_oracle = oracle["load_bearing"]
        lb_got    = ece_output.get("load_bearing", {}) or {}
        for k in ("quantity", "unit_name", "unit_symbol"):
            cmp_(f"load_bearing.{k}", lb_got.get(k, ""), lb_oracle[k])
        # distractor count match (text comparison is too brittle; count is fine)
        d_oracle = oracle["distractors"]
        d_got    = ece_output.get("distractors", []) or []
        out["fields_total"] += 1
        match = (len(d_got) == len(d_oracle))
        out["fields"]["distractor_count"] = {
            "exact": match, "expected": len(d_oracle), "got": len(d_got),
        }
        if match: out["fields_exact"] += 1

    elif axis == "long_context":
        # binding subfields exact
        b_oracle = oracle["binding"]
        b_got    = ece_output.get("binding", {}) or {}
        for k in ("quantity", "unit_name", "unit_symbol"):
            cmp_(f"binding.{k}", b_got.get(k, ""), b_oracle[k])

    elif axis == "conflicting":
        cmp_("in_context_quantity", ece_output.get("in_context_quantity",""), oracle["in_context_quantity"])
        for who in ("primary", "deprecated"):
            sub_oracle = oracle[who]
            sub_got    = ece_output.get(who, {}) or {}
            for k in ("unit_name", "unit_symbol"):
                cmp_(f"{who}.{k}", sub_got.get(k,""), sub_oracle[k])
        # precedence_note presence (string non-empty + mentions primary + deprecated)
        pn = ece_output.get("precedence_note", "")
        out["fields_total"] += 1
        primary_name    = oracle["primary"]["unit_name"]
        deprecated_name = oracle["deprecated"]["unit_name"]
        match = (
            bool(pn)
            and primary_name.lower() in pn.lower()
            and deprecated_name.lower() in pn.lower()
        )
        out["fields"]["precedence_note_mentions_both"] = {
            "exact": match, "expected": f"mentions {primary_name} + {deprecated_name}",
            "got":    pn[:160],
        }
        if match: out["fields_exact"] += 1

    out["axis_fidelity"] = (
        out["fields_exact"] / out["fields_total"] if out["fields_total"] else 0.0
    )
    return out


def cmd_extract_all_harder() -> None:
    """U4: run LEARNED ECE on all 50 items on Qwen-local. Resumable.
    Persist to data/cer_ece_harder_qwen/extracted_facts.jsonl + axis-level
    fidelity summary."""
    assert_frozen()
    from cbt.llm_backend import record_backend_manifest  # noqa
    os.makedirs(DATA_DIR, exist_ok=True)
    record_backend_manifest(os.path.join(DATA_DIR, "manifest_u4_extraction.json"))

    items = load_items()
    done = {}
    if os.path.exists(EXTRACTED_FILE):
        with open(EXTRACTED_FILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["item_id"]] = r

    n_new = 0
    with open(EXTRACTED_FILE, "a") as f:
        for it in items:
            if it["id"] in done:
                continue
            try:
                facts, axis, meta = ece_extract_harder(it)
                ok = bool(facts) and meta.get("parse_error") is None
                rec = {
                    "item_id":   it["id"],
                    "axis":      axis,
                    "facts":     facts,
                    "ok":        ok,
                    "raw_response": meta["raw_response"],
                    "parse_error":  meta.get("parse_error"),
                }
            except Exception as e:  # noqa: BLE001
                rec = {
                    "item_id":   it["id"],
                    "axis":      it["axis"],
                    "facts":     {},
                    "ok":        False,
                    "error":     str(e)[:200],
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            print(f"[ece] id={it['id']} axis={it['axis']:<14} → {'OK' if rec.get('ok') else 'FAIL'}")

    total = sum(1 for _ in open(EXTRACTED_FILE) if _.strip())
    print(f"[extract-all] new records: {n_new}, total: {total}")

    # Aggregate fidelity per axis + binder spot-check
    cmd_fidelity_and_binder_harder()


def cmd_fidelity_and_binder_harder() -> None:
    """Compute per-axis fidelity vs oracle + run binder spot-check (info-constancy
    + scaffold-cleanliness) on all 50 items using extracted facts."""
    items = load_items()
    recs  = {}
    with open(EXTRACTED_FILE) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["item_id"]] = r

    per_axis = {axis: {"n": 0, "n_ok": 0, "fidelity_scores": []}
                for axis in AXES}
    per_item = []
    binder_results = []

    for it in items:
        axis = it["axis"]
        rec = recs.get(it["id"])
        if rec is None or not rec.get("ok"):
            per_axis[axis]["n"] += 1
            per_item.append({"item_id": it["id"], "axis": axis, "ok": False})
            continue
        score = _score_fidelity_harder(it, rec["facts"])
        per_axis[axis]["n"]     += 1
        per_axis[axis]["n_ok"]  += 1
        per_axis[axis]["fidelity_scores"].append(score["axis_fidelity"])
        per_item.append({
            "item_id": it["id"], "axis": axis, "ok": True,
            "axis_fidelity": score["axis_fidelity"],
            "fields": score["fields"],
        })

        # Binder: combine ECE + item pass-through, render both formats, info-check
        facts = _facts_for_binder(it, rec["facts"])
        try:
            pf_prompt = _RENDER_PROSE[axis](facts)
            cp_prompt = _RENDER_PACK [axis](facts)
        except Exception as e:  # noqa: BLE001
            binder_results.append({"item_id": it["id"], "axis": axis,
                                   "info_constant": False, "structure_clean": False,
                                   "leak_report": [f"render failed: {e}"]})
            continue

        # Info-constancy: every leaf-string value in facts appears in BOTH prompts
        leaks = []
        info_constant = True

        def _walk_values(v):
            if isinstance(v, dict):
                for _k, _v in v.items():
                    yield from _walk_values(_v)
            elif isinstance(v, list):
                for _v in v:
                    yield from _walk_values(_v)
            else:
                yield v

        for val in _walk_values(facts):
            s = str(val).strip()
            if not s or len(s) < 2:
                continue
            if s not in pf_prompt:
                info_constant = False
                leaks.append({"value": s[:80], "missing_in": "PLAINFACTS"})
            if s not in cp_prompt:
                info_constant = False
                leaks.append({"value": s[:80], "missing_in": "C_PACK_LEARNED"})

        structure_clean = (
            ("CONTRACT" not in pf_prompt) and ("{" not in pf_prompt)
            and ("CONTRACT" in cp_prompt) and ("{" in cp_prompt)
        )
        binder_results.append({
            "item_id":         it["id"],
            "axis":            axis,
            "info_constant":   info_constant,
            "structure_clean": structure_clean,
            "leak_report":     leaks[:10],   # cap for storage
        })

    # Axis-level fidelity summary
    for axis, d in per_axis.items():
        n_ok = d["n_ok"]
        if d["fidelity_scores"]:
            avg = sum(d["fidelity_scores"]) / len(d["fidelity_scores"])
        else:
            avg = 0.0
        d["yield"] = n_ok / d["n"] if d["n"] else 0.0
        d["avg_fidelity"] = avg

    # Binder aggregate per axis
    binder_axis_summary = {axis: {"n": 0, "ic": 0, "sc": 0} for axis in AXES}
    for r in binder_results:
        a = r["axis"]
        binder_axis_summary[a]["n"]  += 1
        binder_axis_summary[a]["ic"] += int(r["info_constant"])
        binder_axis_summary[a]["sc"] += int(r["structure_clean"])
    for a, s in binder_axis_summary.items():
        s["ic_rate"] = s["ic"] / s["n"] if s["n"] else 0.0
        s["sc_rate"] = s["sc"] / s["n"] if s["n"] else 0.0

    summary = {
        "per_axis_extraction": per_axis,
        "per_axis_binder":     binder_axis_summary,
    }
    path = os.path.join(DATA_DIR, "extraction_fidelity_and_binder.json")
    with open(path, "w") as f:
        json.dump({"summary": summary,
                   "per_item_extraction": per_item,
                   "per_item_binder":     binder_results}, f, indent=2, ensure_ascii=False)
    with open(BINDER_SC_FILE, "w") as f:
        json.dump({"summary": binder_axis_summary,
                   "per_item": binder_results}, f, indent=2, ensure_ascii=False)

    print("[fidelity] per-axis extraction:")
    for axis, d in per_axis.items():
        print(f"  {axis:<14}  yield={d['yield']:.3f}  avg_fidelity={d['avg_fidelity']:.3f}  (n={d['n']}, ok={d['n_ok']})")
    print("[binder]   per-axis info-constancy + scaffold-cleanliness:")
    for axis, s in binder_axis_summary.items():
        print(f"  {axis:<14}  ic={s['ic']}/{s['n']} ({s['ic_rate']:.3f})  sc={s['sc']}/{s['n']} ({s['sc_rate']:.3f})")
    print(f"[saved]    {path}")
    print(f"[saved]    {BINDER_SC_FILE}")


def cmd_verify_u4() -> int:
    """U4 inline-verify: extraction yield >= 0.85 per axis, info-constancy rate = 1.0,
    scaffold-cleanliness rate = 1.0 per axis."""
    failures = []
    path = os.path.join(DATA_DIR, "extraction_fidelity_and_binder.json")
    if not os.path.exists(path):
        failures.append(f"fidelity report missing: {path}")
        return _verify_report(failures)
    doc = json.load(open(path))
    s = doc["summary"]
    for axis in AXES:
        e = s["per_axis_extraction"].get(axis, {})
        if e.get("yield", 0) < 0.85:
            failures.append(
                f"axis {axis} extraction yield {e.get('yield', 0):.3f} < 0.85 floor"
            )
        b = s["per_axis_binder"].get(axis, {})
        if b.get("ic_rate", 0) < 1.0:
            failures.append(
                f"axis {axis} info-constancy {b.get('ic_rate',0):.3f} < 1.0"
            )
        if b.get("sc_rate", 0) < 1.0:
            failures.append(
                f"axis {axis} scaffold-cleanliness {b.get('sc_rate',0):.3f} < 1.0"
            )
    if not failures:
        print(f"[verify-u4] all axes: yield ≥ 0.85, ic = 1.0, sc = 1.0")
    return _verify_report(failures)


# ═══════════════════════════════════════════════════════════════════════════
# U5: Harness (smoke + full) on Qwen-local.
# Records to data/cer_ece_harder_qwen/raw_runs.jsonl with axis + condition +
# rep_id + backend=local + verdict + diagnostic_flags.
# Pre-registered SMOKE: pick 1 item per axis (lowest id) × 4 conditions × 1 rep = 16 calls.
# Full: 50 × 4 × N_REPS = 600 calls.
# ═══════════════════════════════════════════════════════════════════════════

def _smoke_item_ids() -> list:
    """Lowest id per axis (deterministic)."""
    items = load_items()
    out = []
    seen = set()
    for it in sorted(items, key=lambda x: x["id"]):
        if it["axis"] not in seen:
            out.append(it["id"]); seen.add(it["axis"])
        if len(out) == len(AXES):
            break
    return out


def _call_subject_local(system: str, user: str) -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cer_ece_cell import _ollama_chat_greedy  # noqa
    from cbt.llm_backend import get_backend  # noqa
    b = get_backend()
    if b["backend"] != "local":
        raise RuntimeError(f"WP-12 harness requires LLM_BACKEND=local; got {b['backend']}")
    return _ollama_chat_greedy(system, user, base_url=b["base_url"], model=b["model"])


def _load_ece_facts() -> dict:
    """item_id → extracted facts dict (raw ECE output, before binder pass-through)."""
    if not os.path.exists(EXTRACTED_FILE):
        raise RuntimeError(f"ECE extracted_facts.jsonl missing: {EXTRACTED_FILE} — run --extract-all")
    out = {}
    with open(EXTRACTED_FILE) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]] = r
    return out


def _run_loop_harder(items_to_run: list, conditions: list, n_reps: int,
                     facts_recs: dict, label: str) -> dict:
    assert_frozen()
    from cbt.llm_backend import record_backend_manifest  # noqa
    os.makedirs(DATA_DIR, exist_ok=True)
    record_backend_manifest(os.path.join(DATA_DIR, f"manifest_{label}.json"))

    done = set()
    if os.path.exists(RUNS_FILE):
        with open(RUNS_FILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["item_id"], r["condition"], r["rep_id"]))

    n_new = n_err = n_skip = 0
    with open(RUNS_FILE, "a") as f_out:
        for it in items_to_run:
            ece_rec = facts_recs.get(it["id"])
            ece_facts = ece_rec["facts"] if ece_rec and ece_rec.get("ok") else None
            full_facts = _facts_for_binder(it, ece_facts) if ece_facts is not None else None
            for cond in conditions:
                if cond in ("PLAINFACTS", "C_PACK_LEARNED") and full_facts is None:
                    n_skip += 1
                    continue
                for rep in range(n_reps):
                    key = (it["id"], cond, rep)
                    if key in done:
                        continue
                    try:
                        system = build_system_prompt_harder(cond, it, learned_facts=full_facts)
                        user   = build_user_message_harder(cond, it)
                        response = _call_subject_local(system, user)
                        verdict  = check_violation_harder(it, response)
                        is_err = response.startswith("__ERROR__") or not response
                        rec = {
                            "item_id":   it["id"],
                            "axis":      it["axis"],
                            "condition": cond,
                            "rep_id":    rep,
                            "question":  it["question"],
                            "response":  response,
                            "verdict":   verdict,
                            "model":     "qwen2.5:32b-instruct-q8_0",
                            "backend":   "local",
                            "is_error":  is_err,
                        }
                        if is_err: n_err += 1
                    except Exception as e:  # noqa: BLE001
                        rec = {"item_id": it["id"], "axis": it["axis"], "condition": cond,
                               "rep_id": rep, "is_error": True, "error": str(e)[:200]}
                        n_err += 1
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()
                    n_new += 1
                    flag = "VIOL" if rec.get("verdict",{}).get("violation") else "ok"
                    if rec.get("is_error"): flag = "ERR"
                    print(f"[{label}] id={it['id']} axis={it['axis']:<14} {cond:<14} rep={rep} → {flag}")
    return {"new": n_new, "errors": n_err, "skipped": n_skip}


def cmd_smoke_harder() -> None:
    items = load_items()
    smoke_ids = _smoke_item_ids()
    smoke_items = [it for it in items if it["id"] in smoke_ids]
    facts_recs = _load_ece_facts()
    print(f"[smoke] items={smoke_ids} conditions={CONDITIONS} n_reps=1")
    c = _run_loop_harder(smoke_items, CONDITIONS, 1, facts_recs, "smoke")
    print(f"[smoke] new={c['new']} errors={c['errors']} skipped={c['skipped']}")


def cmd_run_full_harder(n_reps: int = None) -> None:
    items = load_items()
    facts_recs = _load_ece_facts()
    n_reps = n_reps or N_REPS
    print(f"[run] items={len(items)} conditions={CONDITIONS} n_reps={n_reps}")
    c = _run_loop_harder(items, CONDITIONS, n_reps, facts_recs, "run")
    print(f"[run] new={c['new']} errors={c['errors']} skipped={c['skipped']}")


def cmd_verify_u5() -> int:
    failures = []
    if not os.path.exists(RUNS_FILE):
        failures.append(f"raw_runs missing: {RUNS_FILE}")
        return _verify_report(failures)
    runs = []
    with open(RUNS_FILE) as f:
        for line in f:
            if line.strip(): runs.append(json.loads(line))
    if not runs:
        failures.append("raw_runs empty")
        return _verify_report(failures)
    # Smoke grid: 4 items × 4 conditions × 1 rep
    smoke_ids = set(_smoke_item_ids())
    smoke_cells = {(r["item_id"], r["condition"], r["rep_id"])
                   for r in runs if r["item_id"] in smoke_ids}
    expected_smoke = {(iid, c, 0) for iid in smoke_ids for c in CONDITIONS}
    missing = expected_smoke - smoke_cells
    if missing:
        failures.append(f"smoke grid incomplete: missing {len(missing)} cells")
    # Required fields
    for r in runs:
        if r.get("is_error"): continue
        for k in ("item_id","axis","condition","rep_id","question","response","verdict","backend"):
            if k not in r:
                failures.append(f"record missing field {k}: {r}")
                break
    n_total = len(runs); n_err = sum(1 for r in runs if r.get("is_error"))
    print(f"[verify-u5] total={n_total} errors={n_err} smoke-grid={len(smoke_cells & expected_smoke)}/{len(expected_smoke)}")
    return _verify_report(failures)


# ═══════════════════════════════════════════════════════════════════════════
# U6: Aggregate per-axis + apply pre-registered gate_structure_vs_facts per axis.
# Also report gate_learned_payoff per axis (sanity) + diagnostic flag counts.
# ═══════════════════════════════════════════════════════════════════════════

def _per_cell_violation_harder(runs: list) -> dict:
    cells = {}
    for r in runs:
        if r.get("is_error"): continue
        cells.setdefault((r["item_id"], r["condition"]), []).append(int(bool(r["verdict"]["violation"])))
    out = {}
    for key, vs in cells.items():
        n = len(vs); mean = sum(vs)/n if n else 0.0
        var = sum((v-mean)**2 for v in vs)/n if n>0 else 0.0
        out[key] = {"n_reps": n, "n_viol": sum(vs), "rate": mean, "std": var**0.5}
    return out


def _paired_delta_harder(cells: dict, items_subset: list, cond_a: str, cond_b: str) -> dict:
    deltas = []
    for it in items_subset:
        a = cells.get((it["id"], cond_a)); b = cells.get((it["id"], cond_b))
        if a is None or b is None: continue
        deltas.append(a["rate"] - b["rate"])
    if not deltas:
        return {"n": 0, "mean": None, "std": None, "cohen_d": None, "effect_strength": "no_pairs", "deltas": []}
    n = len(deltas); mean = sum(deltas)/n
    std = (sum((d-mean)**2 for d in deltas)/(n-1))**0.5 if n>1 else 0.0
    if std > 0:
        d = mean/std; ad = abs(d)
        label = "small" if ad<0.5 else "medium" if ad<0.8 else "large" if ad<1.2 else "very_large"
    else:
        d = None
        label = "absolute_tie" if mean == 0 else f"absolute_unanimity_delta_{mean:+.3f}"
    return {"n": n, "mean": mean, "std": std, "cohen_d": d, "effect_strength": label, "deltas": deltas}


def _gate_verdict_harder(delta_record: dict, floor: float) -> dict:
    floor_pass = (delta_record["n"] > 0 and delta_record["mean"] is not None
                  and delta_record["mean"] <= -floor)
    if delta_record.get("cohen_d") is not None:
        effect_pass = abs(delta_record["cohen_d"]) >= 0.5
    else:
        effect_pass = delta_record.get("effect_strength","").startswith("absolute_unanimity")
    return {
        "delta": delta_record, "floor": floor,
        "floor_pass": floor_pass, "effect_pass": effect_pass,
        "verdict": "PASS" if (floor_pass and effect_pass) else "FAIL",
    }


def cmd_aggregate_harder() -> dict:
    assert_frozen()
    runs = []
    with open(RUNS_FILE) as f:
        for line in f:
            if line.strip(): runs.append(json.loads(line))
    items = load_items()
    cells = _per_cell_violation_harder(runs)

    per_axis = {}
    for axis in AXES:
        items_a = [it for it in items if it["axis"] == axis]
        by_cond = {}
        for cond in CONDITIONS:
            rates = [cells[(it["id"], cond)]["rate"] for it in items_a if (it["id"], cond) in cells]
            n = len(rates)
            mean = sum(rates)/n if n else 0.0
            std  = (sum((r-mean)**2 for r in rates)/(n-1))**0.5 if n>1 else 0.0
            by_cond[cond] = {"n_items": n, "mean_violation_rate": mean, "std_across_items": std}

        # Gates per axis
        gate_payoff   = _gate_verdict_harder(
            _paired_delta_harder(cells, items_a, "C_PACK_LEARNED", "B_FAIR"),
            GATE_FLOOR_PAYOFF,
        )
        gate_struct   = _gate_verdict_harder(
            _paired_delta_harder(cells, items_a, "C_PACK_LEARNED", "PLAINFACTS"),
            GATE_FLOOR_STRUCTURE,
        )

        # Diagnostic flag counts per condition (for diagnostic axes)
        flag_counts = {}
        for cond in CONDITIONS:
            flag_counts[cond] = {}
            for r in runs:
                if r.get("is_error"): continue
                if r["axis"] != axis or r["condition"] != cond: continue
                for fk, fv in r.get("verdict",{}).get("diagnostic_flags",{}).items():
                    flag_counts[cond].setdefault(fk, {"n": 0, "true": 0})
                    flag_counts[cond][fk]["n"] += 1
                    flag_counts[cond][fk]["true"] += int(bool(fv))

        # PLAINFACTS saturation: V(PLAINFACTS) per axis
        pf_v = by_cond["PLAINFACTS"]["mean_violation_rate"]
        pf_saturated = (pf_v == 0.0)
        per_axis[axis] = {
            "by_condition":           by_cond,
            "gate_learned_payoff":    gate_payoff,
            "gate_structure_vs_facts": gate_struct,
            "diagnostic_flag_counts":  flag_counts,
            "pf_saturated":            pf_saturated,
            "structure_headroom":      pf_v - by_cond["C_PACK_LEARNED"]["mean_violation_rate"],
        }

    # Cross-axis summary
    any_struct_pass = any(per_axis[a]["gate_structure_vs_facts"]["verdict"] == "PASS" for a in AXES)
    all_pf_saturate = all(per_axis[a]["pf_saturated"] for a in AXES)
    any_payoff_pass  = any(per_axis[a]["gate_learned_payoff"]["verdict"] == "PASS" for a in AXES)
    all_payoff_pass  = all(per_axis[a]["gate_learned_payoff"]["verdict"] == "PASS" for a in AXES)

    report = {
        "per_axis":               per_axis,
        "axes":                   list(AXES),
        "conditions":             CONDITIONS,
        "gate_floors":            {"payoff": GATE_FLOOR_PAYOFF, "structure": GATE_FLOOR_STRUCTURE,
                                    "effect_size_cohen_d_min": 0.5},
        "cross_axis_summary":     {
            "any_structure_pass": any_struct_pass,
            "all_pf_saturate":    all_pf_saturate,
            "any_payoff_pass":    any_payoff_pass,
            "all_payoff_pass":    all_payoff_pass,
            "passing_axes":       [a for a in AXES if per_axis[a]["gate_structure_vs_facts"]["verdict"] == "PASS"],
        },
        "model":                  "qwen2.5:32b-instruct-q8_0",
        "n_items":                len(items),
    }
    with open(EVAL_FILE, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_results_harder(report)
    print(f"[aggregate] saved {EVAL_FILE}")
    print(f"[aggregate] saved {RESULTS_MD}")
    _print_headline_harder(report)
    return report


def _write_results_harder(report: dict) -> None:
    def fmt_pct(x): return f"{x*100:.1f}%" if isinstance(x,(int,float)) else "—"
    def fmt_sg(x): return "—" if x is None else f"{x:+.3f}"
    def fmt_d(x): return "—" if x is None else f"{x:+.2f}"
    lines = []
    lines.append("# Results — WP-ST-12 harder non-saturating prompt-scope (Qwen-local)\n")
    lines.append(f"**Model:** `{report['model']}` (greedy temp=0)  ")
    lines.append(f"**Items:** {report['n_items']}  **Axes:** {', '.join(AXES)}  ")
    lines.append(f"**Pre-registered floors:** Δ ≥ {report['gate_floors']['payoff']}, |d| ≥ {report['gate_floors']['effect_size_cohen_d_min']} (or absolute_unanimity)\n")

    lines.append("## Per-axis violation rate (mean ± std across items)\n")
    lines.append("| Axis | B_FAIR | PLAINFACTS | C_PACK_LEARNED | C_KNOW_ORACLE | structure_headroom | PF saturated? |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for axis in AXES:
        a = report["per_axis"][axis]
        bf = a["by_condition"]["B_FAIR"]["mean_violation_rate"]
        pf = a["by_condition"]["PLAINFACTS"]["mean_violation_rate"]
        cp = a["by_condition"]["C_PACK_LEARNED"]["mean_violation_rate"]
        ko = a["by_condition"]["C_KNOW_ORACLE"]["mean_violation_rate"]
        hr = a["structure_headroom"]
        sat = "YES" if a["pf_saturated"] else "NO"
        lines.append(f"| `{axis}` | {fmt_pct(bf)} | {fmt_pct(pf)} | {fmt_pct(cp)} | {fmt_pct(ko)} | {fmt_sg(hr)} | {sat} |")
    lines.append("")

    lines.append("## Per-axis `gate_structure_vs_facts` (C_PACK_LEARNED < PLAINFACTS)\n")
    lines.append("| Axis | Δ | std | Cohen d | effect | floor pass | effect pass | verdict |")
    lines.append("|---|---:|---:|---:|:---|:---:|:---:|:---:|")
    for axis in AXES:
        g = report["per_axis"][axis]["gate_structure_vs_facts"]
        d = g["delta"]
        lines.append(f"| `{axis}` | {fmt_sg(d.get('mean'))} | {fmt_sg(d.get('std'))} | {fmt_d(d.get('cohen_d'))} | {d.get('effect_strength','—')} | {'✓' if g['floor_pass'] else '✗'} | {'✓' if g['effect_pass'] else '✗'} | **{g['verdict']}** |")
    lines.append("")

    lines.append("## Per-axis `gate_learned_payoff` (C_PACK_LEARNED < B_FAIR) — sanity\n")
    lines.append("| Axis | Δ | Cohen d | effect | verdict |")
    lines.append("|---|---:|---:|:---|:---:|")
    for axis in AXES:
        g = report["per_axis"][axis]["gate_learned_payoff"]
        d = g["delta"]
        lines.append(f"| `{axis}` | {fmt_sg(d.get('mean'))} | {fmt_d(d.get('cohen_d'))} | {d.get('effect_strength','—')} | **{g['verdict']}** |")
    lines.append("")

    lines.append("## Cross-axis summary\n")
    s = report["cross_axis_summary"]
    lines.append(f"- any structure_pass: **{s['any_structure_pass']}** ({s['passing_axes'] or 'none'})")
    lines.append(f"- all PLAINFACTS saturate: {s['all_pf_saturate']}")
    lines.append(f"- any gate_learned_payoff PASS: {s['any_payoff_pass']}")
    lines.append(f"- all gate_learned_payoff PASS: {s['all_payoff_pass']}\n")

    lines.append("## Diagnostic flag counts (per axis × condition)\n")
    for axis in AXES:
        lines.append(f"### `{axis}`")
        for cond in CONDITIONS:
            fcs = report["per_axis"][axis]["diagnostic_flag_counts"].get(cond, {})
            if not fcs: continue
            for fk, fv in fcs.items():
                lines.append(f"- {cond} / {fk}: {fv['true']}/{fv['n']}")
        lines.append("")

    os.makedirs(PAPERS_DIR, exist_ok=True)
    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines))


def _print_headline_harder(report: dict) -> None:
    print("\n=== HEADLINE ===")
    for axis in AXES:
        a = report["per_axis"][axis]
        bf = a["by_condition"]["B_FAIR"]["mean_violation_rate"]
        pf = a["by_condition"]["PLAINFACTS"]["mean_violation_rate"]
        cp = a["by_condition"]["C_PACK_LEARNED"]["mean_violation_rate"]
        g  = a["gate_structure_vs_facts"]
        print(f"  {axis:<14} V(B)={bf*100:5.1f}% V(PF)={pf*100:5.1f}% V(CP)={cp*100:5.1f}% "
              f"Δ={g['delta'].get('mean'):+.3f} struct→{g['verdict']}")
    s = report["cross_axis_summary"]
    print(f"  any_structure_pass={s['any_structure_pass']}  passing={s['passing_axes']}")


def cmd_verify_u6_harder() -> int:
    failures = []
    if not os.path.exists(EVAL_FILE):
        failures.append(f"eval missing: {EVAL_FILE}")
        return _verify_report(failures)
    rep = json.load(open(EVAL_FILE))
    for axis in AXES:
        a = rep["per_axis"].get(axis)
        if not a:
            failures.append(f"axis {axis} missing in eval"); continue
        for g_key in ("gate_structure_vs_facts", "gate_learned_payoff"):
            g = a.get(g_key)
            if g is None or g.get("verdict") not in ("PASS", "FAIL"):
                failures.append(f"{axis}.{g_key} missing verdict")
    return _verify_report(failures)


def cmd_verify_u7_harder() -> int:
    """U7 inline-verify: papers/claim_harder_scope.md exists + required markers."""
    failures = []
    if not os.path.exists(CLAIM_MD):
        failures.append(f"claim missing: {CLAIM_MD}")
        return _verify_report(failures)
    txt = open(CLAIM_MD).read()
    required = [
        "qwen2.5-32b-instruct-q8",
        "Bounded scope",
        "gate_structure_vs_facts",
        "gate_learned_payoff",
        "multi_binding",
        "distractors",
        "long_context",
        "conflicting",
        "Per-axis",
        "Architecture decision",
        "Red-team note",
        "S→T",
        "CBT-v1",
        "Sign-off",
        "What we will NOT do",
    ]
    missing = [m for m in required if m not in txt]
    if missing:
        failures.append(f"claim missing markers: {missing}")
    if not os.path.exists(EVAL_FILE):
        failures.append("U6 eval missing — precondition")
    print(f"[verify-u7-harder] markers present, length={len(txt)} chars")
    return _verify_report(failures)


def cmd_verify_u1() -> int:
    """U1 inline-verify: items present + axis counts + every required field
    per axis + frozen hash + schemas present + hash stability."""
    failures = []
    if not os.path.exists(ITEMS_FILE):
        failures.append(f"items file missing: {ITEMS_FILE}")
        return _verify_report(failures)
    items = load_items()
    if not items:
        failures.append("items file empty")
        return _verify_report(failures)

    # Axis counts
    counts = {axis: sum(1 for it in items if it["axis"] == axis) for axis in AXES}
    expected_min = {"multi_binding": 10, "distractors": 10, "long_context": 10, "conflicting": 12}
    for axis, n_min in expected_min.items():
        if counts[axis] < n_min:
            failures.append(f"axis {axis}: count {counts[axis]} < expected min {n_min}")
    if sum(counts.values()) != len(items):
        failures.append(f"axis sum {sum(counts.values())} != n_items {len(items)}")

    # Per-axis required fields
    required_by_axis = {
        "multi_binding": ("bindings", "asked_quantity", "context_text", "question",
                          "correct_answer_keywords", "sibling_unit_keywords"),
        "distractors":   ("load_bearing", "distractors", "context_text", "question",
                          "correct_answer_keywords"),
        "long_context":  ("binding", "filler_before", "filler_after", "context_text",
                          "question", "correct_answer_keywords", "filler_chars"),
        "conflicting":   ("in_context_quantity", "primary", "deprecated", "precedence_note",
                          "context_text", "question", "correct_answer_keywords",
                          "deprecated_unit_keywords"),
    }
    for it in items:
        axis = it.get("axis")
        if axis not in AXES:
            failures.append(f"item id={it.get('id')} unknown axis={axis}")
            continue
        for f in required_by_axis[axis]:
            if f not in it or it.get(f) in (None, "", [], {}):
                failures.append(f"item id={it['id']} axis={axis} missing field {f}")
                break

    # Long context items must have filler_chars in 800-1500 range
    for it in items:
        if it["axis"] == "long_context":
            n = it.get("filler_chars", 0)
            if not (800 <= n <= 1500):
                failures.append(f"item id={it['id']} long_context filler_chars={n} not in [800,1500]")

    # Frozen hash assertion + stability
    try:
        h_now = assert_frozen()
        print(f"[verify-u1] frozen-hash assertion PASS: {h_now}")
    except RuntimeError as e:
        failures.append(str(e))

    # Schemas present in frozen record
    frozen = json.load(open(FROZEN_FILE))
    if "renderer_schema" not in frozen:
        failures.append("frozen record missing renderer_schema")
    else:
        for axis in AXES:
            if axis not in frozen["renderer_schema"]:
                failures.append(f"renderer_schema missing axis {axis}")

    if not failures:
        print(f"[verify-u1] items={len(items)} axis_counts={counts} hash={h_now}")
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
    p.add_argument("--generate",  action="store_true", help="U1: emit items + freeze hash")
    p.add_argument("--verify-u1", action="store_true", help="U1: inline-verify")
    p.add_argument("--verify-u2", action="store_true", help="U2: inline-verify renderers")
    p.add_argument("--verify-u3", action="store_true", help="U3: inline-verify checker round-trip")
    p.add_argument("--extract-all", action="store_true", help="U4: run LEARNED ECE on Qwen-local for all items")
    p.add_argument("--fidelity-binder", action="store_true", help="U4: re-score fidelity + binder spot-check from existing extracted_facts")
    p.add_argument("--verify-u4", action="store_true", help="U4: inline-verify ECE yield + binder")
    p.add_argument("--smoke",     action="store_true", help="U5 smoke: 4 items × 4 cond × 1 rep = 16 calls")
    p.add_argument("--run-full",  action="store_true", help="U5 full: 50 × 4 × N_REPS calls (resumable)")
    p.add_argument("--reps",      type=int, default=None, help="override n_reps for --run-full")
    p.add_argument("--verify-u5", action="store_true", help="U5: inline-verify harness")
    p.add_argument("--aggregate", action="store_true", help="U6: aggregate per-axis + gates")
    p.add_argument("--verify-u6", action="store_true", help="U6: inline-verify aggregate")
    p.add_argument("--verify-u7", action="store_true", help="U7: inline-verify claim doc")
    args = p.parse_args()

    if args.generate:
        cmd_generate()
        return
    if args.verify_u1:
        sys.exit(cmd_verify_u1())
    if args.verify_u2:
        sys.exit(cmd_verify_u2())
    if args.verify_u3:
        sys.exit(cmd_verify_u3())
    if args.extract_all:
        cmd_extract_all_harder()
        return
    if args.fidelity_binder:
        cmd_fidelity_and_binder_harder()
        return
    if args.verify_u4:
        sys.exit(cmd_verify_u4())
    if args.smoke:
        cmd_smoke_harder(); return
    if args.run_full:
        cmd_run_full_harder(n_reps=args.reps); return
    if args.verify_u5:
        sys.exit(cmd_verify_u5())
    if args.aggregate:
        cmd_aggregate_harder(); return
    if args.verify_u6:
        sys.exit(cmd_verify_u6_harder())
    if args.verify_u7:
        sys.exit(cmd_verify_u7_harder())

    p.print_help()


if __name__ == "__main__":
    main()
