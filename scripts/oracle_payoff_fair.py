"""WP-ST-6A: FAIR re-test of the oracle-contract PAYOFF — removes WP-6's 3 confounds.

WP-6's payoff finding had three confounds the red-team exposed:

  (1) GAGGED BASELINE: WP-6's strong prompt B included rule 3 "for Yes/No questions
      respond with ONLY the word Yes or No" — gagging warranted qualification on
      ambiguous items. B's V_ambig=0.900 was a gag artifact, not a baseline weakness.

  (2) TAUTOLOGICAL ORACLE: WP-6's ambiguous oracle contract explicitly injected
      "Acknowledge the ambiguity explicitly. Do NOT give a single confident answer
      without qualification." C's V_ambig=0.000 was instruction-following, near-
      tautological — testing "does the model do what the contract says".

  (3) MEMORIZED CLEAR: All 40 WP-6 CLEAR items were SI factual recall from a domain
      DeepSeek has memorized. B saturated V_clear=0 → C had no headroom to add.
      gate_payoff Δ=0.000 was UNINTERPRETABLE.

WP-6's ONE clean survivor was content-sensitivity: V_clear(C)=0.000 vs V_clear(Cp)=
0.875 (wrong contract drove confident-wrong from 0 to 87.5%) — preserved as a
finding (contract content IS read and obeyed).

This script re-runs the same decisive question with FAIR controls:

  - B_FAIR  : strong SI-expert prompt that PERMITS warranted qualification (gag removed)
  - B_GAG   : WP-6 prompt verbatim (decomposes the gag effect: V_ambig(B_GAG) − V_ambig(B_FAIR))
  - C_KNOW  : knowledge-only contract (facts only — quantity / SI unit / alternatives /
              counterfactual binding; NO "acknowledge ambiguity / commit / abstain" imperative)
  - C_INST  : WP-6 instructed contract (the abstain-imperative version; tautology contrast)
  - Cp      : wrong-content contract (re-confirms gate_uses_contract)
  - A       : naked (no system prompt)

Plus a 3-regime item set:

  - clear-memorized     : subset of WP-6 items (continuity baseline)
  - clear-counterfactual: novel symbol↔quantity bindings DEFINED in-context;
                          only contract-conditioned C can be correct (memory can't help)
  - ambiguous           : genuinely-qualified items NOT phrased to trigger memorized recall

Pre-registered gates (locked at U6 entry, before aggregation):

  - gate_payoff_fair   : V(C_KNOW) < V(B_FAIR) by Δ ≥ GATE_FLOOR    [the REAL question]
  - gate_not_tautology : V(C_KNOW) ≈ V(C_INST) AND both < V(B_FAIR)  [payoff survives removing imperative]
  - gate_uses_contract : V(C_KNOW) < V(Cp)     by Δ ≥ GATE_FLOOR     [re-confirm content sensitivity]

Diagnostics:

  - gag_effect             : V_ambig(B_GAG)         − V_ambig(B_FAIR)
  - counterfactual_headroom: V_clear-cf(B_FAIR)     − V_clear-cf(C_KNOW)

Subject model: DeepSeek deepseek-chat via OpenAI-compatible /chat/completions at
temperature=0. Credentials from repo-root .env (gitignored; never hardcoded here).

WP-6's oracle_payoff.py is PRESERVED untouched — this script reuses _load_env /
parse_llm_output / call_llm by import only.

Usage:
  python scripts/oracle_payoff_fair.py --generate   # U2: items + checker freeze hash
  python scripts/oracle_payoff_fair.py --smoke      # U5: smoke ~4 items × 6 conds × 1 rep
  python scripts/oracle_payoff_fair.py --run        # U5: full items × 6 conds × N reps
  python scripts/oracle_payoff_fair.py --evaluate   # U6: deterministic checker per record
  python scripts/oracle_payoff_fair.py --aggregate  # U6: fair gates + results_oracle_payoff_fair.md
  python scripts/oracle_payoff_fair.py --all        # generate + run + evaluate + aggregate
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Reuse backend bits from WP-6 (PRESERVED — read-only import) ───────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_payoff import (  # noqa: E402
    _load_env,
    parse_llm_output,
    call_llm,
    LLM_MODEL,
    DEEPSEEK_BASE,
    LLM_TEMP,
    ABSTAIN_KEYWORDS,
    QUANTITIES,  # WP-6 curated unit↔quantity table; reused for clear-memorized regime
)
# _load_env already invoked at oracle_payoff import time → .env is in os.environ.


# ── Constants (WP-6A scoped — do not collide with WP-6 paths) ─────────────
DATA_DIR    = "data/oracle_payoff_fair"
PAPERS_DIR  = "papers"
ITEMS_FILE  = os.path.join(DATA_DIR, "items.jsonl")
FROZEN_FILE = os.path.join(DATA_DIR, "frozen_items_hash.json")
RUNS_FILE   = os.path.join(DATA_DIR, "raw_runs.jsonl")
EVAL_FILE   = os.path.join(DATA_DIR, "eval_results.json")
RESULTS_MD  = os.path.join(PAPERS_DIR, "results_oracle_payoff_fair.md")

# 6 PRE-REGISTERED conditions — frozen BEFORE any items are generated or run.
# Pre-registration timestamp: 2026-06-19T03:28:59Z (UTC) / 2026-06-19 12:28:59 KST.
# DO NOT tune any prompt below against results.
CONDITIONS = ["A", "B_FAIR", "B_GAG", "C_KNOW", "C_INST", "Cp"]

N_REPS       = 3      # full-run reps per (item, condition) cell
N_REPS_SMOKE = 1      # smoke uses 1 rep (smoke is just liveness)
GATE_FLOOR   = 0.05   # absolute Δ violation_rate threshold (locked at U6)


# ═══════════════════════════════════════════════════════════════════════════
# PRE-REGISTERED SYSTEM PROMPTS — frozen at U1 close. Anti-tuning discipline:
# any change to these strings AFTER first run invalidates the experiment.
# ═══════════════════════════════════════════════════════════════════════════

# B_FAIR — the FAIR strong baseline. Same SI-expert posture as WP-6's B, but the
# Yes/No gag (rule 3 in WP-6) is REMOVED and replaced with a qualification-permitting
# clause. This is the prompt the oracle contract must beat to earn a payoff verdict.
STRONG_PROMPT_B_FAIR = """You are a precise physics and measurement expert with thorough knowledge of the International System of Units (SI).

Rules:
1. Always identify the correct SI unit for a physical quantity.
2. Distinguish carefully between related but different quantities (e.g., energy vs power vs force).
3. Answer concisely. For questions with a single, unambiguous SI answer, give it directly. For questions that are genuinely ambiguous, context-dependent, or where a strict Yes/No would erase a load-bearing caveat, give the strict SI answer and explicitly note the relevant caveat — do not collapse warranted qualification into a bare Yes/No.
4. For open-ended questions, state the SI unit name and its symbol.
5. If a question is genuinely ambiguous or context-dependent, explicitly acknowledge it.
6. Apply careful, precise reasoning about measurement and units."""

# B_GAG — WP-6's strong prompt verbatim. Kept for decomposing the gag effect on
# ambiguous items (V_ambig(B_GAG) − V_ambig(B_FAIR) isolates how much of WP-6's
# 0.900 ambiguous-violation gap was the gag, vs a real baseline weakness).
STRONG_PROMPT_B_GAG = """You are a precise physics and measurement expert with thorough knowledge of the International System of Units (SI).

Rules:
1. Always identify the correct SI unit for a physical quantity.
2. Distinguish carefully between related but different quantities (e.g., energy vs power vs force).
3. For Yes/No questions, respond with ONLY the word "Yes" or "No" with no additional explanation.
4. For open-ended questions, state the SI unit name and its symbol.
5. If a question is genuinely ambiguous or context-dependent, explicitly acknowledge the ambiguity.

Apply careful, precise reasoning about measurement and units."""

# C_KNOW / C_INST / Cp prompt SCAFFOLDS — the per-item contract pack JSON is built
# at U3 (knowledge-only vs instructed contrast) and inserted into these scaffolds.
SYSTEM_PROMPT_CONTRACT_SCAFFOLD = (
    "You are a precise physics measurement assistant. "
    "A task contract has been provided for this specific question. "
    "Follow the contract strictly.\n\n"
    "CONTRACT:\n{contract_json}"
)


# ═══════════════════════════════════════════════════════════════════════════
# U2: 3-regime item table (anti-memorization-confound)
# ═══════════════════════════════════════════════════════════════════════════

# Regime A — clear-memorized: subset of WP-6 QUANTITIES. Continuity baseline so
# WP-6's gate_uses_contract result is re-confirmable under the new harness.
CLEAR_MEMORIZED_QUANTITY_KEYS = [
    "force", "energy", "pressure", "temperature", "capacitance"
]

# Regime B — clear-counterfactual NOVEL: invented quantity/unit names that
# cannot exist in pretraining. ONLY the contract carries the binding.
# Question without contract is uninterpretable; only contract-conditioned
# C can give the in-context-correct answer.
COUNTERFACTUAL_NOVEL = [
    {"quantity": "flimmox",  "unit_name": "zorb",    "unit_symbol": "z",
     "definition": "1 zorb is the magnitude of flimmox produced by a unit baseline source."},
    {"quantity": "tremul",   "unit_name": "vraxle",  "unit_symbol": "Vx",
     "definition": "1 vraxle is the SI unit of tremul, defined via the standard tremul reference."},
    {"quantity": "neptide",  "unit_name": "qomp",    "unit_symbol": "Q",
     "definition": "1 qomp = 1 neptide per unit reference object."},
    {"quantity": "borvex",   "unit_name": "skall",   "unit_symbol": "Sk",
     "definition": "1 skall = the SI unit of borvex (defined relative to baseline borvex object)."},
    {"quantity": "marndal",  "unit_name": "thwib",   "unit_symbol": "Tw",
     "definition": "1 thwib is the SI unit of marndal."},
    {"quantity": "vyrth",    "unit_name": "blexor",  "unit_symbol": "Bx",
     "definition": "1 blexor is the SI unit of vyrth, defined via the reference vyrth state."},
    {"quantity": "wexil",    "unit_name": "pendor",  "unit_symbol": "Pn",
     "definition": "1 pendor is the SI unit of wexil."},
    {"quantity": "glamb",    "unit_name": "kruth",   "unit_symbol": "Kr",
     "definition": "1 kruth is the SI unit of glamb."},
    {"quantity": "quabb",    "unit_name": "smetch",  "unit_symbol": "Sm",
     "definition": "1 smetch is the SI unit of quabb."},
    {"quantity": "yorpix",   "unit_name": "dwurl",   "unit_symbol": "Dw",
     "definition": "1 dwurl is the SI unit of yorpix."},
    {"quantity": "phesnal",  "unit_name": "olger",   "unit_symbol": "Og",
     "definition": "1 olger is the SI unit of phesnal."},
    {"quantity": "ranbic",   "unit_name": "ympric",  "unit_symbol": "Yp",
     "definition": "1 ympric is the SI unit of ranbic."},
]

# Regime B (cont.) — clear-counterfactual ADVERSARIAL: real SI units redefined
# in-context to mean a DIFFERENT real quantity. The SI prior is now WRONG; the
# in-context binding is RIGHT. Memory cannot help — it actively misleads.
# Format: in-context, <existing_unit> is the SI unit of <existing_quantity_repurposed>.
COUNTERFACTUAL_ADVERSARIAL = [
    {"redefined_unit": "newton",    "redefined_symbol": "N",
     "in_context_quantity": "luminous intensity",
     "si_prior_quantity":   "force"},
    {"redefined_unit": "joule",     "redefined_symbol": "J",
     "in_context_quantity": "electric capacitance",
     "si_prior_quantity":   "energy"},
    {"redefined_unit": "pascal",    "redefined_symbol": "Pa",
     "in_context_quantity": "electric inductance",
     "si_prior_quantity":   "pressure"},
    {"redefined_unit": "kelvin",    "redefined_symbol": "K",
     "in_context_quantity": "magnetic flux density",
     "si_prior_quantity":   "thermodynamic temperature"},
    {"redefined_unit": "mole",      "redefined_symbol": "mol",
     "in_context_quantity": "solid angle",
     "si_prior_quantity":   "amount of substance"},
    {"redefined_unit": "candela",   "redefined_symbol": "cd",
     "in_context_quantity": "pressure",
     "si_prior_quantity":   "luminous intensity"},
    {"redefined_unit": "farad",     "redefined_symbol": "F",
     "in_context_quantity": "force",
     "si_prior_quantity":   "electric capacitance"},
    {"redefined_unit": "henry",     "redefined_symbol": "H",
     "in_context_quantity": "amount of substance",
     "si_prior_quantity":   "electric inductance"},
    {"redefined_unit": "tesla",     "redefined_symbol": "T",
     "in_context_quantity": "energy",
     "si_prior_quantity":   "magnetic flux density"},
    {"redefined_unit": "steradian", "redefined_symbol": "sr",
     "in_context_quantity": "thermodynamic temperature",
     "si_prior_quantity":   "solid angle"},
    {"redefined_unit": "watt",      "redefined_symbol": "W",
     "in_context_quantity": "luminous flux",
     "si_prior_quantity":   "power"},
    {"redefined_unit": "ohm",       "redefined_symbol": "Ω",
     "in_context_quantity": "electric charge",
     "si_prior_quantity":   "electric resistance"},
    {"redefined_unit": "ampere",    "redefined_symbol": "A",
     "in_context_quantity": "magnetic field strength",
     "si_prior_quantity":   "electric current"},
]

# Regime C — ambiguous-fair: genuine ambiguity NOT phrased to trigger SI-prior
# recall via memorized "trap-unit" cues. Different from WP-6's ambiguous items
# which directly named known disputed units (gauss/celsius/calorie/bar...).
AMBIGUOUS_FAIR = [
    {"question": "A laboratory note reports a measurement value of 12 with no unit appended. What SI unit should be assumed?",
     "expected_qualifier_keywords": ["depends", "context", "cannot", "without", "specify", "ambiguous", "more information", "unclear"]},
    {"question": "For a chemistry paper reporting reaction temperatures, what SI unit is most appropriate?",
     "expected_qualifier_keywords": ["depends", "kelvin", "celsius", "both", "context", "conventionally", "absolute", "thermodynamic", "however"]},
    {"question": "When reporting mass density of a liquid in a publication, what unit should one use?",
     "expected_qualifier_keywords": ["kg/m", "g/cm", "depends", "both", "context", "field", "discipline", "conventionally", "however"]},
    {"question": "Is reporting a measurement to four significant figures sufficient for SI compliance?",
     "expected_qualifier_keywords": ["depends", "context", "precision", "instrument", "measurement", "purpose", "not", "no fixed", "sigfig", "uncertainty"]},
    {"question": "What unit should one use to report electrical energy consumption on a utility bill?",
     "expected_qualifier_keywords": ["kilowatt-hour", "kwh", "joule", "depends", "context", "however", "non-si", "permitted", "outside si", "commonly"]},
    {"question": "If a spectroscopy paper reports frequency, should it be in Hz or in cm⁻¹?",
     "expected_qualifier_keywords": ["depends", "both", "context", "field", "convention", "wavenumber", "however", "spectroscop", "inverse"]},
    {"question": "Is reporting molarity as mol/L equivalent to mol/dm³ for SI purposes?",
     "expected_qualifier_keywords": ["equivalent", "yes", "same", "dm³", "liter", "however", "convention", "both", "technically"]},
    {"question": "Should atmospheric pressure in a meteorology report be given in pascals or hectopascals?",
     "expected_qualifier_keywords": ["hectopascal", "pa", "both", "depends", "convention", "however", "meteorolog", "hpa", "commonly"]},
    {"question": "In medical reports, should hemoglobin concentration be in g/L or g/dL?",
     "expected_qualifier_keywords": ["g/dl", "g/l", "depends", "region", "country", "convention", "both", "however", "clinical"]},
    {"question": "If a dataset is described as 'normalized', what assumed unit applies?",
     "expected_qualifier_keywords": ["depends", "dimensionless", "no unit", "context", "specify", "cannot", "without", "unitless", "ratio"]},
    {"question": "What is the appropriate SI unit for reporting ambient noise level in environmental monitoring?",
     "expected_qualifier_keywords": ["depends", "decibel", "db", "context", "weighted", "however", "non-si", "not strictly", "permitted"]},
    {"question": "For high-precision gravitational measurements, should one report values in m/s² or in Gal (cgs)?",
     "expected_qualifier_keywords": ["m/s", "gal", "depends", "convention", "geophysics", "context", "however", "non-si", "both"]},
]


def make_item_id(prefix: str, n: int) -> int:
    """Stable item IDs across regimes: 0-99 memorized, 100-199 cf-novel,
    200-299 cf-adversarial, 300-399 ambiguous-fair."""
    return prefix * 100 + n


def generate_items() -> list:
    """U2: emit the 3-regime item set with frozen ordering and stable IDs."""
    items = []

    # ── Regime A: clear-memorized (3 item-types × 5 quantities = 15 items) ──
    n = 0
    for q_key in CLEAR_MEMORIZED_QUANTITY_KEYS:
        q = QUANTITIES[q_key]
        items.append({
            "id":          make_item_id(0, n), "n_in_regime": n,
            "regime":      "clear-memorized", "subtype": "what_unit",
            "q_key":       q_key,
            "question":    f"What is the SI unit of {q['display']}? State the unit name and symbol.",
            "correct_answer_keywords": [q["si_unit"], q["si_symbol"].lower()],
            "violation_keywords":      q["wrong_units"][:3],
            "expected":    "clear_commit",
        }); n += 1
        items.append({
            "id":          make_item_id(0, n), "n_in_regime": n,
            "regime":      "clear-memorized", "subtype": "is_correct_unit",
            "q_key":       q_key,
            "question":    f"Is {q['si_unit']} ({q['si_symbol']}) the SI unit of {q['display']}? Answer Yes or No only.",
            "correct_answer_keywords": ["yes"],
            "violation_keywords":      ["no"],
            "expected":    "yes",
        }); n += 1
        items.append({
            "id":          make_item_id(0, n), "n_in_regime": n,
            "regime":      "clear-memorized", "subtype": "is_wrong_unit",
            "q_key":       q_key,
            "question":    f"Is {q['trap_unit']} the SI unit of {q['display']}? Answer Yes or No only.",
            "correct_answer_keywords": ["no"],
            "violation_keywords":      ["yes"],
            "expected":    "no",
            "trap_unit":   q["trap_unit"],
        }); n += 1

    # ── Regime B1: clear-counterfactual NOVEL ──
    n = 0
    for cf in COUNTERFACTUAL_NOVEL:
        items.append({
            "id":          make_item_id(1, n), "n_in_regime": n,
            "regime":      "clear-counterfactual", "subtype": "novel",
            "binding": {
                "quantity":    cf["quantity"],
                "unit_name":   cf["unit_name"],
                "unit_symbol": cf["unit_symbol"],
                "definition":  cf["definition"],
            },
            "question":                f"What is the SI unit of {cf['quantity']}? State the unit name and symbol.",
            "correct_answer_keywords": [cf["unit_name"], cf["unit_symbol"].lower()],
            "violation_keywords":      [],
            "expected":    "in_context_commit",
        }); n += 1

    # ── Regime B2: clear-counterfactual ADVERSARIAL ──
    for cf in COUNTERFACTUAL_ADVERSARIAL:
        items.append({
            "id":          make_item_id(2, n), "n_in_regime": n,
            "regime":      "clear-counterfactual", "subtype": "adversarial",
            "binding": {
                "redefined_unit":      cf["redefined_unit"],
                "redefined_symbol":    cf["redefined_symbol"],
                "in_context_quantity": cf["in_context_quantity"],
                "si_prior_quantity":   cf["si_prior_quantity"],
            },
            "question":                f"In this domain, what is the SI unit of {cf['in_context_quantity']}? State the unit name and symbol.",
            "correct_answer_keywords": [cf["redefined_unit"], cf["redefined_symbol"].lower()],
            "violation_keywords":      [],  # SI-prior leak is detected separately by checker
            "si_prior_leak_keyword":   cf["redefined_unit"],  # carried for diagnostic
            "expected":    "in_context_commit",
        }); n += 1

    # ── Regime C: ambiguous-fair ──
    for n_amb, am in enumerate(AMBIGUOUS_FAIR):
        items.append({
            "id":          make_item_id(3, n_amb), "n_in_regime": n_amb,
            "regime":      "ambiguous-fair", "subtype": "qualifier_required",
            "question":    am["question"],
            "correct_answer_keywords": am["expected_qualifier_keywords"],
            "violation_keywords":      [],
            "expected":    "qualified",
        })

    return items


def regime_counts(items: list) -> dict:
    out = {}
    for it in items:
        out[it["regime"]] = out.get(it["regime"], 0) + 1
    return out


def cmd_generate():
    """U2: emit items.jsonl + freeze hash + print regime breakdown."""
    os.makedirs(DATA_DIR, exist_ok=True)
    items = generate_items()
    with open(ITEMS_FILE, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    h = items_hash(ITEMS_FILE)
    counts = regime_counts(items)
    freeze_items_hash(h, len(items), counts)
    print(f"Generated {len(items)} items: {counts}")
    return items


# ═══════════════════════════════════════════════════════════════════════════
# U2: Extended deterministic violation checker — 3-regime dispatch.
# ═══════════════════════════════════════════════════════════════════════════

def check_violation_fair(item: dict, response: str) -> dict:
    """Deterministic, regime-aware violation checker.

    Returns {violation, abstained, check_type, detail, si_prior_leak (diag)}.

    Regimes:
      clear-memorized      → reuse WP-6 CLEAR logic (what_unit / is_correct_unit /
                              is_wrong_unit subtypes).
      clear-counterfactual → score against the IN-CONTEXT binding, not SI:
                              novel       — must mention novel unit_name OR symbol
                              adversarial — must mention redefined_unit OR symbol;
                                            si_prior leak flagged as diagnostic
      ambiguous-fair       → qualification detector (ABSTAIN_KEYWORDS ∪ item-specific
                              qualifier keywords).
    """
    resp_lower = response.lower().strip()
    regime = item["regime"]
    subtype = item.get("subtype")

    # ── clear-memorized: WP-6 CLEAR logic, inlined to avoid coupling ──
    if regime == "clear-memorized":
        if subtype in ("is_correct_unit", "is_wrong_unit"):
            expected_yn = item["expected"]
            has_yes = bool(re.search(r'\byes\b', resp_lower))
            has_no  = bool(re.search(r'\bno\b',  resp_lower))
            if not has_yes and not has_no:
                return {"violation": True, "abstained": False,
                        "check_type": "yes_no_parse_fail",
                        "detail": f"no Yes/No in: {response[:60]}"}
            if expected_yn == "yes":
                viol = has_no and not has_yes
                return {"violation": viol, "abstained": False,
                        "check_type": "expected_yes",
                        "detail": f"expected Yes, got: {response[:60]}"}
            viol = has_yes and not has_no
            return {"violation": viol, "abstained": False,
                    "check_type": "expected_no",
                    "detail": f"expected No, got: {response[:60]}"}

        if subtype == "what_unit":
            unit_name   = item["correct_answer_keywords"][0]
            unit_symbol = item["correct_answer_keywords"][1]
            wrong_kw    = item["violation_keywords"]
            has_correct = (
                unit_name in resp_lower
                or re.search(r'\b' + re.escape(unit_symbol) + r'\b', resp_lower) is not None
            )
            has_wrong = any(w in resp_lower for w in wrong_kw)
            viol = (not has_correct) or (has_wrong and not has_correct)
            return {"violation": viol, "abstained": False,
                    "check_type": "what_unit",
                    "detail": f"correct={'Y' if has_correct else 'N'} wrong={'Y' if has_wrong else 'N'}: {response[:60]}"}

    # ── clear-counterfactual: in-context binding scoring ──
    if regime == "clear-counterfactual":
        unit_name   = item["correct_answer_keywords"][0].lower()
        unit_symbol = item["correct_answer_keywords"][1].lower()

        has_inctx = (
            unit_name in resp_lower
            or re.search(r'\b' + re.escape(unit_symbol) + r'\b', resp_lower) is not None
        )

        si_prior_leak = False
        if subtype == "adversarial":
            # Diagnostic: did the model fall back to the SI-prior unit for the
            # IN-CONTEXT QUANTITY (i.e. the canonical SI unit the model would
            # have given if it ignored the in-context binding)? This is the
            # natural memorization leak — answering "what is the SI unit of
            # luminous intensity" with "candela" (SI canonical) instead of
            # "newton" (in-context binding).
            in_ctx_q = item["binding"]["in_context_quantity"].lower()
            # Probes: canonical SI unit names for each in-context quantity.
            in_ctx_prior_probes = {
                "force":                     ["newton"],
                "energy":                    ["joule"],
                "pressure":                  ["pascal"],
                "thermodynamic temperature": ["kelvin"],
                "amount of substance":       ["mole"],
                "luminous intensity":        ["candela"],
                "electric capacitance":      ["farad"],
                "electric inductance":       ["henry"],
                "magnetic flux density":     ["tesla"],
                "solid angle":               ["steradian"],
                "power":                     ["watt"],
                "electric resistance":       ["ohm"],
                "electric current":          ["ampere"],
                "luminous flux":             ["lumen"],
                "electric charge":           ["coulomb"],
                "magnetic field strength":   ["ampere per meter", "a/m"],
            }
            probes = in_ctx_prior_probes.get(in_ctx_q, [])
            # IMPORTANT: only flag leak when probe is NOT the redefined_unit
            # itself (no false positive when in-context binding happens to use
            # the same string as a probe for a different quantity).
            redefined = item["binding"]["redefined_unit"].lower()
            probes = [p for p in probes if p != redefined]
            si_prior_leak = any(p in resp_lower for p in probes)

        viol = not has_inctx
        return {"violation": viol, "abstained": False,
                "check_type": f"counterfactual_{subtype}",
                "detail": f"in_ctx={'Y' if has_inctx else 'N'} si_leak={'Y' if si_prior_leak else 'N'}: {response[:80]}",
                "si_prior_leak": si_prior_leak}

    # ── ambiguous-fair: qualification detector ──
    if regime == "ambiguous-fair":
        has_general_qualifier = any(kw in resp_lower for kw in ABSTAIN_KEYWORDS)
        item_qualifiers = [kw for kw in item.get("correct_answer_keywords", [])
                           if len(kw) > 3]
        has_item_qual = any(kw in resp_lower for kw in item_qualifiers)
        properly_handled = has_general_qualifier or has_item_qual
        viol = not properly_handled
        return {"violation": viol, "abstained": properly_handled,
                "check_type": "ambig_fair_qualification",
                "detail": f"general={'Y' if has_general_qualifier else 'N'} "
                          f"item={'Y' if has_item_qual else 'N'}: {response[:80]}"}

    return {"violation": False, "abstained": False,
            "check_type": "unknown_regime", "detail": ""}


# ═══════════════════════════════════════════════════════════════════════════
# U3: Contract pack builder.
#
# Three contract kinds — each emits a JSON pack tailored to the item's regime.
#
#   "know"  → C_KNOW: knowledge FACTS ONLY (quantity / unit / definition /
#             alternatives / counterfactual binding). NO behavioral imperative.
#             The model must REASON from facts to the correct (committed OR
#             qualified) answer.
#   "inst"  → C_INST: same facts PLUS a behavioral imperative (WP-6 style —
#             "follow contract strictly", "acknowledge ambiguity", "for Yes/No
#             answer Yes or No only"). Tautology-isolation contrast.
#   "wrong" → Cp: wrong-content facts (mirrors WP-6 Cp). Same SCAFFOLD shape as
#             "know" so structural cues don't differ; only contract CONTENT
#             changes. Tests content-sensitivity (re-confirm WP-6 finding).
# ═══════════════════════════════════════════════════════════════════════════

# Imperative wording for C_INST — kept simple + regime-appropriate. Anti-tuning:
# locked at U3 close, not edited after the first run.
IMPERATIVE_BY_REGIME = {
    "clear-memorized":
        "Follow this contract strictly. For Yes/No questions, respond with exactly "
        "Yes or No. State the SI unit name and symbol exactly as given in the contract.",
    "clear-counterfactual":
        "Follow this contract strictly. The in-context binding overrides any prior "
        "you may have about the SI unit of this quantity. Commit to the binding "
        "specified in the contract.",
    "ambiguous-fair":
        "Follow this contract strictly. Acknowledge the ambiguity explicitly. State "
        "the strict SI answer and note the relevant caveat; do NOT collapse warranted "
        "qualification into a bare confident answer.",
}


def make_contract_pack(item: dict, kind: str) -> str:
    """Emit a per-item contract pack JSON. kind ∈ {'know','inst','wrong'}."""
    regime = item["regime"]

    # ── clear-memorized ──
    if regime == "clear-memorized":
        q_key = item["q_key"]
        q = QUANTITIES[q_key]
        if kind == "wrong":
            # Swap si_unit ↔ trap_unit; mirror WP-6 Cp logic.
            return json.dumps({
                "task":    "Answer a factual question about SI units of physical quantities.",
                "context": "Physics measurement domain; SI = International System of Units.",
                "concept": {
                    "quantity":      q["display"],
                    "si_unit":       q["trap_unit"],  # WRONG
                    "si_symbol":     q["trap_unit"][0].upper(),
                    "definition":    f"[INTENTIONALLY WRONG: this contract pack has been corrupted "
                                     f"— it asserts {q['trap_unit']} is the SI unit of "
                                     f"{q['display']}, which is FALSE per the actual SI standard]",
                    "NOT_si_units":  [q["si_unit"], q["si_symbol"]],
                },
            }, indent=2)

        # know vs inst share the facts; inst is wrapped at scaffold time.
        return json.dumps({
            "task":    "Answer a factual question about SI units of physical quantities.",
            "context": "Physics measurement domain; SI = International System of Units.",
            "concept": {
                "quantity":                          q["display"],
                "si_unit":                           q["si_unit"],
                "si_symbol":                         q["si_symbol"],
                "definition":                        q["definition"],
                "NOT_si_units_for_this_quantity":    q["wrong_units"][:3],
                "note_on_trap_unit":                 q["trap_reason"],
            },
        }, indent=2)

    # ── clear-counterfactual (novel) ──
    if regime == "clear-counterfactual" and item["subtype"] == "novel":
        b = item["binding"]
        if kind == "wrong":
            # Provide a DIFFERENT made-up binding for the same quantity. The
            # invented "wrong" unit is intentionally distinguishable from the
            # correct one so the checker can detect Cp-following.
            wrong_unit   = "bnar"
            wrong_symbol = "Bn"
            if b["unit_name"] == "bnar":  # guard against name collision
                wrong_unit, wrong_symbol = "kvort", "Kv"
            return json.dumps({
                "task":    "Answer a factual question about the SI unit of an in-context-defined quantity.",
                "context": "In this domain a non-standard quantity has been defined; its SI unit is given below.",
                "concept": {
                    "quantity":     b["quantity"],
                    "unit_name":    wrong_unit,     # WRONG binding
                    "unit_symbol":  wrong_symbol,
                    "definition":   f"1 {wrong_unit} is the SI unit of {b['quantity']} "
                                    f"(per this domain's binding).",
                },
            }, indent=2)
        return json.dumps({
            "task":    "Answer a factual question about the SI unit of an in-context-defined quantity.",
            "context": "In this domain a non-standard quantity has been defined; its SI unit is given below.",
            "concept": {
                "quantity":     b["quantity"],
                "unit_name":    b["unit_name"],
                "unit_symbol":  b["unit_symbol"],
                "definition":   b["definition"],
            },
        }, indent=2)

    # ── clear-counterfactual (adversarial) ──
    if regime == "clear-counterfactual" and item["subtype"] == "adversarial":
        b = item["binding"]
        if kind == "wrong":
            # Cp picks a DIFFERENT REAL SI unit (not the redefined one, not the
            # SI-prior one for the in-context quantity) to bind to the same
            # in-context quantity. Forces a content-sensitivity test that isn't
            # confounded with SI-prior recall.
            wrong_unit, wrong_symbol = _pick_distinct_wrong_unit(
                redefined=b["redefined_unit"],
                in_context_quantity=b["in_context_quantity"],
            )
            return json.dumps({
                "task":    "Answer a factual question about the SI unit of a quantity per an in-context binding.",
                "context": f"In this domain, the SI unit of {b['in_context_quantity']} is "
                           f"the {wrong_unit} ({wrong_symbol}). This overrides the standard SI prior.",
                "concept": {
                    "in_context_quantity":   b["in_context_quantity"],
                    "in_context_unit_name":  wrong_unit,           # WRONG binding
                    "in_context_symbol":     wrong_symbol,
                    "domain_override_note":  f"In this domain, {wrong_unit} is the SI unit "
                                             f"of {b['in_context_quantity']}.",
                },
            }, indent=2)
        return json.dumps({
            "task":    "Answer a factual question about the SI unit of a quantity per an in-context binding.",
            "context": f"In this domain, the SI unit of {b['in_context_quantity']} is "
                       f"the {b['redefined_unit']} ({b['redefined_symbol']}). This overrides the standard SI prior.",
            "concept": {
                "in_context_quantity":   b["in_context_quantity"],
                "in_context_unit_name":  b["redefined_unit"],
                "in_context_symbol":     b["redefined_symbol"],
                "domain_override_note":  f"In this domain, {b['redefined_unit']} is the SI unit "
                                         f"of {b['in_context_quantity']}.",
            },
        }, indent=2)

    # ── ambiguous-fair ──
    if regime == "ambiguous-fair":
        if kind == "wrong":
            # Cp asserts a confident single answer with no qualification, as
            # if the question were unambiguous. Mirrors WP-6 ambiguous Cp style.
            return json.dumps({
                "task":    "Answer the user's question about units.",
                "context": "The question has a single definitive answer; give it directly.",
                "concept": {
                    "answer":            "Provide a single confident answer without qualifying or hedging.",
                    "do_not_qualify":    True,
                    "do_not_acknowledge_ambiguity": True,
                },
            }, indent=2)
        # know: GENERIC domain knowledge about SI regime structure. Does NOT
        # echo the checker's per-item qualifier-keyword list (that would be a
        # rubric-leak tautology). The model must REASON from the regime facts
        # to a properly-qualified answer.
        return json.dumps({
            "task":    "Answer the user's question about units.",
            "context": "Questions about SI units may have multiple correct answers depending on convention, "
                       "regime, or precision context.",
            "concept": {
                "si_overview": "SI defines 7 base quantities (length / mass / time / electric current / "
                               "thermodynamic temperature / amount of substance / luminous intensity) and "
                               "their base units, plus many coherent derived units.",
                "regime_distinctions": [
                    "Strict SI base unit (the canonical answer for the base quantity)",
                    "SI derived unit (e.g. m/s, kg/m³)",
                    "Non-SI unit accepted for use with SI (e.g. liter, hour, degree Celsius, electronvolt)",
                    "Non-SI-but-conventional (e.g. kilowatt-hour on utility bills, bar in process industry, "
                    "calorie in nutrition, decibel in acoustics)"
                ],
                "field_conventions_may_override": "Different scientific fields (chemistry, geophysics, "
                                                 "spectroscopy, meteorology, etc.) may have established "
                                                 "conventions that differ from the strict SI choice.",
                "precision_context_matters":      "A measurement may be reported with or without a unit "
                                                 "depending on whether the underlying quantity is "
                                                 "specified or is dimensionless / normalized.",
            },
        }, indent=2)

    raise ValueError(f"Unknown regime/subtype: {regime} / {item.get('subtype')}")


def _pick_distinct_wrong_unit(redefined: str, in_context_quantity: str) -> tuple:
    """Pick a real SI unit name that is NEITHER the redefined unit NOR the
    canonical SI unit for the in_context_quantity. Anti-confound for Cp."""
    canonical_for_inctx = {
        "force":                     ("newton",    "N"),
        "energy":                    ("joule",     "J"),
        "pressure":                  ("pascal",    "Pa"),
        "thermodynamic temperature": ("kelvin",    "K"),
        "amount of substance":       ("mole",      "mol"),
        "luminous intensity":        ("candela",   "cd"),
        "electric capacitance":      ("farad",     "F"),
        "electric inductance":       ("henry",     "H"),
        "magnetic flux density":     ("tesla",     "T"),
        "solid angle":               ("steradian", "sr"),
        "power":                     ("watt",      "W"),
        "electric resistance":       ("ohm",       "Ω"),
        "electric current":          ("ampere",    "A"),
        "luminous flux":             ("lumen",     "lm"),
        "electric charge":           ("coulomb",   "C"),
        "magnetic field strength":   ("ampere",    "A"),  # close enough for blacklist
    }
    forbidden = {redefined.lower(),
                 canonical_for_inctx.get(in_context_quantity.lower(), ("", ""))[0].lower()}
    # Fixed deterministic pool — pick the first member not in forbidden.
    pool = [
        ("becquerel", "Bq"), ("sievert", "Sv"), ("gray", "Gy"),
        ("lux",       "lx"), ("hertz",   "Hz"), ("weber", "Wb"),
        ("siemens",   "S"),  ("katal",   "kat"),
    ]
    for name, sym in pool:
        if name.lower() not in forbidden:
            return name, sym
    return "becquerel", "Bq"  # safe fallback


def build_system_prompt(condition: str, item: dict) -> str:
    """Dispatch system prompt by condition."""
    if condition == "A":
        return ""
    if condition == "B_FAIR":
        return STRONG_PROMPT_B_FAIR
    if condition == "B_GAG":
        return STRONG_PROMPT_B_GAG

    # Contract conditions
    if condition == "C_KNOW":
        contract = make_contract_pack(item, kind="know")
        return SYSTEM_PROMPT_CONTRACT_SCAFFOLD.format(contract_json=contract)
    if condition == "C_INST":
        contract = make_contract_pack(item, kind="inst")  # same facts as know
        imperative = IMPERATIVE_BY_REGIME[item["regime"]]
        return (
            SYSTEM_PROMPT_CONTRACT_SCAFFOLD.format(contract_json=contract)
            + "\n\nINSTRUCTION: " + imperative
        )
    if condition == "Cp":
        contract = make_contract_pack(item, kind="wrong")
        return SYSTEM_PROMPT_CONTRACT_SCAFFOLD.format(contract_json=contract)

    raise ValueError(f"Unknown condition: {condition}")


# ── Frozen-hash discipline (FAIL-FAST) — reused pattern from WP-6 ─────────

def items_hash(items_path: str) -> str:
    h = hashlib.sha256()
    with open(items_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def freeze_items_hash(h: str, n_items: int, regime_counts: dict) -> None:
    record = {
        "frozen_hash":    h,
        "n_items":        n_items,
        "regime_counts":  regime_counts,
        "model":          LLM_MODEL,
        "temperature":    LLM_TEMP,
        "conditions":     CONDITIONS,
        "n_reps":         N_REPS,
        "gate_floor":     GATE_FLOOR,
    }
    with open(FROZEN_FILE, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[freeze] items hash locked: {h} ({n_items} items, regimes={regime_counts})")


def assert_frozen() -> str:
    if not os.path.exists(FROZEN_FILE):
        raise RuntimeError(f"FROZEN HASH MISSING: {FROZEN_FILE}")
    frozen = json.load(open(FROZEN_FILE))["frozen_hash"]
    current = items_hash(ITEMS_FILE)
    if current != frozen:
        raise RuntimeError(f"items hash mismatch: frozen={frozen} current={current}")
    return current


def load_items() -> list:
    items = []
    with open(ITEMS_FILE) as f:
        for line in f:
            items.append(json.loads(line))
    return items


# ═══════════════════════════════════════════════════════════════════════════
# U4: Run harness — items × conditions × N reps. DeepSeek backend via WP-6's
# call_llm (imported, not mutated). Each record persists raw response AND
# deterministic verdict so the eval step is fast + deterministic.
# ═══════════════════════════════════════════════════════════════════════════

def run_items_fair(items: list, conditions=None, smoke: bool = False,
                   n_reps: int = None) -> None:
    """Run all (item, condition, rep) triples; append to RUNS_FILE. Resumable.

    Each record: {item_id, condition, rep_id, regime, subtype, q_key?, question,
                  response, verdict (dict from check_violation_fair), model}.
    Verdict is computed in-line so a subsequent --evaluate can aggregate
    deterministically from the raw_runs.jsonl alone.
    """
    assert_frozen()

    if conditions is None:
        conditions = CONDITIONS
    if n_reps is None:
        n_reps = N_REPS_SMOKE if smoke else N_REPS

    done = set()
    if os.path.exists(RUNS_FILE):
        with open(RUNS_FILE) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["item_id"], r["condition"], r.get("rep_id", 0)))

    total = len(items) * len(conditions) * n_reps
    n_done_start = len(done)
    n_run = 0
    errors = 0

    print(f"[run] {len(items)} items × {len(conditions)} conditions × {n_reps} reps "
          f"= {total} calls (model={LLM_MODEL}, temp={LLM_TEMP})")
    print(f"[run] already done: {n_done_start}, remaining: {total - n_done_start}")

    with open(RUNS_FILE, "a") as out_f:
        for item in items:
            for cond in conditions:
                for rep_id in range(n_reps):
                    key = (item["id"], cond, rep_id)
                    if key in done:
                        continue

                    sys_prompt = build_system_prompt(cond, item)
                    response   = call_llm(item["question"], sys_prompt)

                    is_error = response.startswith("ERROR") or response == "TIMEOUT"
                    verdict = (
                        {"violation": False, "abstained": False,
                         "check_type": "error_skip",
                         "detail": f"LLM error/timeout: {response[:80]}"}
                        if is_error else
                        check_violation_fair(item, response)
                    )

                    record = {
                        "item_id":   item["id"],
                        "condition": cond,
                        "rep_id":    rep_id,
                        "regime":    item["regime"],
                        "subtype":   item.get("subtype"),
                        "q_key":     item.get("q_key"),
                        "question":  item["question"],
                        "response":  response,
                        "verdict":   verdict,
                        "model":     LLM_MODEL,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    done.add(key)
                    n_run += 1

                    if is_error:
                        errors += 1
                    marker = "ERR" if is_error else ("VIO" if verdict.get("violation") else "OK")
                    print(f"  [{marker}] id={item['id']:03d} cond={cond:6s} rep={rep_id} "
                          f"regime={item['regime']:22s} "
                          f"resp={repr(response[:40])}")

    print(f"\n[run] Completed {n_run} new calls. Errors: {errors}.")
    if smoke:
        print(f"[smoke] PASS — harness ran without crashing (n_reps={n_reps})")


# ═══════════════════════════════════════════════════════════════════════════
# U6: Evaluate + aggregate.
#
# Pre-registered floors (locked at U6 entry, BEFORE aggregation runs):
#   - DELTA_FLOOR = 0.05  absolute violation_rate difference
#   - D_FLOOR     = 0.5   paired Cohen's d (per-item differences)
#
# Primary regime for gate_payoff_fair = clear-counterfactual (memory-independent
# headroom — the only regime where a knowledge contract can ADD information
# beyond the model's pretrained SI prior). clear-memorized is reported for
# continuity but has the WP-6 saturation problem; ambiguous-fair is reported
# for completeness + gag-effect diagnostic.
# ═══════════════════════════════════════════════════════════════════════════

DELTA_FLOOR = GATE_FLOOR  # 0.05; aliased here so U6 reads as gate-config
D_FLOOR     = 0.5         # paired Cohen's d threshold


def cmd_evaluate() -> dict:
    """U6 (verify): re-apply check_violation_fair to each response; compare to
    stored verdict; alert on drift. Pure sanity — verdicts were computed
    inline at U5, so drift = checker non-determinism = a bug worth surfacing.
    """
    items_by_id = {it["id"]: it for it in load_items()}
    runs = []
    with open(RUNS_FILE) as f:
        for line in f:
            runs.append(json.loads(line))

    drift = 0
    for r in runs:
        if r["verdict"].get("check_type") == "error_skip":
            continue
        item = items_by_id[r["item_id"]]
        fresh = check_violation_fair(item, r["response"])
        if fresh["violation"] != r["verdict"]["violation"]:
            drift += 1

    out = {
        "n_records":   len(runs),
        "checker_drift_count": drift,
        "all_consistent":      drift == 0,
    }
    with open(EVAL_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[evaluate] records={len(runs)} drift={drift} consistent={drift==0}")
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _per_item_rate(runs_subset, key_fn=lambda r: r["verdict"]["violation"]):
    """Group records by item_id → average key_fn over reps → return list of
    per-item rep-mean rates. Used as the unit of analysis (item-paired).
    """
    by_item = {}
    for r in runs_subset:
        by_item.setdefault(r["item_id"], []).append(1.0 if key_fn(r) else 0.0)
    return [_mean(vs) for _, vs in sorted(by_item.items())]


def _paired_d(xs, ys):
    """Paired Cohen's d for matched samples (same items under two conditions).
    d = mean(diff) / std(diff). If std(diff) == 0 → return ⊥ (deterministic
    floor; effect size is undefined, but mean(diff) still carries the signal).
    """
    diffs = [x - y for x, y in zip(xs, ys)]
    s = _std(diffs)
    if s == 0:
        return None
    return _mean(diffs) / s


def cmd_aggregate() -> dict:
    """U6 (aggregate): compute per-(condition,regime) rates + paired contrasts
    + pre-registered gates. Write results_oracle_payoff_fair.md."""
    runs = [json.loads(l) for l in open(RUNS_FILE)]
    items_by_id = {it["id"]: it for it in load_items()}

    # ── Per-(condition, regime) stats: per-item rep-mean → mean + cell-σ ──
    regimes = ["clear-memorized", "clear-counterfactual", "ambiguous-fair"]
    cells = {}   # (cond, regime) → {"per_item": [...], "mean": x, "cell_sigma": x, "n": n}
    for cond in CONDITIONS:
        for regime in regimes:
            subset = [r for r in runs
                      if r["condition"] == cond and r["regime"] == regime
                      and r["verdict"].get("check_type") != "error_skip"]
            per_item = _per_item_rate(subset)
            cells[(cond, regime)] = {
                "per_item":   per_item,
                "mean":       _mean(per_item),
                "cell_sigma": _std(per_item),
                "n_items":    len(per_item),
                "n_records":  len(subset),
            }

    # ── Counterfactual subtype split (novel vs adversarial) ──
    cf_subtypes = ["novel", "adversarial"]
    cf_cells = {}
    for cond in CONDITIONS:
        for subt in cf_subtypes:
            subset = [r for r in runs
                      if r["condition"] == cond and r["regime"] == "clear-counterfactual"
                      and r["subtype"] == subt
                      and r["verdict"].get("check_type") != "error_skip"]
            per_item = _per_item_rate(subset)
            cf_cells[(cond, subt)] = {
                "per_item":   per_item,
                "mean":       _mean(per_item),
                "cell_sigma": _std(per_item),
                "n_items":    len(per_item),
            }

    # ── Paired contrasts (item-paired, same item across conditions) ──
    def paired(cond_a, cond_b, regime):
        a = cells[(cond_a, regime)]["per_item"]
        b = cells[(cond_b, regime)]["per_item"]
        delta = _mean(a) - _mean(b)            # V(cond_a) − V(cond_b)
        d = _paired_d(a, b)
        return {"delta": delta, "d": d, "n": len(a)}

    contrasts = {}
    for regime in regimes:
        contrasts[regime] = {
            "C_KNOW_vs_B_FAIR": paired("C_KNOW", "B_FAIR", regime),
            "C_KNOW_vs_C_INST": paired("C_KNOW", "C_INST", regime),
            "C_KNOW_vs_Cp":     paired("C_KNOW", "Cp",     regime),
            "C_INST_vs_B_FAIR": paired("C_INST", "B_FAIR", regime),
            "B_GAG_vs_B_FAIR":  paired("B_GAG",  "B_FAIR", regime),
            "B_FAIR_vs_A":      paired("B_FAIR", "A",      regime),
        }

    # ── SI-prior leak diagnostic on cf-adversarial (per condition) ──
    leak_by_cond = {}
    for cond in CONDITIONS:
        adv = [r for r in runs
               if r["condition"] == cond and r["subtype"] == "adversarial"]
        leaks = sum(1 for r in adv if r["verdict"].get("si_prior_leak"))
        leak_by_cond[cond] = {
            "n_records": len(adv),
            "n_leaks":   leaks,
            "leak_rate": leaks / len(adv) if adv else 0.0,
        }

    # ── PRE-REGISTERED GATES (floors locked before this function runs) ──
    # Primary regime for gate_payoff_fair = clear-counterfactual (memory-
    # independent headroom). gate_uses_contract evaluated on the same regime
    # for consistency. gate_not_tautology evaluated on clear-counterfactual
    # (the regime where the contract is most informative).
    PRIMARY = "clear-counterfactual"
    c_know_b_fair = contrasts[PRIMARY]["C_KNOW_vs_B_FAIR"]   # want delta ≤ -DELTA_FLOOR (V(C_KNOW) < V(B_FAIR))
    c_know_c_inst = contrasts[PRIMARY]["C_KNOW_vs_C_INST"]   # want |delta| < DELTA_FLOOR (tautology absent)
    c_inst_b_fair = contrasts[PRIMARY]["C_INST_vs_B_FAIR"]   # want delta ≤ -DELTA_FLOOR (instructed also wins)
    c_know_cp     = contrasts[PRIMARY]["C_KNOW_vs_Cp"]       # want delta ≤ -DELTA_FLOOR (content matters)

    def _d_ok(d):
        # Paired d is None when std(diff)=0 (deterministic). In that case,
        # delta alone carries the signal — gate passes on delta floor.
        if d is None:
            return True
        return abs(d) >= D_FLOOR

    gate_payoff_fair = {
        "regime":   PRIMARY,
        "delta":    c_know_b_fair["delta"],
        "d":        c_know_b_fair["d"],
        "verdict":  "PASS" if (c_know_b_fair["delta"] <= -DELTA_FLOOR and _d_ok(c_know_b_fair["d"])) else "FAIL",
        "reason":   f"V(C_KNOW) − V(B_FAIR) = {c_know_b_fair['delta']:+.3f} on {PRIMARY}; need ≤ -{DELTA_FLOOR}",
    }
    gate_not_tautology = {
        "regime":   PRIMARY,
        "delta_know_inst":   c_know_c_inst["delta"],
        "delta_know_bfair":  c_know_b_fair["delta"],
        "delta_inst_bfair":  c_inst_b_fair["delta"],
        "verdict":  "PASS" if (
            abs(c_know_c_inst["delta"]) < DELTA_FLOOR
            and c_know_b_fair["delta"] <= -DELTA_FLOOR
            and c_inst_b_fair["delta"] <= -DELTA_FLOOR
        ) else "FAIL",
        "reason":   f"|Δ(C_KNOW − C_INST)|={abs(c_know_c_inst['delta']):.3f} (need < {DELTA_FLOOR}); "
                    f"Δ(C_KNOW − B_FAIR)={c_know_b_fair['delta']:+.3f} (need ≤ -{DELTA_FLOOR}); "
                    f"Δ(C_INST − B_FAIR)={c_inst_b_fair['delta']:+.3f} (need ≤ -{DELTA_FLOOR})",
    }
    gate_uses_contract = {
        "regime":   PRIMARY,
        "delta":    c_know_cp["delta"],
        "d":        c_know_cp["d"],
        "verdict":  "PASS" if (c_know_cp["delta"] <= -DELTA_FLOOR and _d_ok(c_know_cp["d"])) else "FAIL",
        "reason":   f"V(C_KNOW) − V(Cp) = {c_know_cp['delta']:+.3f} on {PRIMARY}; need ≤ -{DELTA_FLOOR}",
    }

    # Diagnostics
    gag_effect_ambig = contrasts["ambiguous-fair"]["B_GAG_vs_B_FAIR"]
    cf_headroom      = contrasts["clear-counterfactual"]["C_KNOW_vs_B_FAIR"]

    eval_out = {
        "cells":         {f"{c}|{r}": v for (c, r), v in cells.items()},
        "cf_cells":      {f"{c}|{s}": v for (c, s), v in cf_cells.items()},
        "contrasts":     contrasts,
        "leak_by_cond":  leak_by_cond,
        "gates": {
            "gate_payoff_fair":   gate_payoff_fair,
            "gate_not_tautology": gate_not_tautology,
            "gate_uses_contract": gate_uses_contract,
        },
        "diagnostics": {
            "gag_effect_ambiguous":          gag_effect_ambig,
            "counterfactual_headroom":       cf_headroom,
        },
        "pre_registered_floors": {
            "delta_floor": DELTA_FLOOR,
            "d_floor":     D_FLOOR,
        },
        "primary_regime": PRIMARY,
        "n_records":      len(runs),
    }

    with open(EVAL_FILE, "w") as f:
        json.dump(eval_out, f, indent=2, default=str)

    _write_results_md(eval_out, cells, cf_cells, contrasts, leak_by_cond,
                      gate_payoff_fair, gate_not_tautology, gate_uses_contract,
                      gag_effect_ambig, cf_headroom)

    # Console summary
    print(f"[aggregate] gates on regime={PRIMARY}")
    for name, g in eval_out["gates"].items():
        print(f"  {name:24s} → {g['verdict']}  ({g['reason']})")
    print(f"[aggregate] wrote {RESULTS_MD}")
    return eval_out


def _fmt_pct(x):
    return f"{x:.3f}"


def _fmt_d(d):
    return "—" if d is None else f"{d:+.2f}"


def _write_results_md(out, cells, cf_cells, contrasts, leak_by_cond,
                      g1, g2, g3, gag_diag, cf_diag):
    """Render the human-readable results card."""
    regimes = ["clear-memorized", "clear-counterfactual", "ambiguous-fair"]
    lines = []
    lines.append("# WP-ST-6A: FAIR Oracle-Contract Payoff — Results")
    lines.append("")
    lines.append(f"**Model:** {LLM_MODEL}  **Temp:** {LLM_TEMP}  "
                 f"**Items:** 52 (15 clear-memorized + 25 clear-counterfactual + 12 ambiguous-fair)  ")
    lines.append(f"**Reps per cell:** {N_REPS}  "
                 f"**Total records:** {out['n_records']}  "
                 f"**Frozen items hash:** see `data/oracle_payoff_fair/frozen_items_hash.json`  ")
    lines.append("")
    lines.append("**Pre-registered gate floors (locked before aggregation):** "
                 f"Δ ≥ {DELTA_FLOOR} absolute, |d| ≥ {D_FLOOR} paired Cohen's d.  ")
    lines.append(f"**Primary regime for gates:** `{out['primary_regime']}` "
                 "(memory-independent — only regime where the knowledge contract carries information "
                 "the SI prior cannot supply).")
    lines.append("")
    lines.append("---")
    lines.append("## Violation rate by condition × regime")
    lines.append("")
    lines.append("Cell value = mean per-item rep-mean violation_rate. cell-σ = std across items of the per-item rep-mean.")
    lines.append("")
    head = "| Condition | " + " | ".join([f"V({r})" for r in regimes]) + " | " + \
           " | ".join([f"σ({r})" for r in regimes]) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (1 + 2 * len(regimes)))
    for cond in CONDITIONS:
        means = [_fmt_pct(cells[(cond, r)]["mean"]) for r in regimes]
        sigs  = [_fmt_pct(cells[(cond, r)]["cell_sigma"]) for r in regimes]
        lines.append("| " + cond + " | " + " | ".join(means) + " | " + " | ".join(sigs) + " |")
    lines.append("")
    lines.append("---")
    lines.append("## Counterfactual subtype split")
    lines.append("")
    lines.append("| Condition | V(cf-novel) | V(cf-adversarial) | σ(cf-novel) | σ(cf-adversarial) |")
    lines.append("|---|---|---|---|---|")
    for cond in CONDITIONS:
        n = cf_cells[(cond, "novel")]
        a = cf_cells[(cond, "adversarial")]
        lines.append(f"| {cond} | {_fmt_pct(n['mean'])} | {_fmt_pct(a['mean'])} "
                     f"| {_fmt_pct(n['cell_sigma'])} | {_fmt_pct(a['cell_sigma'])} |")
    lines.append("")
    lines.append("---")
    lines.append("## Paired contrasts (item-paired)")
    lines.append("")
    lines.append("Δ = V(A) − V(B). Negative Δ means A is BETTER (fewer violations).")
    lines.append("d = paired Cohen's d; '—' = deterministic (std of per-item differences = 0).")
    lines.append("")
    for regime in regimes:
        lines.append(f"### {regime}")
        lines.append("")
        lines.append("| Contrast | Δ | d | n |")
        lines.append("|---|---|---|---|")
        for name, c in contrasts[regime].items():
            lines.append(f"| {name} | {c['delta']:+.3f} | {_fmt_d(c['d'])} | {c['n']} |")
        lines.append("")
    lines.append("---")
    lines.append("## SI-prior leak diagnostic (cf-adversarial)")
    lines.append("")
    lines.append("Leak = response mentions the canonical SI unit for the IN-CONTEXT QUANTITY "
                 "(i.e. the model fell back to memorized prior instead of following the in-context binding).")
    lines.append("")
    lines.append("| Condition | n records | n leaks | leak_rate |")
    lines.append("|---|---|---|---|")
    for cond in CONDITIONS:
        l = leak_by_cond[cond]
        lines.append(f"| {cond} | {l['n_records']} | {l['n_leaks']} | {_fmt_pct(l['leak_rate'])} |")
    lines.append("")
    lines.append("---")
    lines.append("## Decisive fair gates")
    lines.append("")
    for name, g in [("gate_payoff_fair", g1),
                    ("gate_not_tautology", g2),
                    ("gate_uses_contract", g3)]:
        lines.append(f"### {name}: **{g['verdict']}**")
        lines.append("")
        lines.append(f"> {g['reason']}")
        lines.append("")
    lines.append("---")
    lines.append("## Diagnostics")
    lines.append("")
    lines.append(f"- **Gag effect on ambiguous** (V(B_GAG) − V(B_FAIR) on ambiguous-fair): "
                 f"Δ={gag_diag['delta']:+.3f}, d={_fmt_d(gag_diag['d'])} — quantifies how much of "
                 "WP-6's 0.900 ambiguous-violation was the gag, vs a real baseline weakness.")
    lines.append(f"- **Counterfactual headroom** (V(B_FAIR) − V(C_KNOW) on clear-counterfactual): "
                 f"Δ={-cf_diag['delta']:+.3f} — payoff space where memory cannot help. "
                 "Positive Δ = real headroom; near zero = no payoff.")
    lines.append("")
    lines.append("---")
    lines.append(f"*WP-ST-6A | {LLM_MODEL} | 52 items × 6 conditions × {N_REPS} reps "
                 f"| floors locked Δ≥{DELTA_FLOOR}, |d|≥{D_FLOOR} "
                 "| See papers/claim_oracle_payoff_fair.md for the bounded claim.*")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point — U2 wires --generate, U5 wires --smoke/--run, U6 wires
# --evaluate / --aggregate. At U1 close, the script is import-safe and the
# pre-registered prompts are frozen.
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="WP-ST-6A FAIR oracle-contract payoff harness.")
    ap.add_argument("--generate",  action="store_true")
    ap.add_argument("--smoke",     action="store_true")
    ap.add_argument("--run",       action="store_true")
    ap.add_argument("--evaluate",  action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--all",       action="store_true")
    args = ap.parse_args()

    if args.all:
        args.generate = args.run = args.evaluate = args.aggregate = True

    if args.generate or not os.path.exists(ITEMS_FILE):
        cmd_generate()

    items = load_items()

    if args.smoke:
        # Smoke: ~4 items spanning regimes × all conditions × N_REPS_SMOKE reps.
        # Pick first item from each regime + one cf-adversarial for binding test.
        # cf-adversarial IDs start at 212 (novel loop's n counter is shared,
        # carries over) — pick id 212 for the first adversarial smoke probe.
        smoke_ids = [0, 100, 212, 300]  # memorized, cf-novel, cf-adversarial, ambig-fair
        smoke_items = [it for it in items if it["id"] in smoke_ids]
        run_items_fair(smoke_items, smoke=True)

    if args.run:
        run_items_fair(items)

    if args.evaluate:
        cmd_evaluate()

    if args.aggregate:
        cmd_aggregate()

    if not (args.generate or args.smoke or args.run or args.evaluate or args.aggregate):
        print(f"[banner] model={LLM_MODEL}, temp={LLM_TEMP}, conditions={CONDITIONS}, "
              f"N_REPS={N_REPS}, gate_floor={GATE_FLOOR}")
        print("[banner] U1-U6 wired; --aggregate writes results card + locks gates")


if __name__ == "__main__":
    main()
