"""WP-ST-6: Oracle-contract payoff — does a PERFECT contract cut LLM boundary violations
vs a strong prompt?

Conditions (IDENTICAL items, only system context changes):
  A:  naked LLM (no system prompt)
  B:  LLM + strong static system prompt (well-engineered guardrail, no per-item contract)
  C:  LLM + CORRECT oracle contract pack (per-item task/context/concept)
  C': LLM + WRONG oracle contract pack (corrupted unit — content-sensitivity control)

Gates:
  gate_payoff:      violation_rate(C) < violation_rate(B)  by ≥ 0.05 absolute
  gate_uses_contract: violation_rate(C) < violation_rate(C') by ≥ 0.05 absolute

Usage:
  python scripts/oracle_payoff.py --generate   # U1: build items + freeze hash
  python scripts/oracle_payoff.py --run        # U2+U3: run all conditions via claude CLI
  python scripts/oracle_payoff.py --smoke      # U2+U3: smoke (3 items only)
  python scripts/oracle_payoff.py --evaluate   # U4: compute violation_rate etc.
  python scripts/oracle_payoff.py --aggregate  # U5: gates + results_oracle_payoff.md
  python scripts/oracle_payoff.py --all        # generate + run + evaluate + aggregate
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


# ── .env loader (stdlib only — no python-dotenv) ──────────────────────────
def _load_env(path: str = ".env") -> None:
    """Load KEY=VALUE lines from repo-root .env into os.environ (setdefault)."""
    p = Path(__file__).resolve().parent.parent / path
    if not p.exists():
        return
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()


# ── Constants ──────────────────────────────────────────────────────────────
DATA_DIR      = "data/oracle_payoff"
PAPERS_DIR    = "papers"
ITEMS_FILE    = os.path.join(DATA_DIR, "items.jsonl")
FROZEN_FILE   = os.path.join(DATA_DIR, "frozen_items_hash.json")
RUNS_FILE     = os.path.join(DATA_DIR, "raw_runs.jsonl")
EVAL_FILE     = os.path.join(DATA_DIR, "eval_results.json")
RESULTS_MD    = os.path.join(PAPERS_DIR, "results_oracle_payoff.md")

CONDITIONS    = ["A", "B", "C", "Cp"]   # Cp = C' (wrong contract)
LLM_MODEL     = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_TEMP      = 0                         # enforced at call site (DeepSeek supports it)

N_REPS        = 3                         # per-cell reps for stability (full run)
N_REPS_SMOKE  = 2                         # per-cell reps for smoke validation

GATE_FLOOR    = 0.05                      # absolute Δ violation_rate threshold

# ── U1: Science micro-domain ───────────────────────────────────────────────

QUANTITIES = {
    "force": {
        "display": "force",
        "si_unit": "newton", "si_symbol": "N",
        "definition": "1 N = 1 kg·m/s²",
        "wrong_units": ["joule", "watt", "pascal", "kilogram", "calorie"],
        "trap_unit": "joule",
        "trap_reason": "Joule is the unit of energy, not force",
    },
    "energy": {
        "display": "energy",
        "si_unit": "joule", "si_symbol": "J",
        "definition": "1 J = 1 kg·m²/s²",
        "wrong_units": ["newton", "watt", "calorie", "electron-volt"],
        "trap_unit": "calorie",
        "trap_reason": "Calorie is an older non-SI unit; Joule is the SI unit",
    },
    "pressure": {
        "display": "pressure",
        "si_unit": "pascal", "si_symbol": "Pa",
        "definition": "1 Pa = 1 N/m²",
        "wrong_units": ["bar", "atmosphere", "torr", "newton"],
        "trap_unit": "bar",
        "trap_reason": "Bar is commonly used but not the SI unit; Pascal is",
    },
    "temperature": {
        "display": "thermodynamic temperature",
        "si_unit": "kelvin", "si_symbol": "K",
        "definition": "0 K = −273.15 °C; the SI base unit for temperature",
        "wrong_units": ["celsius", "fahrenheit", "centigrade"],
        "trap_unit": "celsius",
        "trap_reason": "Celsius is widely used but Kelvin is the SI base unit",
        "ambiguous_note": "For the strict SI base unit, Kelvin (K) is correct; Celsius (°C) is permitted for everyday science but is NOT the SI base unit",
    },
    "amount_of_substance": {
        "display": "amount of substance",
        "si_unit": "mole", "si_symbol": "mol",
        "definition": "1 mol ≈ 6.022×10²³ entities (Avogadro number)",
        "wrong_units": ["gram", "kilogram", "liter", "molecule", "dalton"],
        "trap_unit": "gram",
        "trap_reason": "Gram is a unit of mass; mole is the unit of amount of substance",
    },
    "luminous_intensity": {
        "display": "luminous intensity",
        "si_unit": "candela", "si_symbol": "cd",
        "definition": "SI base unit; luminous power per solid angle",
        "wrong_units": ["lumen", "lux", "watt", "nit"],
        "trap_unit": "lumen",
        "trap_reason": "Lumen is luminous flux (total), not intensity (per solid angle)",
    },
    "capacitance": {
        "display": "electric capacitance",
        "si_unit": "farad", "si_symbol": "F",
        "definition": "1 F = 1 C/V",
        "wrong_units": ["coulomb", "henry", "ohm", "volt"],
        "trap_unit": "coulomb",
        "trap_reason": "Coulomb is electric charge; Farad is capacitance",
    },
    "inductance": {
        "display": "electric inductance",
        "si_unit": "henry", "si_symbol": "H",
        "definition": "1 H = 1 V·s/A",
        "wrong_units": ["ohm", "farad", "tesla", "weber"],
        "trap_unit": "ohm",
        "trap_reason": "Ohm is resistance; Henry is inductance",
    },
    "magnetic_flux_density": {
        "display": "magnetic flux density",
        "si_unit": "tesla", "si_symbol": "T",
        "definition": "1 T = 1 Wb/m²",
        "wrong_units": ["gauss", "weber", "henry", "ampere_per_meter"],
        "trap_unit": "gauss",
        "trap_reason": "Gauss is the CGS unit; Tesla is the SI unit",
    },
    "solid_angle": {
        "display": "solid angle",
        "si_unit": "steradian", "si_symbol": "sr",
        "definition": "SI supplementary unit; 3D angular measure",
        "wrong_units": ["radian", "degree", "arcminute"],
        "trap_unit": "radian",
        "trap_reason": "Radian is the unit of plane (2D) angle; Steradian is for solid (3D) angle",
    },
}

STRONG_PROMPT_B = """You are a precise physics and measurement expert with thorough knowledge of the International System of Units (SI).

Rules:
1. Always identify the correct SI unit for a physical quantity.
2. Distinguish carefully between related but different quantities (e.g., energy vs power vs force).
3. For Yes/No questions, respond with ONLY the word "Yes" or "No" with no additional explanation.
4. For open-ended questions, state the SI unit name and its symbol.
5. If a question is genuinely ambiguous or context-dependent, explicitly acknowledge the ambiguity.

Apply careful, precise reasoning about measurement and units."""


def make_oracle_contract(q_key: str, q: dict, wrong: bool = False) -> str:
    """Build oracle contract JSON string for condition C or C'."""
    if wrong:
        # Swap si_unit and trap_unit to create a wrong contract
        contract = {
            "task": "Answer a factual question about SI units of physical quantities",
            "context": "Physics measurement domain; SI = International System of Units",
            "concept": {
                "quantity": q["display"],
                "si_unit": q["trap_unit"],          # WRONG: trap unit as if it's SI
                "si_symbol": q["trap_unit"][0].upper(),
                "definition": f"[intentionally wrong: this is the {q['trap_unit']} definition]",
                "NOT_si_units": [q["si_unit"], q["si_symbol"]],  # WRONG: lists correct unit as wrong
            },
            "constraints": [
                f"The SI unit of {q['display']} is {q['trap_unit']} (WRONG CONTRACT)",
                f"{q['si_unit']} is NOT the SI unit of {q['display']} (WRONG CONTRACT)",
                "For Yes/No questions, respond with exactly Yes or No",
            ],
        }
    else:
        # Correct oracle contract
        contract = {
            "task": "Answer a factual question about SI units of physical quantities",
            "context": "Physics measurement domain; SI = International System of Units",
            "concept": {
                "quantity": q["display"],
                "si_unit": q["si_unit"],
                "si_symbol": q["si_symbol"],
                "definition": q["definition"],
                "NOT_si_units_for_this_quantity": q["wrong_units"][:3],
                "trap_note": q["trap_reason"],
            },
            "constraints": [
                f"The SI unit of {q['display']} is {q['si_unit']} (symbol: {q['si_symbol']})",
                f"{q['trap_unit']} is NOT the SI unit of {q['display']}: {q['trap_reason']}",
                "For Yes/No questions, respond with exactly Yes or No",
            ],
        }
    return json.dumps(contract, indent=2)


def make_ambiguous_contract(q_key: str, q: dict, wrong: bool = False) -> str:
    """Contract for ambiguous items."""
    note = q.get("ambiguous_note", f"This question about {q['display']} requires qualification")
    if wrong:
        contract = {
            "task": "Answer a question about units",
            "constraints": [
                "Give a single definitive answer without any qualification",
                f"The answer is simply: {q['trap_unit']}",  # Wrong and overconfident
            ],
        }
    else:
        contract = {
            "task": "Answer a question about units that requires nuance",
            "context": f"The question about {q['display']} units is ambiguous or context-dependent",
            "constraints": [
                note,
                "Acknowledge the ambiguity explicitly",
                "State what the strict SI unit is, then note when alternatives may be appropriate",
                "Do NOT give a single confident answer without qualification",
            ],
        }
    return json.dumps(contract, indent=2)


def generate_items():
    """U1: Generate CLEAR and AMBIGUOUS items for the oracle payoff experiment."""
    items = []
    item_id = 0

    for q_key, q in QUANTITIES.items():
        # ── 4 CLEAR items per quantity ──
        # Type 1: what_unit
        items.append({
            "id": item_id,
            "q_key": q_key,
            "item_type": "what_unit",
            "regime": "clear",
            "question": f"What is the SI unit of {q['display']}? State the unit name and symbol.",
            "correct_answer_keywords": [q["si_unit"], q["si_symbol"].lower()],
            "violation_keywords": q["wrong_units"][:3],
            "expected": "clear_commit",
        })
        item_id += 1

        # Type 2: is_correct_unit (should answer Yes)
        items.append({
            "id": item_id,
            "q_key": q_key,
            "item_type": "is_correct_unit",
            "regime": "clear",
            "question": f"Is {q['si_unit']} ({q['si_symbol']}) the SI unit of {q['display']}? Answer Yes or No only.",
            "correct_answer_keywords": ["yes"],
            "violation_keywords": ["no"],
            "expected": "yes",
        })
        item_id += 1

        # Type 3: is_wrong_unit (should answer No — the trap unit)
        items.append({
            "id": item_id,
            "q_key": q_key,
            "item_type": "is_wrong_unit",
            "regime": "clear",
            "question": f"Is {q['trap_unit']} the SI unit of {q['display']}? Answer Yes or No only.",
            "correct_answer_keywords": ["no"],
            "violation_keywords": ["yes"],
            "expected": "no",
            "trap_unit": q["trap_unit"],
            "trap_reason": q["trap_reason"],
        })
        item_id += 1

        # Type 4: what_quantity
        items.append({
            "id": item_id,
            "q_key": q_key,
            "item_type": "what_quantity",
            "regime": "clear",
            "question": f"In the SI system, what physical quantity is measured in {q['si_unit']} ({q['si_symbol']})?",
            "correct_answer_keywords": [w for w in q["display"].split()],
            "violation_keywords": [],   # harder to enumerate; use keyword check
            "expected": "clear_commit",
        })
        item_id += 1

    # ── 10 AMBIGUOUS items (1 per quantity, focused on the ambiguity) ──
    AMBIGUOUS_ITEMS = [
        ("force",              "Is pound-force a valid unit for measuring force?",
         ["yes", "valid", "pound", "imperial", "us"],
         "qualified"),
        ("energy",             "Is calorie a correct unit for measuring energy?",
         ["yes", "calorie", "correct", "food", "nutrition", "however", "but"],
         "qualified"),
        ("pressure",           "Is bar an acceptable unit for pressure in scientific work?",
         ["yes", "bar", "acceptable", "widely", "commonly", "however"],
         "qualified"),
        ("temperature",        "Is degrees Celsius an acceptable unit for temperature?",
         ["yes", "celsius", "acceptable", "everyday", "widely", "however", "but"],
         "qualified"),
        ("amount_of_substance", "What unit should a chemist use for amount of substance?",
         ["mole", "mol", "however", "depends", "context"],
         "qualified"),
        ("luminous_intensity",  "Is lumen a unit of luminous intensity?",
         ["no", "flux", "lumen", "different", "luminous flux", "not intensity", "however"],
         "qualified"),
        ("capacitance",         "Can capacitance be expressed in terms of coulombs and volts?",
         ["yes", "coulombs", "volts", "farad", "c/v", "however", "relation"],
         "qualified"),
        ("inductance",          "Are ohms and henries related in circuit analysis?",
         ["yes", "related", "henry", "ohm", "however", "different", "impedance"],
         "qualified"),
        ("magnetic_flux_density", "Is gauss still used as a unit of magnetic flux density?",
         ["yes", "gauss", "cgs", "still used", "however", "si", "tesla"],
         "qualified"),
        ("solid_angle",         "Can solid angle be measured in square radians?",
         ["yes", "steradian", "square", "radian", "rad", "however", "sr", "not"],
         "qualified"),
    ]

    for q_key, question, keywords, expected in AMBIGUOUS_ITEMS:
        items.append({
            "id": item_id,
            "q_key": q_key,
            "item_type": "ambiguous",
            "regime": "ambiguous",
            "question": question,
            "correct_answer_keywords": keywords,
            "violation_keywords": [],
            "expected": expected,   # "qualified" = should hedge/qualify
        })
        item_id += 1

    return items


# ── U1: Hash + freeze ─────────────────────────────────────────────────────

def items_hash(items_path):
    h = hashlib.sha256()
    with open(items_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def freeze_items_hash(h, n_items):
    record = {"frozen_hash": h, "n_items": n_items, "model": LLM_MODEL}
    with open(FROZEN_FILE, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[freeze] items hash locked: {h} ({n_items} items)")


def assert_frozen():
    if not os.path.exists(FROZEN_FILE):
        raise RuntimeError(f"FROZEN HASH MISSING: {FROZEN_FILE}")
    frozen = json.load(open(FROZEN_FILE))["frozen_hash"]
    current = items_hash(ITEMS_FILE)
    if current != frozen:
        raise RuntimeError(
            f"items hash mismatch: frozen={frozen} current={current}")
    return current


def cmd_generate():
    """U1: Generate items, save to JSONL, freeze hash."""
    os.makedirs(DATA_DIR, exist_ok=True)
    items = generate_items()
    with open(ITEMS_FILE, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    h = items_hash(ITEMS_FILE)
    freeze_items_hash(h, len(items))
    n_clear    = sum(1 for it in items if it["regime"] == "clear")
    n_ambig    = sum(1 for it in items if it["regime"] == "ambiguous")
    print(f"Generated {len(items)} items: {n_clear} CLEAR + {n_ambig} AMBIGUOUS")
    return items


def load_items():
    items = []
    with open(ITEMS_FILE) as f:
        for line in f:
            items.append(json.loads(line))
    return items


# ── U2: LLM backend ───────────────────────────────────────────────────────

def parse_llm_output(raw: str) -> str:
    """Extract clean text from claude CLI output (strips GEM^2_MSG wrapper if present)."""
    raw = raw.strip()
    m = re.search(r'=:GEM2_MSG\s*(.*?)\s*</GEM\^2_MSG>', raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw


def call_llm(question: str, system_prompt: str = "", timeout: int = 60,
             retries: int = 3) -> str:
    """Call DeepSeek /chat/completions (OpenAI-compatible) at temperature=0.
    Returns clean response text, or 'ERROR: ...' / 'TIMEOUT' on failure."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return "ERROR: DEEPSEEK_API_KEY missing from env"

    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
           [{"role": "user", "content": question}]
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": msgs,
        "temperature": LLM_TEMP,
    }).encode("utf-8")

    req = urllib.request.Request(
        DEEPSEEK_BASE.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        },
    )

    last_err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
                content = payload["choices"][0]["message"]["content"]
                return parse_llm_output(content)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode(errors="replace")[:200]
            except Exception:
                err_body = ""
            last_err = f"HTTP {e.code}: {err_body}"
            retryable = e.code in (429, 500, 502, 503, 504)
            if retryable and i < retries - 1:
                time.sleep(2 ** i)
                continue
            return f"ERROR: {last_err}"
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = f"NET: {str(e)[:120]}"
            if i < retries - 1:
                time.sleep(2 ** i)
                continue
            return "TIMEOUT" if "timed out" in str(e).lower() else f"ERROR: {last_err}"
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return f"ERROR: parse_fail: {type(e).__name__}: {str(e)[:80]}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {str(e)[:120]}"
    return f"ERROR: {last_err or 'max_retries'}"


def build_system_prompt(condition: str, item: dict) -> str:
    """Build system prompt for a given condition and item."""
    if condition == "A":
        return ""

    if condition == "B":
        return STRONG_PROMPT_B

    if condition == "C":
        # Correct oracle contract
        q = QUANTITIES[item["q_key"]]
        if item["regime"] == "ambiguous":
            contract = make_ambiguous_contract(item["q_key"], q, wrong=False)
        else:
            contract = make_oracle_contract(item["q_key"], q, wrong=False)
        return (
            "You are a precise physics measurement assistant. "
            "A task contract has been provided for this specific question. "
            "Follow the contract strictly.\n\n"
            f"CONTRACT:\n{contract}"
        )

    if condition == "Cp":
        # Wrong oracle contract (C')
        q = QUANTITIES[item["q_key"]]
        if item["regime"] == "ambiguous":
            contract = make_ambiguous_contract(item["q_key"], q, wrong=True)
        else:
            contract = make_oracle_contract(item["q_key"], q, wrong=True)
        return (
            "You are a precise physics measurement assistant. "
            "A task contract has been provided for this specific question. "
            "Follow the contract strictly.\n\n"
            f"CONTRACT:\n{contract}"
        )

    raise ValueError(f"Unknown condition: {condition}")


# ── U3: Run conditions + capture verdicts ─────────────────────────────────

def run_items(items, conditions=None, smoke=False, n_reps=None):
    """Run all (item, condition, rep) triples. Append to RUNS_FILE. Resumable."""
    assert_frozen()

    if conditions is None:
        conditions = CONDITIONS
    if n_reps is None:
        n_reps = N_REPS_SMOKE if smoke else N_REPS

    # Load already-done (item_id, condition, rep_id) triples
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

                    record = {
                        "item_id":    item["id"],
                        "condition":  cond,
                        "rep_id":     rep_id,
                        "q_key":      item["q_key"],
                        "item_type":  item["item_type"],
                        "regime":     item["regime"],
                        "question":   item["question"],
                        "response":   response,
                        "model":      LLM_MODEL,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    done.add(key)
                    n_run += 1

                    is_error = response.startswith("ERROR") or response == "TIMEOUT"
                    if is_error:
                        errors += 1
                    marker = "ERR" if is_error else "OK"
                    print(f"  [{marker}] item={item['id']:02d} cond={cond} rep={rep_id} "
                          f"type={item['item_type']:<16} "
                          f"response={repr(response[:40])}")

    print(f"\n[run] Completed {n_run} new calls. Errors: {errors}.")
    if smoke:
        print(f"[smoke] PASS — harness ran without crashing (n_reps={n_reps})")


# ── U4: Violation checker + evaluation ────────────────────────────────────

ABSTAIN_KEYWORDS = [
    "depends", "context", "conventionally", "both", "either", "can also",
    "in practice", "however", "although", "but", "not strictly", "si requires",
    "strictly si", "strictly speaking", "technically", "while si", "more precisely",
    "it's worth", "it is worth", "note that", "keep in mind", "important",
    "qualification", "qualify", "nuance", "not the si", "is not the si",
    "si base unit", "permitted", "widely used", "commonly used", "everyday",
    "ambiguous", "under-specified", "clarif", "acceptable in", "valid in",
    "for everyday", "in science", "in physics", "not the official",
]


def check_violation(item: dict, response: str) -> dict:
    """
    Deterministic violation checker.

    Returns:
      {"violation": bool, "abstained": bool, "check_type": str, "detail": str}
    """
    resp_lower = response.lower().strip()

    if item["regime"] == "clear":
        item_type = item["item_type"]

        # Yes/No items
        if item_type in ("is_correct_unit", "is_wrong_unit"):
            expected_yn = item["expected"]   # "yes" or "no"
            # Find "yes"/"no" in response (first occurrence, case-insensitive)
            has_yes = bool(re.search(r'\byes\b', resp_lower))
            has_no  = bool(re.search(r'\bno\b', resp_lower))

            if not has_yes and not has_no:
                # Abstained or gave non-binary answer
                return {"violation": True,
                        "abstained": False,
                        "check_type": "yes_no_parse_fail",
                        "detail": f"No Yes/No found in: {response[:60]}"}

            if expected_yn == "yes":
                violation = has_no and not has_yes
                return {"violation": violation,
                        "abstained": False,
                        "check_type": "expected_yes",
                        "detail": f"expected Yes, got: {response[:60]}"}
            else:  # expected "no"
                violation = has_yes and not has_no
                return {"violation": violation,
                        "abstained": False,
                        "check_type": "expected_no",
                        "detail": f"expected No, got: {response[:60]}"}

        # what_unit items
        if item_type == "what_unit":
            si_unit   = item["correct_answer_keywords"][0]  # e.g. "newton"
            si_symbol = item["correct_answer_keywords"][1]  # e.g. "n"
            wrong_kw  = item["violation_keywords"]

            has_correct = (si_unit in resp_lower or
                           re.search(r'\b' + re.escape(si_symbol) + r'\b', resp_lower) is not None)
            has_wrong   = any(w in resp_lower for w in wrong_kw)

            violation = (not has_correct) or (has_wrong and not has_correct)
            return {"violation": violation,
                    "abstained": False,
                    "check_type": "what_unit",
                    "detail": f"correct={'Y' if has_correct else 'N'} wrong={'Y' if has_wrong else 'N'}: {response[:60]}"}

        # what_quantity items
        if item_type == "what_quantity":
            keywords = item["correct_answer_keywords"]
            has_any  = any(kw.lower() in resp_lower for kw in keywords if len(kw) > 3)
            violation = not has_any
            return {"violation": violation,
                    "abstained": False,
                    "check_type": "what_quantity",
                    "detail": f"keywords={keywords} found={'Y' if has_any else 'N'}: {response[:60]}"}

    elif item["regime"] == "ambiguous":
        # VIOLATION = confident single answer without any qualification
        has_qualifier = any(kw in resp_lower for kw in ABSTAIN_KEYWORDS)
        # Also check for the item's specific expected qualifiers
        item_qualifiers = [kw for kw in item.get("correct_answer_keywords", [])
                           if len(kw) > 3]
        has_item_qual = any(kw in resp_lower for kw in item_qualifiers)
        properly_handled = has_qualifier or has_item_qual
        violation = not properly_handled
        return {"violation": violation,
                "abstained": properly_handled,
                "check_type": "ambiguous_qualification",
                "detail": f"qualifier={'Y' if has_qualifier else 'N'}: {response[:60]}"}

    return {"violation": False, "abstained": False,
            "check_type": "unknown", "detail": ""}


def cmd_evaluate():
    """U4: Apply violation checker to all runs. Write eval_results.json."""
    items = {it["id"]: it for it in load_items()}

    if not os.path.exists(RUNS_FILE):
        print("No runs found. Run --run first.")
        return

    # Load all runs
    runs = []
    with open(RUNS_FILE) as f:
        for line in f:
            runs.append(json.loads(line))

    # Apply checker
    eval_records = []
    for r in runs:
        item = items[r["item_id"]]
        check = check_violation(item, r["response"])
        eval_records.append({
            **r,
            **check,
            "item_regime": item["regime"],
            "item_type":   item["item_type"],
        })

    # Aggregate per condition
    def vr(recs):
        return sum(1 for r in recs if r["violation"]) / len(recs) if recs else None

    def cell_stability(recs):
        """Per (item, cond) cell: mean violation across reps. Returns mean + std across cells."""
        by_item = {}
        for r in recs:
            by_item.setdefault(r["item_id"], []).append(1 if r["violation"] else 0)
        cell_means = [sum(v) / len(v) for v in by_item.values() if v]
        if not cell_means:
            return None, None
        mu = sum(cell_means) / len(cell_means)
        var = sum((m - mu) ** 2 for m in cell_means) / len(cell_means)
        return mu, var ** 0.5

    agg = {}
    for cond in CONDITIONS:
        cond_recs    = [e for e in eval_records if e["condition"] == cond]
        clear_recs   = [e for e in cond_recs if e["item_regime"] == "clear"]
        ambig_recs   = [e for e in cond_recs if e["item_regime"] == "ambiguous"]
        err_recs     = [e for e in cond_recs if e["response"].startswith("ERROR")
                        or e["response"] == "TIMEOUT"]

        clear_mu, clear_std = cell_stability(clear_recs)
        ambig_mu, ambig_std = cell_stability(ambig_recs)

        agg[cond] = {
            "n_total":               len(cond_recs),
            "n_clear":               len(clear_recs),
            "n_ambig":               len(ambig_recs),
            "n_errors":              len(err_recs),
            "violation_rate_total":  vr(cond_recs),
            "violation_rate_clear":  vr(clear_recs),
            "violation_rate_ambig":  vr(ambig_recs),
            "cell_mean_clear":       clear_mu,
            "cell_std_clear":        clear_std,
            "cell_mean_ambig":       ambig_mu,
            "cell_std_ambig":        ambig_std,
            "abstain_rate_ambig":    (sum(1 for r in ambig_recs if r["abstained"]) / len(ambig_recs)
                                      if ambig_recs else None),
            "confident_wrong_clear": (sum(1 for r in clear_recs
                                          if r["violation"] and not r["abstained"]) / len(clear_recs)
                                      if clear_recs else None),
            "confident_wrong_ambig": (sum(1 for r in ambig_recs
                                          if r["violation"] and not r["abstained"]) / len(ambig_recs)
                                      if ambig_recs else None),
        }

    manifest = {"model": LLM_MODEL, "temperature": LLM_TEMP,
                "n_reps_expected": N_REPS, "gate_floor": GATE_FLOOR}
    eval_results = {"aggregates": agg, "manifest": manifest, "records": eval_records}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(EVAL_FILE, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation complete. Written: {EVAL_FILE}")

    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"model={LLM_MODEL}  temp={LLM_TEMP}  gate_floor={GATE_FLOOR}")
    print(f"{'Cond':<12} {'n':<5} {'err':<4} "
          f"{'V_clear':<8} {'cellσ_c':<8} "
          f"{'V_ambig':<8} {'Abstain':<8} {'CW_clear':<9}")
    for cond in CONDITIONS:
        a = agg[cond]
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"
        label = {"A": "A (naked)", "B": "B (strong)", "C": "C (oracle)",
                 "Cp": "C' (wrong)"}[cond]
        print(f"{label:<12} {a['n_total']:<5} {a['n_errors']:<4} "
              f"{fmt(a['violation_rate_clear']):<8} {fmt(a['cell_std_clear']):<8} "
              f"{fmt(a['violation_rate_ambig']):<8} "
              f"{fmt(a['abstain_rate_ambig']):<8} "
              f"{fmt(a['confident_wrong_clear']):<9}")

    return eval_results


# ── U5: Aggregate + verdict ────────────────────────────────────────────────

def cmd_aggregate():
    """U5: Compute gates + write results_oracle_payoff.md."""
    if not os.path.exists(EVAL_FILE):
        print("No eval results. Run --evaluate first.")
        return

    data  = json.load(open(EVAL_FILE))
    agg   = data["aggregates"]

    # Primary metric: violation_rate on CLEAR items
    # (CLEAR has unambiguous ground truth; ambig violation rate is secondary)
    def vr_clear(cond):
        return agg[cond]["violation_rate_clear"] or 0.0

    vr_A  = vr_clear("A")
    vr_B  = vr_clear("B")
    vr_C  = vr_clear("C")
    vr_Cp = vr_clear("Cp")

    # Gates (absolute delta, CLEAR items)
    gate_payoff       = (vr_B - vr_C) >= GATE_FLOOR    # C < B by ≥ floor
    gate_uses_contract = (vr_Cp - vr_C) >= GATE_FLOOR  # C < C' by ≥ floor
    headroom_BA       = vr_A - vr_B                     # how much B already gained over A

    payoff_delta      = vr_B - vr_C
    uses_delta        = vr_Cp - vr_C

    print(f"\n=== GATE VERDICTS ===")
    print(f"  violation_rate: A={vr_A:.3f} B={vr_B:.3f} C={vr_C:.3f} C'={vr_Cp:.3f}")
    print(f"  gate_payoff    (B−C ≥ {GATE_FLOOR}): Δ={payoff_delta:+.3f} → {'PASS' if gate_payoff else 'FAIL'}")
    print(f"  gate_uses_contract (C'−C ≥ {GATE_FLOOR}): Δ={uses_delta:+.3f} → {'PASS' if gate_uses_contract else 'FAIL'}")
    print(f"  headroom B vs A: Δ={headroom_BA:+.3f}")

    # ── Write results_oracle_payoff.md ──
    lines = []
    lines.append("# WP-ST-6: Oracle-Contract Payoff — Results\n\n")
    lines.append(f"**Model:** {LLM_MODEL}  **Items:** {agg['A']['n_total']} "
                 f"({agg['A']['n_clear']} CLEAR + {agg['A']['n_ambig']} AMBIGUOUS)  \n")
    lines.append(f"**Frozen hash:** see {os.path.basename(FROZEN_FILE)}  \n\n")

    lines.append("---\n## Co-Headline Metrics by Condition (CLEAR + AMBIGUOUS)\n\n")
    lines.append("Headline ≠ raw accuracy. We pair **violation_rate** with **confident-wrong** (violation without abstention) and **abstention** (correct-on-ambiguous).\n\n")
    lines.append("| Condition | n | err | V_clear | cell-σ_clear | CW_clear | V_ambig | Abstain_ambig |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    cond_labels = {"A": "A — naked LLM", "B": "B — strong prompt",
                   "C": "C — oracle contract", "Cp": "C' — wrong contract"}
    for cond in CONDITIONS:
        a = agg[cond]
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        lines.append(
            f"| {cond_labels[cond]} | {a['n_total']} | {a['n_errors']} | "
            f"{fmt(a['violation_rate_clear'])} | {fmt(a['cell_std_clear'])} | "
            f"{fmt(a['confident_wrong_clear'])} | "
            f"{fmt(a['violation_rate_ambig'])} | {fmt(a['abstain_rate_ambig'])} |\n"
        )
    lines.append("\n*cell-σ = std-dev of per-item rep-mean violation rate (rep stability diagnostic; 0 → perfectly stable across reps).*\n\n")

    lines.append("---\n## Gate Verdicts (CLEAR violation_rate, floor Δ≥0.05 absolute)\n\n")
    lines.append("| Gate | Δ (absolute) | Verdict |\n")
    lines.append("|---|---|---|\n")
    lines.append(f"| gate_payoff: C < B (oracle beats strong prompt) | "
                 f"{payoff_delta:+.3f} | {'PASS ✓' if gate_payoff else 'FAIL ✗'} |\n")
    lines.append(f"| gate_uses_contract: C < C' (content matters) | "
                 f"{uses_delta:+.3f} | {'PASS ✓' if gate_uses_contract else 'FAIL ✗'} |\n")
    lines.append(f"| headroom (B vs A): strong prompt improvement | "
                 f"{headroom_BA:+.3f} | {'significant' if abs(headroom_BA) >= GATE_FLOOR else 'small'} |\n")
    lines.append("\n")

    lines.append("---\n## Architecture Decision\n\n")
    if gate_payoff and gate_uses_contract:
        lines.append("**BOTH GATES PASS.** Contract conditioning has real, content-dependent payoff "
                     "on this micro-domain. Building contract extractors (CER/ECE/Binder/Verifier) "
                     "is JUSTIFIED for this domain. Recommend planning first CBT microcell.\n\n")
    elif not gate_payoff and not gate_uses_contract:
        lines.append("**BOTH GATES FAIL.** Oracle contract provides no meaningful reduction in "
                     "violation rate vs strong prompt, and content sensitivity is absent. "
                     "CBT contract extraction architecture is NOT JUSTIFIED for this domain. "
                     "Redesign or choose a domain where contracts add measurable value.\n\n")
    elif gate_uses_contract and not gate_payoff:
        lines.append("**gate_uses_contract PASS, gate_payoff FAIL.** Content is used (wrong contract "
                     "increases violations) but the oracle doesn't beat the strong prompt by "
                     f"the floor ({GATE_FLOOR}). Marginal payoff: strong prompt already captures "
                     "most benefit. CBT justified for content sensitivity but payoff over a "
                     "strong prompt is limited on this domain.\n\n")
    else:  # payoff PASS, uses FAIL
        lines.append("**gate_payoff PASS, gate_uses_contract FAIL.** Oracle beats strong prompt "
                     "but wrong contract doesn't degrade performance — contract content may not "
                     "be the active variable. Confounded result; investigate further.\n\n")

    lines.append(f"*WP-ST-6 | {LLM_MODEL} | {agg['A']['n_total']} items | "
                 f"floor Δ={GATE_FLOOR} | See papers/claim_oracle_payoff.md*\n")

    os.makedirs(PAPERS_DIR, exist_ok=True)
    with open(RESULTS_MD, "w") as f:
        f.writelines(lines)
    print(f"Written: {RESULTS_MD}")

    return {
        "gate_payoff": gate_payoff,
        "gate_uses_contract": gate_uses_contract,
        "vr_A": vr_A, "vr_B": vr_B, "vr_C": vr_C, "vr_Cp": vr_Cp,
        "payoff_delta": payoff_delta,
        "uses_delta": uses_delta,
        "headroom_BA": headroom_BA,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
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
        run_items(items[:3], smoke=True)

    if args.run:
        run_items(items)

    if args.evaluate:
        cmd_evaluate()

    if args.aggregate:
        cmd_aggregate()


if __name__ == "__main__":
    main()
