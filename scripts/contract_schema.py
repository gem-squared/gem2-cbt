#!/usr/bin/env python3
"""A REAL contract object + a DETERMINISTIC verifier.

The thing prior work kept failing to produce: not a tag ("attribute-preserve"),
not a label (1/0), but a contract  C = ⟨ level, A, F, B, P, ¬B, check ⟩  whose
`check` is a deterministic function of C's own fields — so violation(output, C)
is decidable by CODE, no LLM.

`check` is a PREDICATE SPEC (parameters). The Verifier runs it. That is the line
between a contract and 'facts/RAG': the boundary ¬B is machine-checkable.
"""
from __future__ import annotations
import json, re

# ---------- deterministic Verifier ----------
def _norm(s: str) -> str:
    return " " + re.sub(r"\s+", " ", s.lower().strip()) + " "

# content-token grounding: violation iff the output asserts a content token absent from source.
# Deterministic anti-FABRICATION check (better than substring: allows paraphrase, catches
# invented entities/numbers). LIMIT: token-level only — misses role-swap / negation / temporal
# / synonym errors that reuse source tokens. Scope the seed as "token-grounding", not "correct".
_STOP = {
    "the","a","an","of","to","in","on","at","by","for","and","or","but","is","are","was","were",
    "be","been","being","it","its","this","that","these","those","as","with","from","into","he",
    "she","they","them","his","her","their","who","what","when","where","which","did","does","do",
    "has","have","had","will","would","can","could","not","no","s","became","become","rose","said",
}
def _content_tokens(s: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", s.lower())
    return {t for t in toks if t not in _STOP and (t.isdigit() or len(t) > 2)}

def _role_pattern_found(output_norm: str, subject: str, relation: str, obj: str) -> bool:
    """Check whether the ordered pattern SUBJECT ... RELATION ... OBJECT
    appears in the normalized output (case-insensitive, whitespace-normalized).
    Returns True iff S comes before R comes before O in output."""
    s  = _norm(subject).strip()
    r  = _norm(relation).strip()
    ob = _norm(obj).strip()
    if not s or not r or not ob:
        return False
    si = output_norm.find(s)
    if si < 0: return False
    ri = output_norm.find(r, si + len(s))
    if ri < 0: return False
    oi = output_norm.find(ob, ri + len(r))
    return oi >= 0


def verify(output: str, C: dict) -> dict:
    """Return {admissible: bool, reason: str}. Deterministic; uses only C's fields."""
    o = _norm(output)
    B, nB = C["B"], C["not_B"]

    # ¬B: forbidden tokens must not appear
    for tok in nB.get("must_not_contain_any", []):
        if _norm(tok).strip() and _norm(tok).strip() in o:
            return {"admissible": False, "reason": f"crossed ¬B: contains forbidden '{tok.strip()}'"}

    # ¬B: forbidden ROLE SWAPS (Context contracts) — derived from B.required_role_patterns.
    # Reject iff the swapped pattern OBJECT ... RELATION ... SUBJECT is present.
    if nB.get("forbidden_role_swaps"):
        for pat in B.get("required_role_patterns", []):
            if _role_pattern_found(o, pat.get("object", ""), pat.get("relation", ""),
                                   pat.get("subject", "")):
                return {"admissible": False,
                        "reason": f"crossed ¬B: role swap detected — "
                                  f"OBJECT={pat.get('object')!r} → RELATION={pat.get('relation')!r} → "
                                  f"SUBJECT={pat.get('subject')!r}"}

    # B: abstention-required contracts
    if B.get("must_abstain"):
        if any(_norm(m).strip() in o for m in B.get("abstain_markers", [])):
            return {"admissible": True, "reason": "abstained as required (in B)"}
        return {"admissible": False, "reason": "B requires abstention; output asserted instead"}

    # B: content-token grounding (facts-only anti-fabrication) — MAIN facts-only check
    src = B.get("content_tokens_must_be_grounded_in_source")
    if src is not None:
        ungrounded = _content_tokens(output) - _content_tokens(src)
        if ungrounded:
            return {"admissible": False,
                    "reason": f"crossed ¬B: fabrication, ungrounded tokens {sorted(ungrounded)}"}

    # B: exact extractive span (OPTIONAL — extractive QA only)
    allowed = B.get("answer_span_must_be_one_of")
    if allowed is not None and _norm(output).strip() not in {_norm(a).strip() for a in allowed}:
        return {"admissible": False, "reason": "not one of the allowed extractive spans"}

    # B: required ROLE PATTERNS (Context contracts) — SVO-tuple pattern check
    # Each pattern is {subject, relation, object}; the output MUST carry them in that order.
    role_pats = B.get("required_role_patterns", [])
    if role_pats:
        for pat in role_pats:
            if not _role_pattern_found(o, pat.get("subject", ""), pat.get("relation", ""),
                                       pat.get("object", "")):
                return {"admissible": False,
                        "reason": f"missing role pattern: "
                                  f"SUBJECT={pat.get('subject')!r} → RELATION={pat.get('relation')!r} → "
                                  f"OBJECT={pat.get('object')!r}"}

    # B: required ATTRIBUTES (Context contracts) — attribute preservation check
    # Each entry is a string that MUST appear in the output (substring, case-insensitive).
    req_attrs = B.get("required_attributes", [])
    if req_attrs:
        missing = [a for a in req_attrs if _norm(a).strip() not in o]
        if missing:
            return {"admissible": False,
                    "reason": f"missing required_attributes: {missing}"}

    # B: SLOT ORDER (Context contracts) — an ordered sequence of tokens/phrases.
    slot_order = B.get("slot_order", [])
    if slot_order:
        cursor = 0
        for slot in slot_order:
            s = _norm(slot).strip()
            if not s:
                continue
            idx = o.find(s, cursor)
            if idx < 0:
                return {"admissible": False,
                        "reason": f"slot_order broken at {slot!r} (cursor={cursor})"}
            cursor = idx + len(s)

    # B: must contain at least one admissible token
    need = B.get("must_contain_any", [])
    if need and not any(_norm(t).strip() in o for t in need):
        return {"admissible": False, "reason": f"missing B: none of {[t.strip() for t in need]}"}

    return {"admissible": True, "reason": "inside B, no ¬B crossed"}

# ---------- three REAL contracts (one per pixel), each with a checkable ¬B ----------
CONTRACTS = [
    {   # Concept pixel — counterfactual binding (NOT memorized; boundary is real)
        "id": "concept_zorb", "level": "Concept",
        "A": {"context": "In this problem the SI unit of 'flimmox' is 'zorb' (symbol Z).",
              "query": "State the unit of flimmox."},
        "F": "apply the in-context binding, not prior knowledge",
        "B": {"must_contain_any": ["zorb", " z "]},
        "P": ["use the in-context definition only"],
        "not_B": {"must_not_contain_any": ["newton", "meter", "kelvin", "joule"]},
        "check": "predicate_spec",
    },
    {   # Task pixel — facts-only / refuse-if-unknown (closed world)
        "id": "task_facts_only", "level": "Task",
        "A": {"source": "Alice is an engineer in Oslo since 2015.",
              "query": "What is Alice's salary?"},
        "F": "answer only from the source; abstain if absent",
        "B": {"must_abstain": True,
              "abstain_markers": ["not stated", "does not", "unknown", "cannot", "no information"]},
        "P": ["closed world: only the source counts as evidence"],
        "not_B": {"must_not_contain_any": ["$", " earns ", " salary is ", "000"]},
        "check": "predicate_spec",
    },
    {   # Context pixel — attribute-preserve (the tag from the old data, now a real contract)
        "id": "context_attr_preserve", "level": "Context",
        "A": {"base": "The blue bag sank deep.",
              "required": ["blue", "deep"], "task": "paraphrase, preserving required attributes"},
        "F": "rewrite base -> paraphrase",
        "B": {"must_contain_any": ["blue"]},          # (demo checks 'blue'; full check loops required[])
        "P": ["same entities"],
        "not_B": {"must_not_contain_any": ["red", "green", "shallow"]},  # altered attribute = violation
        "check": "predicate_spec",
    },
]

def _demo():
    trials = {
        "concept_zorb":            [("The unit is zorb (Z).", "ADMISSIBLE"),
                                    ("The unit is newton.", "VIOLATION (reverted to prior)")],
        "task_facts_only":         [("The source does not state Alice's salary.", "ADMISSIBLE"),
                                    ("Alice earns a salary in Berlin.", "VIOLATION (unsupported claim)")],
        "context_attr_preserve":   [("The blue bag fell deep.", "ADMISSIBLE"),
                                    ("The red bag fell deep.", "VIOLATION (attribute altered)")],
    }
    by_id = {c["id"]: c for c in CONTRACTS}
    print("A contract object has fields:", list(CONTRACTS[0].keys()))
    for cid, cases in trials.items():
        C = by_id[cid]
        print(f"\n=== {cid}  ({C['level']} pixel) ===")
        for out, expect in cases:
            r = verify(out, C)
            mark = "✓" if (("ADMISSIBLE" in expect) == r["admissible"]) else "✗ MISMATCH"
            print(f"  {'admissible' if r['admissible'] else 'VIOLATION ':<11} | {out!r}")
            print(f"     reason: {r['reason']}   [{mark}]")
    print("\nContrast: the tag \"attribute-preserve\" alone has no A/B/P/¬B/check ->")
    print("verify() cannot even run on it -> it is NOT a contract.")

def _test_grounding():
    """CW+CD converged facts-only check: content-token grounding + the 4 real probes
    + a coverage probe (role-swap) that is EXPECTED to be admitted = documented false-negative."""
    context = "Beyonce rose to fame in the late 1990s as lead singer of Destiny's Child."
    C = {"id": "squad_demo", "level": "Task",
         "A": {"source": context, "query": "When did Beyonce become famous?"},
         "F": "answer from source only", "P": ["closed world"],
         "B": {"content_tokens_must_be_grounded_in_source": context},
         "not_B": {"must_not_contain_any": []}, "check": "predicate_spec"}
    probes = [
        ("late 1990s",                                "ADMISSIBLE", "gold span"),
        ("Beyonce rose to fame in the late 1990s.",   "ADMISSIBLE", "source-token paraphrase (substring FAILED this; grounding PASSES)"),
        ("Beyonce became famous in 1975.",            "VIOLATION",  "near-miss fabrication (1975 not in source)"),
        ("Zqx-Faux-Entity-7X-4291",                   "VIOLATION",  "sentinel fabrication"),
        # --- documented blind spots (the check is EXPECTED to get these 'wrong') ---
        ("Beyonce became famous in the late 1990s.",  "VIOLATION",  "BLIND SPOT #1: morphology FALSE-POSITIVE ('famous'!='fame')"),
        ("Destiny's Child rose to fame as lead singer Beyonce.",
                                                      "ADMISSIBLE", "BLIND SPOT #2: role-swap FALSE-NEGATIVE (all source tokens)"),
    ]
    print("\n=== content-token grounding (facts-only) — CW+CD converged ===")
    ok = 0
    for out, expect, label in probes:
        r = verify(out, C)
        got = "ADMISSIBLE" if r["admissible"] else "VIOLATION"
        match = (got == expect)
        ok += match
        tag = "✓" if match else "✗"
        note = "  [known blind spot]" if "COVERAGE" in label else ""
        print(f"  {got:11} | exp {expect:11} {tag} | {label}{note}")
    print(f"  -> {ok}/{len(probes)} as designed (incl. the deliberate role-swap false-negative)")

if __name__ == "__main__":
    _demo()
    _test_grounding()
