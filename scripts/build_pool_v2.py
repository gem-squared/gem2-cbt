#!/usr/bin/env python3
"""WP-ST-18 U1: Clean shared pool v2 — single DeepSeek teacher, diverse framings.

Per passage: 3 level FRAMINGS (task / concept / context), each a natural NL user
prompt. STYLE is rotated by persona (per passage) to prevent single-template
collapse (D1 leak). LEVEL differs only by semantic intent of the ask, not surface
form (so U2 leakage audit only needs to strip a small residue).

Pool structure:
  data/ece_shared_pool_v2/items.jsonl   — one record per (passage, level)
  data/ece_shared_pool_v2/manifest.json — sources, counts, D1 diversity, freeze-hash
  data/ece_shared_pool_v2/frozen_dataset_hash.json  — freeze record (from cbt.fingerprint)

Sources: squad, wsd, paws, mnli.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from oracle_payoff import _load_env, call_llm  # noqa: E402  DeepSeek backend
_load_env(str(ROOT / ".env"))

POOL_DIR = ROOT / "data" / "ece_shared_pool_v2"
TEACHER_MODEL = "deepseek-chat"

# ── Passage loaders ────────────────────────────────────────────────────────

def _load_squad(n: int, seed: int) -> list[dict]:
    """SQuAD passages from local WP-14 seed (data/contract_squad/items.jsonl)."""
    path = ROOT / "data" / "contract_squad" / "items.jsonl"
    rng = _rng(seed)
    out = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "answerable" and isinstance(rec.get("source"), str):
                out.append({"passage": rec["source"], "orig_id": rec["id"]})
    _shuffle(out, rng)
    # dedupe by passage
    seen, dedup = set(), []
    for r in out:
        if r["passage"] not in seen:
            seen.add(r["passage"]); dedup.append(r)
        if len(dedup) >= n:
            break
    return [{"passage_id": f"squad_{i:04d}", "source_label": "squad",
             "passage": r["passage"], "orig_id": r["orig_id"]}
            for i, r in enumerate(dedup[:n])]


def _load_wsd(n: int, seed: int) -> list[dict]:
    """WSD sentence-level passages from data/processed_v3/wsd_instances.jsonl."""
    path = ROOT / "data" / "processed_v3" / "wsd_instances.jsonl"
    rng = _rng(seed)
    out = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            s = rec.get("sentence", "")
            if isinstance(s, str) and len(s) > 60:
                out.append({"passage": s, "orig_id": rec.get("id", "")})
    _shuffle(out, rng)
    seen, dedup = set(), []
    for r in out:
        if r["passage"] not in seen:
            seen.add(r["passage"]); dedup.append(r)
        if len(dedup) >= n:
            break
    return [{"passage_id": f"wsd_{i:04d}", "source_label": "wsd",
             "passage": r["passage"], "orig_id": r["orig_id"]}
            for i, r in enumerate(dedup[:n])]


def _load_paws(n: int, seed: int) -> list[dict]:
    """PAWS sentence1 as passage."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
    rng = _rng(seed)
    idxs = list(range(len(ds)))
    _shuffle(idxs, rng)
    seen, out = set(), []
    for i in idxs:
        s = ds[i]["sentence1"]
        if isinstance(s, str) and len(s) > 60 and s not in seen:
            seen.add(s)
            out.append({"passage": s, "orig_id": ds[i]["id"]})
        if len(out) >= n:
            break
    return [{"passage_id": f"paws_{i:04d}", "source_label": "paws",
             "passage": r["passage"], "orig_id": str(r["orig_id"])}
            for i, r in enumerate(out[:n])]


def _load_mnli(n: int, seed: int) -> list[dict]:
    """MNLI premise as passage."""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/multi_nli", split="train")
    rng = _rng(seed)
    idxs = list(range(len(ds)))
    _shuffle(idxs, rng)
    seen, out = set(), []
    for i in idxs:
        s = ds[i]["premise"]
        if isinstance(s, str) and len(s) > 60 and s not in seen:
            seen.add(s)
            out.append({"passage": s, "orig_id": ds[i]["pairID"]})
        if len(out) >= n:
            break
    return [{"passage_id": f"mnli_{i:04d}", "source_label": "mnli",
             "passage": r["passage"], "orig_id": r["orig_id"]}
            for i, r in enumerate(out[:n])]


LOADERS = {"squad": _load_squad, "wsd": _load_wsd,
           "paws": _load_paws, "mnli": _load_mnli}


def _rng(seed: int):
    import random
    return random.Random(seed)


def _shuffle(xs, rng):
    rng.shuffle(xs)


# ── DeepSeek framing generation ───────────────────────────────────────────

PERSONAS = [
    ("student",   "You are a curious student asking an AI tutor. Keep it conversational and natural."),
    ("researcher","You are a researcher writing a query to a colleague AI. Be precise and specific."),
    ("editor",    "You are an editor giving instructions to an assistant. Be direct and imperative."),
    ("teacher",   "You are a teacher explaining a task to a novice AI. Be pedagogical."),
    ("writer",    "You are a technical writer specifying a request. Be formal and structured."),
]

LEVEL_INTENT = {
    "task":    ("Ask the assistant a specific extractive question about a concrete factual detail "
                "in the passage (who / what / when / where / how many / etc.). The answer MUST be "
                "grounded in the passage. Do NOT paraphrase the passage. Do NOT ask about "
                "'meaning of a word' — that is a different task."),
    "concept": ("Identify a single ambiguous term (one polysemous word or short phrase) that appears "
                "in the passage, and ask the assistant which contextual sense of that term is "
                "intended here. Do NOT ask any factual extractive question about the passage's "
                "content. Do NOT ask about paraphrasing."),
    "context": ("Ask the assistant to judge whether a proposed rewording of the passage "
                "would preserve the essential roles and attributes (who did what, key qualifiers). "
                "Do NOT include the paraphrase itself. Do NOT ask about a specific fact. Do NOT "
                "ask about the meaning of a single word."),
}

FRAMING_META_PROMPT = """\
{persona_line}

Write ONE natural user prompt that a real user (in your persona) would send to
an AI assistant, given the passage below. Vary your phrasing — do NOT copy any
rigid template like "is this X?" or "which sense of X?". Sound like a real
person, not a form-filler.

Your prompt must INVOKE this task on the passage:
{level_intent}

Rules:
- Output ONLY the user prompt text — no JSON, no code fence, no preface, no quotes.
- Between 1 and 3 sentences.
- Do NOT include the passage verbatim in your prompt (assume the passage is already in
  the assistant's context).
- Do NOT use words that reveal the LEVEL name (do not write "task", "concept",
  "context", "framing", "extractive", "paraphrase" if you can avoid them —
  it is OK to imply the task naturally).
- **Vary your opening — do NOT start your prompt with any of the following:**
  "In the passage,", "In the sentence,", "Based on the passage,", "Looking at the passage,",
  "According to the passage,", "From the passage,", "The passage says", "Reading the passage,".
  Start with a verb (e.g. "Tell me"), a question word ("Which", "What", "How", "Who"),
  a direct request ("I need", "I'd like", "Please"), or a natural conversational hook
  ("Hey", "Hmm", "So", "OK"). Use different openings for each of your responses.

PASSAGE:
{passage}
"""


def _deepseek_framing(passage: str, level: str, persona_line: str,
                      timeout: int = 90, max_retries: int = 2) -> tuple[str, str]:
    """Return (framing_text, raw_response)."""
    prompt = FRAMING_META_PROMPT.format(
        persona_line=persona_line, level_intent=LEVEL_INTENT[level], passage=passage)
    last = ""
    for attempt in range(max_retries + 1):
        try:
            out = call_llm(prompt, "", timeout=timeout)
            if out and not out.startswith("ERROR") and out != "TIMEOUT":
                # strip leading quotes / code fences if the model ignored the rule
                cleaned = out.strip().strip('`').strip()
                if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 2:
                    cleaned = cleaned[1:-1].strip()
                return cleaned, out
            last = out
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
        time.sleep(1.5 * (attempt + 1))
    return f"ERROR: {last}", last


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── D1 diversity metric ───────────────────────────────────────────────────

_STOP = {"the","a","an","of","to","in","on","at","by","for","and","or","is","are"}

def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower())]

def _opening_5gram(s: str) -> str:
    toks = _tokens(s)[:5]
    return " ".join(toks) if len(toks) >= 3 else ""

def d1_diversity(records: list[dict]) -> dict:
    """Per-level: max 5-gram opening frequency ≤ 0.30 (D1 ≥ 0.70)."""
    out = {}
    for lvl in ("task", "concept", "context"):
        prompts = [r["framing"]["prompt"] for r in records if r["level"] == lvl]
        if not prompts:
            out[lvl] = {"n": 0, "max_opening_frac": 1.0, "d1": 0.0,
                        "top_opening": "", "pass": False}
            continue
        cnts = Counter(_opening_5gram(p) for p in prompts)
        top_op, top_c = cnts.most_common(1)[0]
        frac = top_c / len(prompts)
        out[lvl] = {"n": len(prompts), "max_opening_frac": frac,
                    "d1": 1.0 - frac, "top_opening": top_op,
                    "pass": frac <= 0.30}
    out["all_pass"] = all(v["pass"] for k, v in out.items() if k != "all_pass")
    out["min_d1"] = min(v["d1"] for k, v in out.items() if k != "all_pass")
    return out


# ── Build ─────────────────────────────────────────────────────────────────

def build(n_per_source: int, seed: int, out_dir: Path, resume: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"

    passages: list[dict] = []
    for src, loader in LOADERS.items():
        got = loader(n_per_source, seed)
        assert len(got) == n_per_source, f"{src}: got {len(got)} != {n_per_source}"
        passages.extend(got)
    print(f"[U1] loaded {len(passages)} passages "
          f"({n_per_source} × 4 sources)")

    # resume: skip (passage_id, level) already emitted
    done: set[tuple[str, str]] = set()
    if resume and items_path.exists():
        with open(items_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["passage_id"], r["level"]))
        print(f"[U1] resume: {len(done)} records already present")

    records = []
    if items_path.exists():
        with open(items_path) as f:
            records = [json.loads(line) for line in f]

    n_new = 0
    with open(items_path, "a") as f:
        for pi, p in enumerate(passages):
            persona = PERSONAS[pi % len(PERSONAS)]
            for lvl in ("task", "concept", "context"):
                key = (p["passage_id"], lvl)
                if key in done:
                    continue
                framing_txt, raw = _deepseek_framing(p["passage"], lvl, persona[1])
                if framing_txt.startswith("ERROR"):
                    print(f"[U1][WARN] {p['passage_id']}/{lvl} teacher error: {framing_txt[:100]}")
                    continue
                rec = {
                    "passage_id": p["passage_id"],
                    "source_label": p["source_label"],
                    "orig_id": p["orig_id"],
                    "passage": p["passage"],
                    "level": lvl,
                    "persona": persona[0],
                    "framing": {"prompt": framing_txt},
                    "teacher_raw": raw,
                    "teacher_raw_sha256": _sha256(raw),
                    "teacher_model": TEACHER_MODEL,
                    "wp": "WP-ST-18",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                records.append(rec)
                n_new += 1
                if n_new % 20 == 0:
                    print(f"[U1] emitted {n_new} new (persona={persona[0]}, "
                          f"latest {p['passage_id']}/{lvl})")

    print(f"[U1] wrote {n_new} new records; total in items.jsonl = {len(records)}")
    return {"n_passages": len(passages), "n_records": len(records),
            "n_new": n_new, "records": records}


def compute_gate(records: list[dict], n_per_source: int) -> dict:
    """WP-18 U1 GATE checks."""
    # (a) all 3 framings per passage
    per_p: dict[str, set[str]] = {}
    for r in records:
        per_p.setdefault(r["passage_id"], set()).add(r["level"])
    complete = sum(1 for lvls in per_p.values() if lvls == {"task","concept","context"})
    incomplete = [pid for pid, lvls in per_p.items()
                  if lvls != {"task","concept","context"}]

    # (b) source ⊥ level counts balanced (by construction — check equal counts per cell)
    cells: Counter = Counter()
    for r in records:
        cells[(r["source_label"], r["level"])] += 1
    expected = n_per_source
    balanced = all(c == expected for c in cells.values()) and len(cells) == 12

    # (c) D1 diversity
    d1 = d1_diversity(records)

    return {
        "n_records": len(records),
        "n_passages_complete": complete,
        "n_passages_incomplete": len(incomplete),
        "incomplete_ids": incomplete[:10],
        "cell_counts": {f"{s}|{l}": c for (s, l), c in sorted(cells.items())},
        "cells_balanced": bool(balanced),
        "expected_per_cell": expected,
        "d1": d1,
        "all_complete": len(incomplete) == 0,
    }


def cmd_smoke(args):
    N = args.n_per_source or 6
    print(f"[SMOKE] N per source = {N}")
    res = build(n_per_source=N, seed=42, out_dir=POOL_DIR / "smoke",
                resume=args.resume)
    gate = compute_gate(res["records"], N)
    print(json.dumps(gate, indent=2))
    # smoke pass = all 3 framings emitted per passage AND D1 healthy
    smoke_pass = gate["all_complete"] and gate["d1"]["all_pass"] and gate["cells_balanced"]
    print(f"\n[SMOKE] pass={smoke_pass}  min_d1={gate['d1']['min_d1']:.3f}  "
          f"cells_balanced={gate['cells_balanced']}")
    (POOL_DIR / "smoke" / "gate.json").write_text(
        json.dumps({"smoke_pass": smoke_pass, **gate}, indent=2))
    return 0 if smoke_pass else 1


def cmd_full(args):
    N = args.n_per_source or 50
    print(f"[FULL] N per source = {N}")
    res = build(n_per_source=N, seed=42, out_dir=POOL_DIR, resume=args.resume)
    gate = compute_gate(res["records"], N)
    print(json.dumps(gate, indent=2))

    gate_pass = (gate["all_complete"] and gate["d1"]["all_pass"]
                 and gate["cells_balanced"])
    print(f"\n[FULL] GATE pass={gate_pass}")

    manifest = {
        "wp": "WP-ST-18", "unit": "U1", "teacher": TEACHER_MODEL,
        "n_per_source": N, "sources": list(LOADERS.keys()),
        "personas": [p[0] for p in PERSONAS],
        "d1_threshold_max_opening_frac": 0.30,
        "gate": gate,
        "gate_pass": gate_pass,
    }
    (POOL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # freeze-hash: SHA256 over items.jsonl (sorted by passage_id, level)
    if gate_pass:
        # write canonical sorted view for hashing
        recs = res["records"]
        recs_sorted = sorted(recs, key=lambda r: (r["passage_id"], r["level"]))
        sorted_path = POOL_DIR / "items_sorted.jsonl"
        with open(sorted_path, "w") as f:
            for r in recs_sorted:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        h = hashlib.sha256()
        with open(sorted_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        frozen = h.hexdigest()[:16]
        freeze_record = {
            "frozen_hash": frozen,
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": "WP-ST-18 U1 pool v2 freeze — sorted items.jsonl SHA256[:16]",
            "n_records": len(recs), "n_per_source": N,
            "sources": list(LOADERS.keys()),
        }
        (POOL_DIR / "frozen_dataset_hash.json").write_text(
            json.dumps(freeze_record, indent=2))
        print(f"[U1] frozen_hash = {frozen}")
    return 0 if gate_pass else 1


def cmd_gate(args):
    """Recompute gate on existing items.jsonl (for debugging)."""
    N = args.n_per_source or 50
    path = (POOL_DIR / "smoke" / "items.jsonl"
            if args.smoke else POOL_DIR / "items.jsonl")
    with open(path) as f:
        records = [json.loads(line) for line in f]
    gate = compute_gate(records, N)
    print(json.dumps(gate, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke")
    s.add_argument("--n-per-source", type=int, default=6)
    s.add_argument("--no-resume", dest="resume", action="store_false")
    s.set_defaults(resume=True)

    f = sub.add_parser("full")
    f.add_argument("--n-per-source", type=int, default=50)
    f.add_argument("--no-resume", dest="resume", action="store_false")
    f.set_defaults(resume=True)

    g = sub.add_parser("gate")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--n-per-source", type=int, default=50)

    args = ap.parse_args()
    if args.cmd == "smoke":
        sys.exit(cmd_smoke(args))
    elif args.cmd == "full":
        sys.exit(cmd_full(args))
    elif args.cmd == "gate":
        cmd_gate(args)


if __name__ == "__main__":
    main()
