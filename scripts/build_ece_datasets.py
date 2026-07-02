"""WP-ST-16: Real per-level ECE datasets — Task / Concept / Context — teacher-
extracted via Claude-CLI headless (OAuth, NOT API key), self-verified against
`scripts/contract_schema.py::verify`.

Three per-level datasets are assembled here so that the CBT design has REAL
teacher-extracted `NL → contract ⟨A, F, B, P, ¬B, check⟩` pairs, not the
mechanically-templated ones WP-14 produced (WP-14's flaw). Then WP-16 U5 uses
these to build a SHARED PASSAGE POOL for the PS validation (U6-U8) where
negatives are TOPIC-MATCHED — one pool of passages, three level-candidates per
passage, so PS validation is not confounded with corpus classification.

Level → source corpus map:
  Task    : reuse data/contract_squad/items.jsonl (WP-14 seed, hash bdc404d760819e19)
  Concept : reuse data/processed_v3/wsd_instances.jsonl (real WSD, 153k rows)
            + optional WordNet MWE additions (first-cut, offline)
  Context : HF PAWS (paraphrase pairs) + MNLI (entailment) — THE REAL GAP
            per the ARCHITECT directive.

Teacher extraction:
  Teacher = Claude-CLI headless via OAuth (`claude -p`). No API key. Provenance
  is recorded per-manifest so downstream readers can audit that no paid API
  key was involved. Rate limits: bounded via hard-cap of 500 teacher calls
  per level (U2 / U3 / U4) and 600 calls in the shared pool (U5).

Self-verify:
  Each extracted contract runs a two-probe test — the level-appropriate
  gold-good output MUST verify admissible, and a level-appropriate gold-bad
  MUST verify violation. Only items where BOTH probes fire correctly are
  retained. Yield stats persisted in manifest.json per dataset.

Output paths (all under data/, LOCAL-ONLY per gitignored data/* policy):
  data/ece_task/          items.jsonl + manifest.json
  data/ece_concept/       items.jsonl + manifest.json
  data/ece_context/       items.jsonl + manifest.json
  data/ece_shared_pool/   items.jsonl + manifest.json  (U5)

Usage:
  python scripts/build_ece_datasets.py --list-teacher-cmd   # U1 audit: show teacher invocation
  python scripts/build_ece_datasets.py --smoke-task         # U2 smoke: 20 items on contract_squad
  python scripts/build_ece_datasets.py --run-task           # U2 full: bounded ~400
  python scripts/build_ece_datasets.py --smoke-concept      # U3 smoke: 20 WSD rows
  python scripts/build_ece_datasets.py --run-concept        # U3 full: bounded ~300
  python scripts/build_ece_datasets.py --smoke-context      # U4 smoke: 10 PAWS + 10 MNLI
  python scripts/build_ece_datasets.py --run-context        # U4 full: bounded ~400
  python scripts/build_ece_datasets.py --shared-pool        # U5: 200 passages × 3 levels
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Reuse repo utilities (READ-ONLY imports) ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contract_schema import verify, _norm, _content_tokens  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────────
SEED = 42

# Data directories (all under data/ → gitignored)
TASK_DIR    = "data/ece_task"
CONCEPT_DIR = "data/ece_concept"
CONTEXT_DIR = "data/ece_context"
POOL_DIR    = "data/ece_shared_pool"

# Reuse WP-14 seed for Task
CONTRACT_SQUAD_DIR   = "data/contract_squad"
CONTRACT_SQUAD_ITEMS = os.path.join(CONTRACT_SQUAD_DIR, "items.jsonl")

# Reuse processed_v3 WSD instances for Concept
WSD_INSTANCES = "data/processed_v3/wsd_instances.jsonl"

# HF corpora for Context (loaded lazily inside U4)
HF_PAWS_ID = ("google-research-datasets/paws", "labeled_final")
HF_MNLI_ID = "nyu-mll/multi_nli"

# Sample sizes (per WP-16 CONTRACT):
N_SMOKE          = 20    # per-level smoke (each level splits internally)
N_RUN_TASK       = 400   # 200 answerable + 200 unanswerable target
N_RUN_CONCEPT    = 300   # 200 WSD + 100 idioms/MWE (first-cut)
N_RUN_CONTEXT    = 400   # 200 PAWS + 200 MNLI
N_POOL           = 200   # shared pool passages (U5 default)
HARD_CAP_PER_LVL = 500   # bounded teacher-call budget per level
HARD_CAP_POOL    = 700   # bounded budget for U5 (200 × 3 = 600 + safety)


# ── Teacher: Claude-CLI headless via OAuth (NOT API key) ──────────────────
TEACHER_CMD = ["claude", "-p"]  # `-p` = print-and-exit non-interactive mode
TEACHER_TIMEOUT_S = 120         # per-call timeout
TEACHER_MODEL_LABEL = "claude-code-cli-headless"

PROVENANCE = {
    "no_api_key":     True,
    "auth":           "OAuth (Claude Code CLI, David's Mac)",
    "teacher_source": "Claude-CLI headless (`claude -p`)",
    "teacher_model":  TEACHER_MODEL_LABEL,
    "no_network_in_verify": True,
}


def teacher_call(prompt: str, timeout: int = TEACHER_TIMEOUT_S) -> str:
    """Invoke Claude-CLI headless with `prompt` on stdin. Returns stdout.

    Raises `RuntimeError` on subprocess failure. Callers wrap with retry as needed.
    """
    proc = subprocess.run(
        TEACHER_CMD,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"teacher call failed: rc={proc.returncode}, stderr={proc.stderr[:200]!r}"
        )
    return proc.stdout.strip()


# ── Provenance / manifest helpers ─────────────────────────────────────────
def write_manifest(out_dir: str, stats: dict) -> str:
    """Persist per-dataset provenance + yield stats."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")
    record = {
        "wp":            "WP-ST-16",
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "seed":          SEED,
        "provenance":    PROVENANCE,
        "stats":         stats,
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


# ══════════════════════════════════════════════════════════════════════════
# U2: Task-level teacher extraction on contract_squad (WP-14 seed)
#
# Teacher decides the contract per row (not templated). Two-probe self-verify
# per item: gold-good must be admissible AND gold-bad must be violation.
# Retain only items where BOTH probes fire correctly.
# ══════════════════════════════════════════════════════════════════════════

# Bad-probe constants — reuse WP-14 conventions
FAB_SPAN   = "Xqzfvbm-Kjplmr-Wntzyq-84391"           # from WP-14 (verified 0/2000 collisions)
FAB_ANSWER = f"The answer is {FAB_SPAN}."            # for unanswerable fab-probe
# GOLD_ABSTAIN_PHRASE: contains "unanswerable" — the marker that every teacher
# chose in the 20-smoke inspection. Broader than "does not say" (which teachers
# often omitted). Real Qwen may phrase abstention any of several ways, but this
# probe checks the teacher's marker set covers a common natural abstention.
GOLD_ABSTAIN_PHRASE = "This question is unanswerable from the given passage."

# The ONLY thing that's templated here is the RUBRIC — the extractor tells the
# teacher what a valid contract looks like (which verifier keys exist), but the
# teacher chooses the specific contract per row. This is exactly the WP-14 vs
# WP-16 distinction the ARCHITECT called out.
TEACHER_PROMPT_TASK = """You are a contract extractor. Given a passage and a question, produce a JSON contract that specifies (a) what an admissible answer looks like and (b) what would VIOLATE the contract. A separate verifier will check answers against your contract deterministically.

The verifier supports these B-keys (pick what fits this item):

- `content_tokens_must_be_grounded_in_source`: <passage-string>
    Admissible iff every content-word in the answer (proper nouns, numbers, non-stopword words of length > 2) appears in the passage. Use for extractive/answerable items.

- `must_contain_any`: [<phrase-1>, <phrase-2>, ...]
    Admissible iff answer contains at least one listed phrase. Use for open-ended answerable items where a specific keyword must be present.

- `must_abstain`: true
  `abstain_markers`: [<phrase-1>, <phrase-2>, ...]
    Admissible iff the answer contains any listed abstain marker (case-insensitive). Use for UNANSWERABLE items (passage does not contain the answer).

And these ¬B keys:
- `not_B.must_not_contain_any`: [<forbidden-phrase>, ...]
    Violation iff the answer contains any of these phrases. Use as belt-and-suspenders for unanswerable items ONLY (e.g. forbidden = ["the answer is"] to catch specific-answer fabrications).

Return a JSON object with EXACTLY these keys and nothing else — no code fences, no prose, no explanations:

{{"A": {{"source": "<the passage verbatim>", "query": "<the question verbatim>"}},
 "F": "<one sentence describing what the answer function should do>",
 "B": {{<the chosen B-key(s)>}},
 "P": ["<one precondition>"],
 "not_B": {{"must_not_contain_any": [<forbidden phrases>]}}}}

KIND: {kind}

PASSAGE:
{passage}

QUESTION:
{question}"""


def _parse_teacher_contract(text: str) -> dict | None:
    """Robust parser for teacher output: strip code fences, find outermost JSON
    object, parse. Returns dict or None if unparseable."""
    if not text:
        return None
    s = text.strip()
    # Strip triple-backtick fences (with or without language tag)
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        s = s.strip()
    # Find outermost { ... }
    start = s.find("{")
    end   = s.rfind("}")
    if start < 0 or end <= start:
        return None
    body = s[start:end + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _two_probe_verify_task(row: dict, contract: dict) -> dict:
    """Run the two-probe self-verify for a Task item.

    answerable   → verify(gold_answer, contract) MUST be admissible
                    AND verify(FAB_SPAN, contract) MUST be violation
    unanswerable → verify(GOLD_ABSTAIN_PHRASE, contract) MUST be admissible
                    AND verify(FAB_ANSWER, contract) MUST be violation
    """
    kind = row["kind"]
    if kind == "answerable":
        gold = row["contract"]["A"]["source"]  # placeholder — real gold below
        # Pull gold from the original contract_squad probes (probe_gold verdict was on gold answer)
        # contract_squad records store `probes.p_gold` — but a simpler path: pull gold from SQuAD id
        # Since contract_squad.items records preserve `contract.A` + `nl_prompt` + `source`, the
        # gold answer is available via a lookup helper. For now use the U2 signature — the caller
        # (loop over records) passes `gold_answer` in the row dict.
        gold = row["_gold_answer"]
        p_good = verify(gold,     contract)
        p_bad  = verify(FAB_SPAN, contract)
    else:
        p_good = verify(GOLD_ABSTAIN_PHRASE, contract)
        p_bad  = verify(FAB_ANSWER,          contract)
    retain = p_good["admissible"] and (not p_bad["admissible"])
    return {"probes": {"p_good": p_good, "p_bad": p_bad},
            "retain": retain}


def _augment_row_with_gold(row: dict) -> dict:
    """Populate `_gold_answer` from the original SQuAD row's probes.p_gold
    (contract_squad's items.jsonl records store the probe verdict but not
    the gold string directly — we recover it by looking up the source SQuAD row).
    For efficiency, we cache the SQuAD gold-answer map on first use."""
    # WP-14 items.jsonl preserves `probes.p_gold.reason` which contains
    # "inside B, no ¬B crossed" — the gold string is NOT persisted there.
    # Simplest path: re-load SQuAD by id and pull answers.text[0].
    # But that requires re-loading SQuAD which is slow for 400 items.
    # Alternative: for smoke, keep an in-memory HF handle; call once, cache.
    from datasets import load_dataset  # lazy
    global _SQUAD_GOLD_CACHE
    if "_SQUAD_GOLD_CACHE" not in globals():
        ds = load_dataset("rajpurkar/squad_v2", split="train")
        _SQUAD_GOLD_CACHE = {r["id"]: (r["answers"]["text"][0] if r["answers"]["text"] else "")
                             for r in ds}
    row["_gold_answer"] = _SQUAD_GOLD_CACHE.get(row["id"], "")
    return row


def _teacher_extract_task_contract(row: dict) -> tuple[dict | None, str]:
    """Call teacher (Claude-CLI headless OAuth) to extract a Task contract.
    Returns (parsed_contract_dict_or_None, raw_teacher_output)."""
    prompt = TEACHER_PROMPT_TASK.format(
        kind=row["kind"],
        passage=row["source"],
        question=row["nl_prompt"],
    )
    try:
        raw = teacher_call(prompt)
    except subprocess.TimeoutExpired:
        return (None, "TIMEOUT")
    except RuntimeError as e:
        return (None, f"ERROR: {e}")
    parsed = _parse_teacher_contract(raw)
    # Ensure minimum schema — teacher may omit `not_B` on answerable
    if parsed and "not_B" not in parsed:
        parsed["not_B"] = {"must_not_contain_any": []}
    if parsed and "P" not in parsed:
        parsed["P"] = ["closed world"]
    if parsed and "check" not in parsed:
        parsed["check"] = "predicate_spec"
    return (parsed, raw)


def _load_contract_squad_items() -> list:
    """Load WP-14 items.jsonl and augment each row with the SQuAD gold answer."""
    items = []
    with open(CONTRACT_SQUAD_ITEMS) as f:
        for line in f:
            r = json.loads(line)
            items.append(r)
    return items


def _pick_stratified_by_kind(items: list, n_per_kind: int, seed: int) -> list:
    """Deterministic stratified sample: n_per_kind answerable + n_per_kind unanswerable."""
    rng = random.Random(seed)
    by_kind = {}
    for r in items:
        by_kind.setdefault(r["kind"], []).append(r)
    picked = []
    for kind in ("answerable", "unanswerable"):
        group = list(by_kind.get(kind, []))
        rng.shuffle(group)
        picked.extend(group[:n_per_kind])
    return picked


def _run_task_extraction(items: list, out_path: str, mode: str) -> dict:
    """Iterate items, teacher-extract, self-verify, write JSONL. Resumable."""
    os.makedirs(TASK_DIR, exist_ok=True)

    # Resume set
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    total = len(items)
    print(f"[{mode}] processing {total} items ({len(done_ids)} already done); "
          f"teacher = {TEACHER_MODEL_LABEL}")

    n_run = n_parse_ok = n_verify_pass = n_calls = 0
    t0 = time.time()

    with open(out_path, "a") as fout:
        for idx, row in enumerate(items):
            if row["id"] in done_ids:
                continue
            _augment_row_with_gold(row)
            n_calls += 1
            if n_calls > HARD_CAP_PER_LVL:
                print(f"[{mode}] hard-cap {HARD_CAP_PER_LVL} calls reached; stopping.")
                break
            contract, raw = _teacher_extract_task_contract(row)
            n_run += 1
            parse_ok = contract is not None
            if parse_ok:
                n_parse_ok += 1
                # Complete the contract's A + fields (teacher may omit)
                if "A" not in contract:
                    contract["A"] = {"source": row["source"], "query": row["nl_prompt"]}
                elif not isinstance(contract["A"], dict):
                    contract["A"] = {"source": row["source"], "query": row["nl_prompt"]}
                verdict = _two_probe_verify_task(row, contract)
                retain = verdict["retain"]
                if retain:
                    n_verify_pass += 1
                # Persist EVERY record — retention flag is a field for U5 filtering
                out_rec = {
                    "id":              row["id"],
                    "kind":            row["kind"],
                    "nl_prompt":       row["nl_prompt"],
                    "source":          row["source"],
                    "gold_answer":     row["_gold_answer"],
                    "contract":        contract,
                    "teacher_raw":     raw,
                    "probes":          verdict["probes"],
                    "retain":          retain,
                    "teacher_model":   TEACHER_MODEL_LABEL,
                }
            else:
                out_rec = {
                    "id":              row["id"],
                    "kind":            row["kind"],
                    "nl_prompt":       row["nl_prompt"],
                    "source":          row["source"],
                    "gold_answer":     row.get("_gold_answer", ""),
                    "contract":        None,
                    "teacher_raw":     raw,
                    "probes":          None,
                    "retain":          False,
                    "teacher_model":   TEACHER_MODEL_LABEL,
                    "parse_error":     True,
                }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            done_ids.add(row["id"])
            elapsed = time.time() - t0
            avg = elapsed / n_run if n_run else 0
            mark = "OK" if (parse_ok and out_rec["retain"]) else ("VER" if parse_ok else "PAR")
            print(f"  [{mark}] {idx + 1:3d}/{total} id={row['id'][:22]:22s} "
                  f"kind={row['kind']:12s} "
                  f"retain={out_rec['retain']}  (avg={avg:.1f}s/call)")

    dt = time.time() - t0

    # Per-regime yield report — enforce David's "never just the pooled figure"
    per_regime = {}
    for r in items:
        k = r["kind"]
        per_regime.setdefault(k, {"n": 0, "n_retain": 0})
        per_regime[k]["n"] += 1
    # Re-scan output file for retained-by-kind counts (resume-safe)
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                rec = json.loads(line)
                k = rec.get("kind")
                if k in per_regime and rec.get("retain"):
                    per_regime[k]["n_retain"] += 1
            except Exception:
                pass
    for k, s in per_regime.items():
        s["rate"] = s["n_retain"] / s["n"] if s["n"] else 0.0

    stats = {
        "mode":               mode,
        "n_target":           total,
        "n_run":              n_run,
        "n_parse_ok":         n_parse_ok,
        "n_verify_pass":      n_verify_pass,
        "yield_parse":        n_parse_ok / n_run if n_run else 0.0,
        "yield_verify":       n_verify_pass / n_run if n_run else 0.0,
        "per_regime":         per_regime,
        "wall_clock_s":       round(dt, 1),
        "avg_s_per_call":     round(dt / n_run, 2) if n_run else 0,
        "hard_cap":           HARD_CAP_PER_LVL,
    }
    print(f"\n[{mode}] {n_run} teacher calls; parse_ok={n_parse_ok} "
          f"({stats['yield_parse']:.1%}); verify_pass={n_verify_pass} "
          f"({stats['yield_verify']:.1%}); {dt:.0f}s")
    print(f"[{mode}] per-regime retention:")
    for k in ("answerable", "unanswerable"):
        if k in per_regime:
            s = per_regime[k]
            print(f"  {k:14s}: {s['n_retain']}/{s['n']} = {s['rate']:.1%}")
    return stats


def _write_balanced_dataset(out_path: str, balanced_path: str) -> dict:
    """Emit a regime-balanced Task dataset: truncate to min(n_retain_ans, n_retain_unans)
    so the final kept set does not collapse to mostly-answerable. Balance-truncation is
    deterministic (order preserved as written). Returns the balance stats."""
    if not os.path.exists(out_path):
        return {"balanced": False, "reason": "raw items.jsonl missing"}
    retained_by_kind: dict = {}
    with open(out_path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("retain"):
                continue
            k = rec.get("kind")
            retained_by_kind.setdefault(k, []).append(rec)

    if "answerable" not in retained_by_kind or "unanswerable" not in retained_by_kind:
        return {"balanced": False, "reason": "one regime missing", "counts": {k: len(v) for k, v in retained_by_kind.items()}}

    n_min = min(len(retained_by_kind["answerable"]), len(retained_by_kind["unanswerable"]))
    with open(balanced_path, "w") as fout:
        for k in ("answerable", "unanswerable"):
            for rec in retained_by_kind[k][:n_min]:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {
        "balanced":     True,
        "n_per_regime": n_min,
        "n_total":      n_min * 2,
        "n_dropped":    {
            "answerable":   max(0, len(retained_by_kind["answerable"])   - n_min),
            "unanswerable": max(0, len(retained_by_kind["unanswerable"]) - n_min),
        },
        "path":         balanced_path,
    }


def cmd_smoke_task():
    """U2 smoke: 10 answerable + 10 unanswerable from contract_squad."""
    from cbt.fingerprint import assert_frozen_hash
    frozen = assert_frozen_hash(CONTRACT_SQUAD_DIR)
    print(f"[smoke-task] contract_squad frozen hash asserted: {frozen}")
    items = _load_contract_squad_items()
    sample = _pick_stratified_by_kind(items, N_SMOKE // 2, seed=SEED)
    print(f"[smoke-task] sample = {len(sample)} items (seed={SEED})")
    smoke_path = os.path.join(TASK_DIR, "smoke.jsonl")
    stats = _run_task_extraction(sample, smoke_path, mode="smoke")
    stats["frozen_seed_hash"] = frozen
    write_manifest(TASK_DIR + "_smoke_manifest", stats)
    print(f"\n[smoke-task] wrote smoke.jsonl + manifest")
    return stats


def cmd_run_task():
    """U2 full: bounded 200 ans + 200 unans from contract_squad (hard-cap 500).
    Emits both items.jsonl (all retained) AND balanced.jsonl (regime-truncated
    to min per regime — David's regime-balance guarantee)."""
    from cbt.fingerprint import assert_frozen_hash
    frozen = assert_frozen_hash(CONTRACT_SQUAD_DIR)
    print(f"[run-task] contract_squad frozen hash asserted: {frozen}")
    items = _load_contract_squad_items()
    sample = _pick_stratified_by_kind(items, N_RUN_TASK // 2, seed=SEED)
    print(f"[run-task] sample = {len(sample)} items (seed={SEED})")
    full_path     = os.path.join(TASK_DIR, "items.jsonl")
    balanced_path = os.path.join(TASK_DIR, "balanced.jsonl")
    stats = _run_task_extraction(sample, full_path, mode="run")
    balance_stats = _write_balanced_dataset(full_path, balanced_path)
    stats["balance_stats"]    = balance_stats
    stats["frozen_seed_hash"] = frozen
    write_manifest(TASK_DIR, stats)
    print(f"\n[run-task] wrote items.jsonl (all retained) + "
          f"balanced.jsonl ({balance_stats.get('n_total', '?')} balanced) + manifest")
    return stats


# ══════════════════════════════════════════════════════════════════════════
# U3: Concept-level teacher extraction on WSD instances (WordNet-derived)
#
# Contract shape (teacher-extracted, not templated):
#   A: {source: sentence, target: word, query: "which sense of {lemma}?"}
#   F: teacher-authored one-liner
#   B: {must_contain_any: [teacher-chosen true-sense keywords]}
#   not_B: {must_not_contain_any: [teacher-chosen distractor keywords]}
#
# Two-probe self-verify per item:
#   good  = true_gloss text → must be admissible
#   bad   = a distractor gloss text → must be violation
# ══════════════════════════════════════════════════════════════════════════

TEACHER_PROMPT_CONCEPT = """You are a contract extractor for word-sense disambiguation. Given a sentence with a target word and a list of candidate senses (with glosses), produce a JSON contract that specifies (a) what a correct sense-identification answer looks like and (b) what would VIOLATE the contract by selecting a wrong sense. A separate deterministic verifier checks answers against your contract.

The verifier's B and ¬B keys you may use:
- `must_contain_any`: [<keyword-1>, <keyword-2>, ...]  →  answer must contain at least one listed keyword (case-insensitive substring).
- `not_B.must_not_contain_any`: [<forbidden-phrase>, ...]  →  answer must NOT contain any of these phrases.

Choose 3-6 keywords from the TRUE gloss (the gloss at TRUE_INDEX). Choose 3-6 forbidden phrases from the other (DISTRACTOR) glosses. Prefer content words (nouns, adjectives, verbs). Avoid stopwords ("a", "the", "of", "to", "in", "on").

Return a JSON object with EXACTLY these keys — no code fences, no prose:

{{"A": {{"source": "<sentence verbatim>", "target": "<target word>", "query": "which sense of {lemma}?"}},
 "F": "<one sentence describing what the answer function should do>",
 "B": {{"must_contain_any": [<true-sense keywords>]}},
 "P": ["use the in-context sense binding, not free association"],
 "not_B": {{"must_not_contain_any": [<distractor forbidden phrases>]}}}}

SENTENCE:
{sentence}

TARGET WORD:
{target}

LEMMA:
{lemma}

CANDIDATES (numbered, TRUE_INDEX marked):
{candidates_block}
"""


def _format_candidates_block(cands: list, glosses: list, true_idx: int) -> str:
    lines = []
    for i, (c, g) in enumerate(zip(cands, glosses)):
        marker = " ← TRUE" if i == true_idx else ""
        lines.append(f"  [{i}] {c}: {g}{marker}")
    return "\n".join(lines)


def _teacher_extract_concept_contract(row: dict) -> tuple[dict | None, str]:
    """Call teacher (Claude-CLI headless OAuth) to extract a Concept contract."""
    prompt = TEACHER_PROMPT_CONCEPT.format(
        sentence=row["sentence"],
        target=row["target"],
        lemma=row["lemma"],
        candidates_block=_format_candidates_block(
            row["candidates"], row["candidate_glosses"], row["true_idx"],
        ),
    )
    try:
        raw = teacher_call(prompt)
    except subprocess.TimeoutExpired:
        return (None, "TIMEOUT")
    except RuntimeError as e:
        return (None, f"ERROR: {e}")
    parsed = _parse_teacher_contract(raw)
    if parsed:
        if "not_B" not in parsed:
            parsed["not_B"] = {"must_not_contain_any": []}
        if "P" not in parsed:
            parsed["P"] = ["use the in-context sense binding"]
        if "check" not in parsed:
            parsed["check"] = "predicate_spec"
    return (parsed, raw)


def _two_probe_verify_concept(row: dict, contract: dict) -> dict:
    """Good probe = true_gloss verbatim; bad probe = a distractor gloss verbatim."""
    true_idx = row["true_idx"]
    glosses = row["candidate_glosses"]
    good_probe = glosses[true_idx]
    # Pick the first distractor
    distractor_idx = next((i for i in range(len(glosses)) if i != true_idx), true_idx)
    bad_probe = glosses[distractor_idx]
    p_good = verify(good_probe, contract)
    p_bad  = verify(bad_probe,  contract)
    retain = p_good["admissible"] and (not p_bad["admissible"])
    return {"probes":     {"p_good": p_good, "p_bad": p_bad,
                           "good_text": good_probe, "bad_text": bad_probe},
            "retain":     retain}


def _load_wsd_instances(n_max: int | None = None) -> list:
    """Load WSD instances; drop rows with < 2 candidates (no distractor)."""
    items = []
    with open(WSD_INSTANCES) as f:
        for line in f:
            r = json.loads(line)
            if len(r.get("candidates", [])) < 2:
                continue
            items.append(r)
            if n_max and len(items) >= n_max:
                break
    return items


def _pick_wsd_stratified_by_mfs(items: list, n_per_group: int, seed: int) -> list:
    """Stratified sample: n_per_group MFS + n_per_group non-MFS (WSD's own labels)."""
    rng = random.Random(seed)
    mfs      = [r for r in items if r.get("is_mfs")]
    non_mfs  = [r for r in items if not r.get("is_mfs")]
    rng.shuffle(mfs)
    rng.shuffle(non_mfs)
    return mfs[:n_per_group] + non_mfs[:n_per_group]


def _run_concept_extraction(items: list, out_path: str, mode: str) -> dict:
    """Iterate items, teacher-extract, self-verify concept probes, write JSONL."""
    os.makedirs(CONCEPT_DIR, exist_ok=True)

    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    total = len(items)
    print(f"[{mode}] processing {total} items ({len(done_ids)} already done); "
          f"teacher = {TEACHER_MODEL_LABEL}")

    n_run = n_parse_ok = n_verify_pass = n_calls = 0
    t0 = time.time()

    with open(out_path, "a") as fout:
        for idx, row in enumerate(items):
            if row["id"] in done_ids:
                continue
            n_calls += 1
            if n_calls > HARD_CAP_PER_LVL:
                print(f"[{mode}] hard-cap {HARD_CAP_PER_LVL} calls reached; stopping.")
                break
            contract, raw = _teacher_extract_concept_contract(row)
            n_run += 1
            parse_ok = contract is not None
            if parse_ok:
                n_parse_ok += 1
                if "A" not in contract or not isinstance(contract["A"], dict):
                    contract["A"] = {"source": row["sentence"], "target": row["target"],
                                     "query": f"which sense of {row['lemma']}?"}
                verdict = _two_probe_verify_concept(row, contract)
                retain = verdict["retain"]
                if retain:
                    n_verify_pass += 1
                out_rec = {
                    "id":                row["id"],
                    "kind":              "wsd",
                    "lemma":             row["lemma"],
                    "target":            row["target"],
                    "true_synset":       row["true_synset"],
                    "true_idx":          row["true_idx"],
                    "is_mfs":            row.get("is_mfs"),
                    "sentence":          row["sentence"],
                    "candidates":        row["candidates"],
                    "candidate_glosses": row["candidate_glosses"],
                    "contract":          contract,
                    "teacher_raw":       raw,
                    "probes":            verdict["probes"],
                    "retain":            retain,
                    "teacher_model":     TEACHER_MODEL_LABEL,
                }
            else:
                out_rec = {
                    "id":                row["id"],
                    "kind":              "wsd",
                    "lemma":             row["lemma"],
                    "target":            row["target"],
                    "true_synset":       row["true_synset"],
                    "true_idx":          row["true_idx"],
                    "is_mfs":            row.get("is_mfs"),
                    "sentence":          row["sentence"],
                    "candidates":        row["candidates"],
                    "candidate_glosses": row["candidate_glosses"],
                    "contract":          None,
                    "teacher_raw":       raw,
                    "probes":            None,
                    "retain":            False,
                    "teacher_model":     TEACHER_MODEL_LABEL,
                    "parse_error":       True,
                }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            done_ids.add(row["id"])
            elapsed = time.time() - t0
            avg = elapsed / n_run if n_run else 0
            mark = "OK" if (parse_ok and out_rec["retain"]) else ("VER" if parse_ok else "PAR")
            print(f"  [{mark}] {idx + 1:3d}/{total} id={row['id'][:22]:22s} "
                  f"lemma={row['lemma'][:12]:12s} mfs={row.get('is_mfs')} "
                  f"retain={out_rec['retain']} (avg={avg:.1f}s/call)")

    dt = time.time() - t0
    # Per-regime: mfs vs non-mfs (with discrimination diagnostics)
    per_regime = {"mfs":     {"n": 0, "n_retain": 0, "n_p_good_rej": 0, "n_p_bad_adm": 0},
                  "non_mfs": {"n": 0, "n_retain": 0, "n_p_good_rej": 0, "n_p_bad_adm": 0}}
    for r in items:
        k = "mfs" if r.get("is_mfs") else "non_mfs"
        per_regime[k]["n"] += 1
    sample_ids = {r["id"] for r in items}
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                rec = json.loads(line)
                if rec.get("id") not in sample_ids:
                    continue
                k = "mfs" if rec.get("is_mfs") else "non_mfs"
                if rec.get("retain"):
                    per_regime[k]["n_retain"] += 1
                # Discrimination diagnostics (parseable items only)
                probes = rec.get("probes")
                if probes and probes.get("p_good") and probes.get("p_bad"):
                    if not probes["p_good"]["admissible"]:
                        per_regime[k]["n_p_good_rej"] += 1
                    if probes["p_bad"]["admissible"]:
                        per_regime[k]["n_p_bad_adm"] += 1
            except Exception:
                pass
    for k, s in per_regime.items():
        s["rate"]              = s["n_retain"]    / s["n"] if s["n"] else 0.0
        s["p_good_rej_rate"]   = s["n_p_good_rej"] / s["n"] if s["n"] else 0.0
        s["p_bad_adm_rate"]    = s["n_p_bad_adm"]  / s["n"] if s["n"] else 0.0

    stats = {
        "mode":           mode,
        "n_target":       total,
        "n_run":          n_run,
        "n_parse_ok":     n_parse_ok,
        "n_verify_pass":  n_verify_pass,
        "yield_parse":    n_parse_ok / n_run if n_run else 0.0,
        "yield_verify":   n_verify_pass / n_run if n_run else 0.0,
        "per_regime":     per_regime,
        "wall_clock_s":   round(dt, 1),
        "avg_s_per_call": round(dt / n_run, 2) if n_run else 0,
        "hard_cap":       HARD_CAP_PER_LVL,
    }
    print(f"\n[{mode}] {n_run} teacher calls; parse_ok={n_parse_ok} "
          f"({stats['yield_parse']:.1%}); verify_pass={n_verify_pass} "
          f"({stats['yield_verify']:.1%}); {dt:.0f}s")
    print(f"[{mode}] per-regime (WSD MFS vs non-MFS):")
    print(f"  {'regime':8s} {'retain':>10s} {'p_good_rej':>13s} {'p_bad_adm':>12s}")
    for k in ("mfs", "non_mfs"):
        s = per_regime[k]
        print(f"  {k:8s} {s['n_retain']}/{s['n']}={s['rate']:.1%}   "
              f"{s['n_p_good_rej']}/{s['n']}={s['p_good_rej_rate']:.1%}   "
              f"{s['n_p_bad_adm']}/{s['n']}={s['p_bad_adm_rate']:.1%}")
    print(f"[{mode}] discrimination of RETAINED = 100% by construction "
          f"(retain formula: p_good.adm AND NOT p_bad.adm)")
    return stats


def cmd_smoke_concept():
    """U3 smoke: 10 MFS + 10 non-MFS WSD rows from processed_v3 (bounded)."""
    items = _load_wsd_instances(n_max=50_000)  # cap loader for smoke speed
    sample = _pick_wsd_stratified_by_mfs(items, N_SMOKE // 2, seed=SEED)
    print(f"[smoke-concept] sample = {len(sample)} items (seed={SEED})")
    smoke_path = os.path.join(CONCEPT_DIR, "smoke.jsonl")
    stats = _run_concept_extraction(sample, smoke_path, mode="smoke")
    write_manifest(CONCEPT_DIR + "_smoke_manifest", stats)
    print(f"\n[smoke-concept] wrote smoke.jsonl + manifest")
    return stats


def cmd_run_concept():
    """U3 full: bounded 150 MFS + 150 non-MFS from processed_v3 (hard-cap 500).
    NOTE: idioms/MWE augmentation deferred to a follow-up call — first-cut is
    WSD-only to keep the hard-cap defensible and per-regime yields interpretable."""
    items = _load_wsd_instances(n_max=100_000)  # broader loader for run
    sample = _pick_wsd_stratified_by_mfs(items, N_RUN_CONCEPT // 2, seed=SEED)
    print(f"[run-concept] sample = {len(sample)} items (seed={SEED})")
    full_path = os.path.join(CONCEPT_DIR, "items.jsonl")
    stats = _run_concept_extraction(sample, full_path, mode="run")
    write_manifest(CONCEPT_DIR, stats)
    print(f"\n[run-concept] wrote items.jsonl + manifest")
    return stats


# ══════════════════════════════════════════════════════════════════════════
# U4: Context-level teacher extraction on HF PAWS + MNLI — THE REAL GAP
#
# Two sub-corpora fold into the same Context level:
#   PAWS  (paraphrase pairs, label 0=non-paraphrase, 1=paraphrase)
#   MNLI  (entailment, label 0=entailment, 1=neutral, 2=contradiction)
#
# Contract shape (teacher-extracted, not templated):
#   A: {base, target, query}
#   F: teacher-authored
#   B: {must_contain_any: [label-appropriate keywords]}
#   not_B: {must_not_contain_any: [opposite-label keywords]}
#
# Two-probe self-verify:
#   good = a natural phrasing carrying the gold label → must be admissible
#   bad  = a natural phrasing carrying an opposite label → must be violation
# ══════════════════════════════════════════════════════════════════════════

# ── U4 REBUILD (CW directive 2026-07-01): REAL structural probes ──────────
# Previous implementation tested discrimination on CANNED verdict strings
# ("These sentences are paraphrases" vs "not paraphrases") which trivially
# discriminated by label vocabulary — NOT by content/role structure. CW
# correctly flagged this as vacuous. Rebuild:
#
#   p_good = the REAL paraphrase / entailment sentence from the dataset
#   p_bad  = the REAL role-swap / contradiction sentence from the dataset
#            (matched to the SAME source premise/sentence1 where possible)
#
# The check itself remains the deterministic content-token / must_contain_any
# scheme — but now measured against REAL structural pairs, so the LOW yield
# CW predicts (token-grounding is blind to role-swap by design) is the
# honest finding, not an artifact.

# Label semantics (informational only — teacher sees them via prompt)
PAWS_LABEL_TEXT = {0: "not a paraphrase (role-swap / adversarial edit)", 1: "paraphrase"}
MNLI_LABEL_TEXT = {0: "entailment", 1: "neutral", 2: "contradiction"}


TEACHER_PROMPT_CONTEXT_PAWS = """You are a STRUCTURAL contract extractor for paraphrase-preserving judgement. Given a SOURCE sentence, produce a JSON contract that specifies what STRUCTURE any valid paraphrase must preserve, and what STRUCTURAL swap would VIOLATE. A deterministic verifier will apply your contract to candidate answers.

**IMPORTANT: you see only the source. Do NOT copy verbatim phrases from any candidate answer — you have none. Decompose the source into STRUCTURAL patterns instead.**

Choose the best pattern subtype for the source. Prefer role-triples if the sentence has a clean subject-verb-object structure; otherwise use attribute or slot-order constraints.

Verifier B-keys (pick what fits — you may use one or combine):

- `required_role_patterns`: [{{"subject": "<S>", "relation": "<V>", "object": "<O>"}}, ...]
    For SVO sentences. The verifier admits iff each pattern appears in the output IN ORDER (S before R before O, case-insensitive substring match). Rejects the derived swap O→R→S if `not_B.forbidden_role_swaps` is true.

- `required_attributes`: [<attribute-string>, ...]
    For descriptive sentences. Each attribute string must appear as substring in the output.

- `slot_order`: [<phrase-1>, <phrase-2>, ...]
    An ordered sequence — each phrase must appear after the previous one in the output.

Verifier not_B keys:
- `forbidden_role_swaps`: true
    Turns on the automatic swap-check derived from `required_role_patterns` — the verifier rejects if the output contains OBJECT → RELATION → SUBJECT (the swap of the pattern).

Return a JSON object with EXACTLY these keys — no code fences, no prose:

{{"A": {{"source": "<source sentence verbatim>", "query": "is the candidate a role-preserving paraphrase of source?"}},
 "F": "<one sentence describing what the answer function should do>",
 "B": {{<pick one or more of: required_role_patterns / required_attributes / slot_order>}},
 "P": ["preserve subject/object roles and attributes; word choice may vary"],
 "not_B": {{"forbidden_role_swaps": true}} }}

SOURCE:
{source}
"""


TEACHER_PROMPT_CONTEXT_MNLI = """You are a STRUCTURAL contract extractor for entailment. Given a PREMISE, produce a JSON contract that captures what any valid entailment must carry over from the premise.

**IMPORTANT: you see only the premise. Do NOT copy verbatim phrases from any candidate answer.**

Note: MNLI entailment is largely a SEMANTIC judgement, not a role-swap detection problem. Token-based structural checks (this WP's verifier) can only partially capture entailment — this is reported as GRADED (⊨), not strict deterministic PASS/FAIL.

Verifier B-keys:
- `required_role_patterns`: [{{"subject": "<S>", "relation": "<V>", "object": "<O>"}}, ...] — for SVO premises
- `required_attributes`: [<attribute-string>, ...] — key attributes an entailment should mention
- `slot_order`: [<phrase-1>, <phrase-2>, ...] — ordered sequence

Return a JSON object with EXACTLY these keys — no code fences, no prose:

{{"A": {{"premise": "<premise verbatim>", "query": "does the answer preserve the premise's entailment structure?"}},
 "F": "<one sentence>",
 "B": {{<pick one or more of the keys above>}},
 "P": ["reason about entailment structure"],
 "not_B": {{"forbidden_role_swaps": true}} }}

PREMISE:
{premise}
"""


def _teacher_extract_context_paws(row: dict) -> tuple[dict | None, str]:
    """PAWS teacher extraction — SEES ONLY SOURCE (probes hidden).

    The teacher must NOT see the paraphrase or role-swap. Its job is to
    decompose the source into structural patterns (role-triples, attributes,
    slot-order) that any valid paraphrase must preserve.
    """
    prompt = TEACHER_PROMPT_CONTEXT_PAWS.format(source=row["source"])
    try:
        raw = teacher_call(prompt)
    except (subprocess.TimeoutExpired, RuntimeError) as e:
        return (None, f"ERROR: {e}")
    parsed = _parse_teacher_contract(raw)
    if parsed:
        if "not_B" not in parsed: parsed["not_B"] = {"forbidden_role_swaps": True}
        if "P"     not in parsed: parsed["P"]     = ["preserve roles/attributes"]
        if "check" not in parsed: parsed["check"] = "predicate_spec"
    return (parsed, raw)


def _teacher_extract_context_mnli(row: dict) -> tuple[dict | None, str]:
    """MNLI teacher extraction — SEES ONLY PREMISE (probes hidden). MNLI is
    reported separately as GRADED, not strict deterministic pass/fail."""
    prompt = TEACHER_PROMPT_CONTEXT_MNLI.format(premise=row["source"])
    try:
        raw = teacher_call(prompt)
    except (subprocess.TimeoutExpired, RuntimeError) as e:
        return (None, f"ERROR: {e}")
    parsed = _parse_teacher_contract(raw)
    if parsed:
        if "not_B" not in parsed: parsed["not_B"] = {"forbidden_role_swaps": True}
        if "P"     not in parsed: parsed["P"]     = ["reason about entailment"]
        if "check" not in parsed: parsed["check"] = "predicate_spec"
    return (parsed, raw)


# ── Held-out generalization probes: synonym-paraphrase + reworded-swap ────
# Two extra teacher calls per row — the RE-WORD teacher gets the source PLUS
# a candidate (paraphrase or role-swap) and rewords it. It does NOT see the
# contract. The reworded versions test whether the check generalizes past the
# exact lexicalization of the original probes.
HELDOUT_SYNONYM_PROMPT = """Rewrite the following sentence using synonyms and different phrasing while preserving the exact meaning (subject/object roles, attributes, negations). Return the rewritten sentence ONLY, no prose.

ORIGINAL:
{text}"""

HELDOUT_REWORDED_SWAP_PROMPT = """Rewrite the following sentence using synonyms and different phrasing while preserving its meaning (which is a role-swap or contradiction of the source below). Return the rewritten sentence ONLY, no prose.

SOURCE (for context, do not preserve its meaning — preserve the ROLE-SWAP's meaning):
{source}

ROLE-SWAP / CONTRADICTION to reword:
{text}"""


def _teacher_reword_synonym(text: str) -> str:
    """Ask teacher to reword `text` with synonyms while preserving meaning."""
    try:
        return teacher_call(HELDOUT_SYNONYM_PROMPT.format(text=text)).strip()
    except (subprocess.TimeoutExpired, RuntimeError) as e:
        return f"ERROR: {e}"


def _teacher_reword_swap(source: str, text: str) -> str:
    """Ask teacher to reword a role-swap / contradiction with synonyms."""
    try:
        return teacher_call(HELDOUT_REWORDED_SWAP_PROMPT.format(
            source=source, text=text)).strip()
    except (subprocess.TimeoutExpired, RuntimeError) as e:
        return f"ERROR: {e}"


def _tautology_check(contract: dict, p_good_text: str, p_bad_text: str) -> dict:
    """Detect whether authoring leaked the probes: are B/¬B strings substrings
    of the probes they're designed to match/reject?"""
    B = contract.get("B", {})
    nB = contract.get("not_B", {})
    good_norm = _norm(p_good_text)
    bad_norm  = _norm(p_bad_text)
    b_leaked  = False
    nb_leaked = False
    # required_attributes / must_contain_any string leakage
    for a in B.get("required_attributes", []) + B.get("must_contain_any", []):
        if _norm(a).strip() and _norm(a).strip() in good_norm and _norm(a).strip() not in bad_norm:
            b_leaked = True
            break
    # slot_order phrase leakage
    for a in B.get("slot_order", []):
        if _norm(a).strip() and _norm(a).strip() in good_norm and _norm(a).strip() not in bad_norm:
            b_leaked = True
            break
    # not_B.must_not_contain_any string leakage
    for a in nB.get("must_not_contain_any", []):
        if _norm(a).strip() and _norm(a).strip() in bad_norm and _norm(a).strip() not in good_norm:
            nb_leaked = True
            break
    return {"b_leaked": b_leaked, "nb_leaked": nb_leaked,
            "any_leaked": b_leaked or nb_leaked}


def _two_probe_verify_context(row: dict, contract: dict, subcorpus: str) -> dict:
    """CW+CD merged: hidden probes + held-out generalization probes + tautology check.

    Real probes (hidden from teacher during authoring):
      p_good  = real PAWS paraphrase / real MNLI entailment
      p_bad   = real PAWS role-swap  / real MNLI contradiction

    Held-out generalization probes (measure ceiling of the check):
      p_good_synonym  = a synonym-reworded version of p_good (should still admit)
      p_bad_reworded  = a synonym-reworded version of p_bad  (should still reject)
    """
    good_probe = row["p_good_text"]
    bad_probe  = row["p_bad_text"]
    p_good = verify(good_probe, contract)
    p_bad  = verify(bad_probe,  contract)
    retain = p_good["admissible"] and (not p_bad["admissible"])

    # Held-out synonym probes (two more teacher calls)
    good_syn = _teacher_reword_synonym(good_probe)
    bad_syn  = _teacher_reword_swap(row["source"], bad_probe)
    p_good_syn = verify(good_syn, contract) if not good_syn.startswith("ERROR:") else None
    p_bad_syn  = verify(bad_syn,  contract) if not bad_syn.startswith("ERROR:")  else None

    # Tautology / probe-leakage check
    taut = _tautology_check(contract, good_probe, bad_probe)

    return {
        "probes": {
            "p_good": p_good, "p_bad": p_bad,
            "good_text": good_probe, "bad_text": bad_probe,
            "p_good_synonym": p_good_syn, "p_bad_reworded": p_bad_syn,
            "good_synonym_text": good_syn, "bad_reworded_text": bad_syn,
        },
        "retain":       retain,
        "tautology":    taut,
    }


def _pick_paws_same_source_pairs(n_target: int, seed: int) -> list:
    """CW-directed: find PAWS sources that have BOTH label=1 (paraphrase) AND
    label=0 (role-swap) variants. Emit rows with real paraphrase / real role-swap
    from the SAME source — the discrimination test is against structural pairs,
    not label vocabulary."""
    from datasets import load_dataset
    ds = load_dataset(*HF_PAWS_ID, split="train")
    by_source_label = {}  # sentence1 → {label → sentence2}
    # Scan up to a bounded window (avoid loading all 49k twice)
    scan_limit = min(len(ds), 30_000)
    for i in range(scan_limit):
        r = ds[i]
        s1 = r["sentence1"]
        by_source_label.setdefault(s1, {})[r["label"]] = r["sentence2"]

    # Retain sources that have BOTH labels
    both = [(s1, d) for s1, d in by_source_label.items()
            if 0 in d and 1 in d]
    rng = random.Random(seed)
    rng.shuffle(both)
    picked = []
    for i, (s1, d) in enumerate(both[:n_target]):
        picked.append({
            "id":          f"paws_pair_{i:04d}",
            "subcorpus":   "paws",
            "source":      s1,
            "p_good_text": d[1],   # real paraphrase
            "p_bad_text":  d[0],   # real role-swap
            "label":       "pair", # marker
        })
    return picked


def _pick_mnli_ent_con_pairs(n_target: int, seed: int) -> list:
    """CW-directed: find MNLI premises with BOTH entailment (label=0) AND
    contradiction (label=2) hypotheses. Real structural discrimination pairs."""
    from datasets import load_dataset
    ds = load_dataset(HF_MNLI_ID, split="train")
    by_premise = {}  # premise → {label → hypothesis}
    scan_limit = min(len(ds), 50_000)
    for i in range(scan_limit):
        r = ds[i]
        by_premise.setdefault(r["premise"], {})[r["label"]] = r["hypothesis"]

    both = [(p, d) for p, d in by_premise.items()
            if 0 in d and 2 in d]
    rng = random.Random(seed)
    rng.shuffle(both)
    picked = []
    for i, (p, d) in enumerate(both[:n_target]):
        picked.append({
            "id":          f"mnli_pair_{i:04d}",
            "subcorpus":   "mnli",
            "source":      p,
            "p_good_text": d[0],   # entailment
            "p_bad_text":  d[2],   # contradiction
            "label":       "pair",
        })
    return picked


def _run_context_extraction(items: list, out_path: str, mode: str) -> dict:
    """Iterate items, dispatch by subcorpus, teacher-extract, self-verify, write JSONL."""
    os.makedirs(CONTEXT_DIR, exist_ok=True)

    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    total = len(items)
    print(f"[{mode}] processing {total} items ({len(done_ids)} already done); "
          f"teacher = {TEACHER_MODEL_LABEL}")

    n_run = n_parse_ok = n_verify_pass = n_calls = 0
    t0 = time.time()

    with open(out_path, "a") as fout:
        for idx, row in enumerate(items):
            if row["id"] in done_ids:
                continue
            n_calls += 1
            if n_calls > HARD_CAP_PER_LVL:
                print(f"[{mode}] hard-cap {HARD_CAP_PER_LVL} calls reached; stopping.")
                break
            subcorpus = row["subcorpus"]
            if subcorpus == "paws":
                contract, raw = _teacher_extract_context_paws(row)
            else:
                contract, raw = _teacher_extract_context_mnli(row)
            n_run += 1
            parse_ok = contract is not None
            if parse_ok:
                n_parse_ok += 1
                verdict = _two_probe_verify_context(row, contract, subcorpus)
                retain = verdict["retain"]
                if retain:
                    n_verify_pass += 1
                out_rec = {
                    "id":            row["id"],
                    "kind":          "context",
                    "subcorpus":     subcorpus,
                    "source":        row["source"],
                    "p_good_text":   row["p_good_text"],
                    "p_bad_text":    row["p_bad_text"],
                    "contract":      contract,
                    "teacher_raw":   raw,
                    "probes":        verdict["probes"],
                    "retain":        retain,
                    "tautology":     verdict["tautology"],
                    "teacher_model": TEACHER_MODEL_LABEL,
                }
            else:
                out_rec = {
                    "id":            row["id"],
                    "kind":          "context",
                    "subcorpus":     subcorpus,
                    "source":        row["source"],
                    "p_good_text":   row["p_good_text"],
                    "p_bad_text":    row["p_bad_text"],
                    "contract":      None,
                    "teacher_raw":   raw,
                    "probes":        None,
                    "retain":        False,
                    "tautology":     None,
                    "teacher_model": TEACHER_MODEL_LABEL,
                    "parse_error":   True,
                }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            done_ids.add(row["id"])
            elapsed = time.time() - t0
            avg = elapsed / n_run if n_run else 0
            mark = "OK" if (parse_ok and out_rec["retain"]) else ("VER" if parse_ok else "PAR")
            print(f"  [{mark}] {idx + 1:3d}/{total} id={row['id'][:22]:22s} "
                  f"sub={subcorpus:4s} label={row['label']} "
                  f"retain={out_rec['retain']} (avg={avg:.1f}s/call)")

    dt = time.time() - t0
    # Per-regime: subcorpus (paws / mnli) with FULL diagnostics (CW+CD)
    #
    # TRUE DISCRIMINATION = row-level conjunction (good admitted AND bad rejected),
    # NOT the average of independent admit/reject rates. Per CW directive: a
    # reject-all check has 0% discrimination, not 50%.
    per_regime = {}
    for r in items:
        key = r["subcorpus"]
        per_regime.setdefault(key, {
            "n": 0, "n_retain": 0,
            "n_p_good_rej": 0, "n_p_bad_adm": 0,
            "n_heldout_good_adm": 0, "n_heldout_bad_rej": 0,
            "n_heldout_true_discrim": 0, "n_heldout_measured": 0,
            "n_tautology": 0, "n_tautology_measured": 0,
        })
        per_regime[key]["n"] += 1
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                rec = json.loads(line)
                key = rec.get("subcorpus")
                if key not in per_regime:
                    continue
                if rec.get("retain"):
                    per_regime[key]["n_retain"] += 1
                probes = rec.get("probes")
                if probes and probes.get("p_good") and probes.get("p_bad"):
                    if not probes["p_good"]["admissible"]:
                        per_regime[key]["n_p_good_rej"] += 1
                    if probes["p_bad"]["admissible"]:
                        per_regime[key]["n_p_bad_adm"] += 1
                    if probes.get("p_good_synonym") and probes.get("p_bad_reworded"):
                        per_regime[key]["n_heldout_measured"] += 1
                        good_syn_adm = probes["p_good_synonym"]["admissible"]
                        bad_rew_rej  = not probes["p_bad_reworded"]["admissible"]
                        if good_syn_adm:
                            per_regime[key]["n_heldout_good_adm"] += 1
                        if bad_rew_rej:
                            per_regime[key]["n_heldout_bad_rej"] += 1
                        # TRUE discrimination — both conditions on same row
                        if good_syn_adm and bad_rew_rej:
                            per_regime[key]["n_heldout_true_discrim"] += 1
                taut = rec.get("tautology")
                if taut:
                    per_regime[key]["n_tautology_measured"] += 1
                    if taut.get("any_leaked"):
                        per_regime[key]["n_tautology"] += 1
            except Exception:
                pass
    for k, s in per_regime.items():
        s["rate"]                  = s["n_retain"]           / s["n"] if s["n"] else 0.0
        s["p_good_rej_rate"]       = s["n_p_good_rej"]       / s["n"] if s["n"] else 0.0
        s["p_bad_adm_rate"]        = s["n_p_bad_adm"]        / s["n"] if s["n"] else 0.0
        # In-sample TRUE discrimination = retention rate = row-level (good adm AND NOT bad adm)
        s["insample_true_discrim"] = s["rate"]
        n_ho = s["n_heldout_measured"]
        s["heldout_good_adm_rate"]     = s["n_heldout_good_adm"] / n_ho if n_ho else 0.0
        s["heldout_bad_rej_rate"]      = s["n_heldout_bad_rej"]  / n_ho if n_ho else 0.0
        # HELD-OUT TRUE discrimination = row-level conjunction (NOT the avg of the two rates)
        s["heldout_true_discrim_rate"] = s["n_heldout_true_discrim"] / n_ho if n_ho else 0.0
        n_t = s["n_tautology_measured"]
        s["tautology_rate"]            = s["n_tautology"] / n_t if n_t else 0.0

    stats = {
        "mode":           mode,
        "n_target":       total,
        "n_run":          n_run,
        "n_parse_ok":     n_parse_ok,
        "n_verify_pass":  n_verify_pass,
        "yield_parse":    n_parse_ok / n_run if n_run else 0.0,
        "yield_verify":   n_verify_pass / n_run if n_run else 0.0,
        "per_regime":     per_regime,
        "wall_clock_s":   round(dt, 1),
        "avg_s_per_call": round(dt / n_run, 2) if n_run else 0,
        "hard_cap":       HARD_CAP_PER_LVL,
    }
    print(f"\n[{mode}] {n_run} teacher calls (auth + reword × 2); parse_ok={n_parse_ok} "
          f"({stats['yield_parse']:.1%}); verify_pass={n_verify_pass} "
          f"({stats['yield_verify']:.1%}); {dt:.0f}s")
    print(f"\n[{mode}] PAWS (STRUCTURAL) and MNLI (GRADED ⊨) — reported separately:")
    for k in ("paws", "mnli"):
        if k not in per_regime: continue
        s = per_regime[k]
        tag = " [GRADED ⊨]" if k == "mnli" else " [STRUCTURAL]"
        print(f"\n  === {k}{tag}  (n={s['n']}) ===")
        print(f"    parse_ok / in-sample admit-reject rates:")
        print(f"      p_good ADMIT:                          {s['n']-s['n_p_good_rej']}/{s['n']} = {1-s['p_good_rej_rate']:.1%}")
        print(f"      p_bad  REJECT:                         {s['n']-s['n_p_bad_adm']}/{s['n']} = {1-s['p_bad_adm_rate']:.1%}")
        print(f"      IN-SAMPLE TRUE DISCRIMINATION:         {s['n_retain']}/{s['n']} = {s['insample_true_discrim']:.1%}")
        print(f"        (row-level: good_adm AND bad_rej — same as retain)")
        n_ho = s['n_heldout_measured']
        print(f"    held-out generalization (n={n_ho}):")
        print(f"      good_syn ADMIT:                        {s['n_heldout_good_adm']}/{n_ho} = {s['heldout_good_adm_rate']:.1%}")
        print(f"      bad_reworded REJECT:                   {s['n_heldout_bad_rej']}/{n_ho} = {s['heldout_bad_rej_rate']:.1%}")
        print(f"      HELD-OUT TRUE DISCRIMINATION:          {s['n_heldout_true_discrim']}/{n_ho} = {s['heldout_true_discrim_rate']:.1%}")
        print(f"        (row-level: good_syn_adm AND bad_reworded_rej — the GATE metric)")
        n_t = s['n_tautology_measured']
        print(f"    tautology rate (B/¬B string ∈ probes):   {s['n_tautology']}/{n_t} = {s['tautology_rate']:.1%}")
        print(f"      (natural base-probe token overlap flagged; NOT authoring leakage)")
    return stats


def cmd_smoke_context():
    """U4 smoke (CW-rebuild): 10 real PAWS same-source pairs + 10 real MNLI
    ent+con premise pairs = 20 structural discrimination tests."""
    paws = _pick_paws_same_source_pairs(n_target=10, seed=SEED)
    mnli = _pick_mnli_ent_con_pairs(n_target=10,  seed=SEED)
    sample = paws + mnli
    print(f"[smoke-context] sample = {len(sample)} items "
          f"({len(paws)} PAWS same-source + {len(mnli)} MNLI ent+con pairs)")
    smoke_path = os.path.join(CONTEXT_DIR, "smoke.jsonl")
    stats = _run_context_extraction(sample, smoke_path, mode="smoke")
    write_manifest(CONTEXT_DIR + "_smoke_manifest", stats)
    print(f"\n[smoke-context] wrote smoke.jsonl + manifest")
    return stats


def cmd_small_full_context():
    """U4 small-full (CW gate): 60 PAWS pairs + 20 MNLI ent+con pairs.
    Tightens the smoke's N=20 estimate into a defensible ~80-item CI before
    deciding whether to run the 3h full (gated by held-out TRUE discrimination > 0)."""
    paws = _pick_paws_same_source_pairs(n_target=60, seed=SEED)
    mnli = _pick_mnli_ent_con_pairs(n_target=20, seed=SEED)
    sample = paws + mnli
    print(f"[small-full] sample = {len(sample)} ({len(paws)} PAWS + {len(mnli)} MNLI)")
    path = os.path.join(CONTEXT_DIR, "small_full.jsonl")
    stats = _run_context_extraction(sample, path, mode="small-full")
    write_manifest(CONTEXT_DIR + "_small_full_manifest", stats)
    print(f"\n[small-full] wrote small_full.jsonl + manifest")
    return stats


def cmd_run_context():
    """U4 full (GATED — do NOT run unless small-full shows held-out TRUE
    discrimination > 0). Bounded ≈200 PAWS + ≈200 MNLI real structural pairs."""
    paws = _pick_paws_same_source_pairs(n_target=200, seed=SEED)
    mnli = _pick_mnli_ent_con_pairs(n_target=200, seed=SEED)
    sample = paws + mnli
    print(f"[run-context] sample = {len(sample)} items ({len(paws)} PAWS + {len(mnli)} MNLI)")
    full_path = os.path.join(CONTEXT_DIR, "items.jsonl")
    stats = _run_context_extraction(sample, full_path, mode="run")
    write_manifest(CONTEXT_DIR, stats)
    print(f"\n[run-context] wrote items.jsonl + manifest")
    return stats


# ══════════════════════════════════════════════════════════════════════════
# U5: SHARED PASSAGE POOL — DeepSeek teacher (NOT Claude-CLI; Claude budget
#     was near cap when this landed and DeepSeek runs independently of quota).
#
# Per passage, DeepSeek emits per-level FRAMINGS (Task / Concept / Context) —
# same passage framed at 3 levels. Topic-matched negatives = the same passage
# framed at a DIFFERENT level than what's being tested. This kills the
# corpus-vocab confound WP-17 exposed on processed_v2.
#
# Deterministic arm: Task + Concept — retention via verify(); U6/U7 use this
# for the deterministic PS bake-off. Graded arm: Context — records the
# framing without a strict deterministic retain (per U4 CW-frozen ⊨ finding).
#
# DeepSeek discipline: temp=0, save + sha256-hash every raw teacher output
# per record (API isn't reproducible over time). Checkpoint per passage.
# ══════════════════════════════════════════════════════════════════════════

import hashlib as _hashlib

POOL_TEACHER_LABEL = "deepseek-chat"  # via oracle_payoff.call_llm
POOL_MIXED_SEED    = 42


def _sha256_hex(text: str) -> str:
    return _hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_pool_passages(n_per_source: int, seed: int) -> list:
    """Draw N passages per source from the 4 sources (mixed pool).
    Source is stored per passage → level ⊥ source at the passage level."""
    rng = random.Random(seed)
    picks = []

    # SQuAD passages via contract_squad seed
    squad_items = _load_contract_squad_items()
    seen_sources = set()
    ans = [r for r in squad_items if r["kind"] == "answerable"]
    rng.shuffle(ans)
    n_squad = 0
    for r in ans:
        if r["source"] in seen_sources:
            continue
        seen_sources.add(r["source"])
        picks.append({
            "passage_id":   f"squad_{r['id']}",
            "source_label": "squad",
            "passage":      r["source"],
        })
        n_squad += 1
        if n_squad >= n_per_source:
            break

    # WSD passages
    wsd = _load_wsd_instances(n_max=50_000)
    rng.shuffle(wsd)
    seen_wsd = set()
    n_wsd = 0
    for r in wsd:
        if r["sentence"] in seen_wsd:
            continue
        seen_wsd.add(r["sentence"])
        picks.append({
            "passage_id":   f"wsd_{r['id']}",
            "source_label": "wsd",
            "passage":      r["sentence"],
        })
        n_wsd += 1
        if n_wsd >= n_per_source:
            break

    # PAWS sentence1s
    from datasets import load_dataset
    ds_paws = load_dataset(*HF_PAWS_ID, split="train")
    paws_idx = list(range(min(len(ds_paws), 20_000)))
    rng.shuffle(paws_idx)
    seen_paws = set()
    n_paws = 0
    for i in paws_idx:
        s = ds_paws[i]["sentence1"]
        if s in seen_paws:
            continue
        seen_paws.add(s)
        picks.append({
            "passage_id":   f"paws_{ds_paws[i]['id']}",
            "source_label": "paws",
            "passage":      s,
        })
        n_paws += 1
        if n_paws >= n_per_source:
            break

    # MNLI premises
    ds_mnli = load_dataset(HF_MNLI_ID, split="train")
    mnli_idx = list(range(min(len(ds_mnli), 30_000)))
    rng.shuffle(mnli_idx)
    seen_mnli = set()
    n_mnli = 0
    for i in mnli_idx:
        p = ds_mnli[i]["premise"]
        if p in seen_mnli:
            continue
        seen_mnli.add(p)
        picks.append({
            "passage_id":   f"mnli_{ds_mnli[i]['pairID']}",
            "source_label": "mnli",
            "passage":      p,
        })
        n_mnli += 1
        if n_mnli >= n_per_source:
            break

    return picks


# ── DeepSeek teacher framings ─────────────────────────────────────────────
POOL_PROMPT_TASK = """You are a contract extractor. Given a passage, frame it as a TASK-level contract for factual question-answering. Emit a JSON object with a self-contained contract PLUS a good probe (a valid answer text) and a bad probe (a fabricated answer text).

Return JSON with EXACTLY these keys — no code fences, no prose:

{{"framing": "task",
 "question": "<a factual question the passage answers>",
 "contract": {{
    "A": {{"source": "<passage verbatim>", "query": "<the question verbatim>"}},
    "F": "answer only from source (token-grounding)",
    "B": {{"content_tokens_must_be_grounded_in_source": "<passage verbatim>"}},
    "P": ["closed world"],
    "not_B": {{"must_not_contain_any": []}}
 }},
 "probe_good": "<the correct answer text (grounded in passage)>",
 "probe_bad":  "<a plausible fabricated answer with content NOT in passage>"}}

PASSAGE:
{passage}
"""

POOL_PROMPT_CONCEPT = """You are a contract extractor. Given a passage, frame it as a CONCEPT-level contract by picking a content word from the passage and specifying which sense/binding it must carry. Emit a JSON object with contract + good probe + bad probe.

Return JSON with EXACTLY these keys — no code fences, no prose:

{{"framing": "concept",
 "target_word": "<a content word from the passage>",
 "contract": {{
    "A": {{"source": "<passage verbatim>", "target": "<the word>", "query": "which sense of <the word>?"}},
    "F": "apply the in-context sense binding",
    "B": {{"must_contain_any": [<3-6 keywords defining the true sense>]}},
    "P": ["use the in-context sense binding"],
    "not_B": {{"must_not_contain_any": [<3-6 phrases that would name a distractor sense>]}}
 }},
 "probe_good": "<a definition of the true sense (should match must_contain_any)>",
 "probe_bad":  "<a definition of a distractor sense (should match not_B forbidden phrases)>"}}

PASSAGE:
{passage}
"""

POOL_PROMPT_CONTEXT = """You are a contract extractor. Given a passage, frame it as a CONTEXT-level contract by identifying a structural role-triple that any valid paraphrase must preserve. Emit contract + good probe + bad probe.

Return JSON with EXACTLY these keys — no code fences, no prose:

{{"framing": "context",
 "contract": {{
    "A": {{"source": "<passage verbatim>", "query": "is this a role-preserving paraphrase?"}},
    "F": "preserve subject/object roles",
    "B": {{"required_role_patterns": [{{"subject": "<S>", "relation": "<V>", "object": "<O>"}}]}},
    "P": ["preserve roles"],
    "not_B": {{"forbidden_role_swaps": true}}
 }},
 "probe_good": "<a role-preserving paraphrase of the passage>",
 "probe_bad":  "<a role-swap of the passage (same words, roles swapped)>"}}

PASSAGE:
{passage}
"""


def _deepseek_call(prompt: str, timeout: int = 90, max_retries: int = 2) -> str:
    """DeepSeek API call reusing WP-6 backend. temp=0, .env auth."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from oracle_payoff import call_llm  # WP-6 backend; reads .env at import time
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            out = call_llm(prompt, "")  # empty system prompt
            if out and not out.startswith("ERROR") and out != "TIMEOUT":
                return out
            last_err = out
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))
    return f"ERROR: {last_err}"


def _pool_extract_one(passage: str, prompt_tpl: str) -> tuple[dict | None, str, str]:
    """Call DeepSeek, parse JSON, return (parsed_dict, raw_text, sha256)."""
    prompt = prompt_tpl.format(passage=passage)
    raw = _deepseek_call(prompt)
    hsh = _sha256_hex(raw)
    if raw.startswith("ERROR"):
        return (None, raw, hsh)
    parsed = _parse_teacher_contract(raw)
    return (parsed, raw, hsh)


def _pool_verify_task(framing: dict) -> dict:
    """Verify Task framing: good probe should admit, bad probe should violate."""
    contract = framing.get("contract", {}) or {}
    probes = {}
    if "probe_good" in framing:
        probes["p_good"] = verify(framing["probe_good"], contract)
    if "probe_bad" in framing:
        probes["p_bad"] = verify(framing["probe_bad"], contract)
    retain = bool(probes.get("p_good", {}).get("admissible")) and (
        not bool(probes.get("p_bad", {}).get("admissible")))
    return {"probes": probes, "retain": retain}


def _pool_verify_concept(framing: dict) -> dict:
    """Verify Concept framing: same shape as Task."""
    return _pool_verify_task(framing)


def _pool_verify_context(framing: dict) -> dict:
    """Verify Context framing (recorded as GRADED; no strict deterministic retain)."""
    verd = _pool_verify_task(framing)
    verd["retain_graded_only"] = True
    return verd


def _run_shared_pool(passages: list, out_path: str, mode: str) -> dict:
    """Iterate passages; per passage emit 3 level-framings via DeepSeek.
    Resumable per (passage_id, level). Each record carries sha256 of raw
    teacher output (freeze discipline)."""
    os.makedirs(POOL_DIR, exist_ok=True)

    # Resume: skip passages already fully-emitted (3 levels each)
    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r["passage_id"], r["level"]))
                except Exception:
                    pass

    total_expected = len(passages) * 3
    print(f"[{mode}] passages={len(passages)}; expected records={total_expected}; "
          f"done={len(done_keys)}; teacher={POOL_TEACHER_LABEL}")

    n_new    = 0
    per_stats = {"task": {"retain": 0, "n": 0}, "concept": {"retain": 0, "n": 0},
                 "context": {"retain": 0, "n": 0}}
    per_source_level = {}   # (source_label, level) → {n, retain}
    t0 = time.time()
    levels = [
        ("task",    POOL_PROMPT_TASK,    _pool_verify_task),
        ("concept", POOL_PROMPT_CONCEPT, _pool_verify_concept),
        ("context", POOL_PROMPT_CONTEXT, _pool_verify_context),
    ]
    with open(out_path, "a") as fout:
        for pi, pp in enumerate(passages):
            for lvl_name, prompt_tpl, verify_fn in levels:
                key = (pp["passage_id"], lvl_name)
                if key in done_keys:
                    continue
                parsed, raw, hsh = _pool_extract_one(pp["passage"], prompt_tpl)
                verdict = verify_fn(parsed) if parsed else {"probes": None, "retain": False}
                retain = bool(verdict.get("retain"))
                rec = {
                    "passage_id":       pp["passage_id"],
                    "source_label":     pp["source_label"],
                    "level":            lvl_name,
                    "passage":          pp["passage"],
                    "framing":          parsed,
                    "teacher_raw":      raw,
                    "teacher_raw_sha256": hsh,
                    "probes":           verdict.get("probes"),
                    "retain":           retain,
                    "retain_graded_only": verdict.get("retain_graded_only", False),
                    "teacher_model":    POOL_TEACHER_LABEL,
                    "wp":               "WP-ST-16",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                done_keys.add(key)
                n_new += 1
                per_stats[lvl_name]["n"] += 1
                if retain:
                    per_stats[lvl_name]["retain"] += 1
                sk = (pp["source_label"], lvl_name)
                per_source_level.setdefault(sk, {"n": 0, "retain": 0})
                per_source_level[sk]["n"] += 1
                if retain:
                    per_source_level[sk]["retain"] += 1
                elapsed = time.time() - t0
                avg = elapsed / n_new if n_new else 0
                mark = "OK" if retain else ("PAR" if parsed is None else "VER")
                print(f"  [{mark}] {pi + 1:3d}/{len(passages)} "
                      f"{pp['source_label']:5s} {lvl_name:7s} "
                      f"retain={retain} (avg={avg:.1f}s/call)")

    dt = time.time() - t0
    print(f"\n[{mode}] wall={dt:.0f}s  {n_new} new records emitted.")

    # Per-level retention
    print(f"[{mode}] per-level retention (deterministic Task+Concept; Context graded):")
    for lvl in ("task", "concept", "context"):
        s = per_stats[lvl]
        tag = " [⊨ GRADED]" if lvl == "context" else " [⊢]"
        rate = s["retain"] / s["n"] if s["n"] else 0.0
        print(f"  {lvl:7s}{tag}:  {s['retain']}/{s['n']} = {rate:.1%}")

    # Source⊥level table
    print(f"\n[{mode}] SOURCE × LEVEL retention table (confound-killer check):")
    sources = sorted({sl[0] for sl in per_source_level})
    print(f"  {'source':6s}  {'task':>12s}  {'concept':>12s}  {'context':>12s}")
    for src in sources:
        cells = []
        for lvl in ("task", "concept", "context"):
            s = per_source_level.get((src, lvl), {"n": 0, "retain": 0})
            rate = s["retain"] / s["n"] if s["n"] else 0.0
            cells.append(f"{s['retain']}/{s['n']}={rate:.1%}")
        print(f"  {src:6s}  {cells[0]:>12s}  {cells[1]:>12s}  {cells[2]:>12s}")
    print(f"  → level ⊥ source by construction (each passage → all 3 levels; same source)")

    return {
        "mode":              mode,
        "n_passages":        len(passages),
        "n_new_records":     n_new,
        "per_level":         per_stats,
        "per_source_level":  {f"{k[0]}|{k[1]}": v for k, v in per_source_level.items()},
        "wall_clock_s":      round(dt, 1),
        "teacher_model":     POOL_TEACHER_LABEL,
        "provenance":        {"no_api_key": False, "auth": "DeepSeek API .env",
                              "note": "U5-only mixed-teacher (U2/U3/U4 = Claude-CLI OAuth)"},
    }


def cmd_shared_pool_smoke():
    """U5 smoke: 5 passages × 4 sources × 3 levels = 60 DeepSeek calls."""
    passages = _load_pool_passages(n_per_source=5, seed=POOL_MIXED_SEED)
    print(f"[smoke-pool] loaded {len(passages)} passages (expected 20)")
    path = os.path.join(POOL_DIR, "smoke.jsonl")
    stats = _run_shared_pool(passages, path, mode="pool-smoke")
    write_manifest(POOL_DIR + "_smoke_manifest", stats)
    return stats


def cmd_shared_pool():
    """U5 full: 50 passages × 4 sources = 200 passages × 3 levels = 600 DeepSeek calls."""
    passages = _load_pool_passages(n_per_source=50, seed=POOL_MIXED_SEED)
    print(f"[shared-pool] loaded {len(passages)} passages (expected 200)")
    path = os.path.join(POOL_DIR, "items.jsonl")
    stats = _run_shared_pool(passages, path, mode="shared-pool")
    write_manifest(POOL_DIR, stats)
    return stats


# ── U1 audit helper ───────────────────────────────────────────────────────
def cmd_list_teacher_cmd():
    """U1 audit: show the exact teacher invocation + provenance without a call."""
    print("=" * 68)
    print("Teacher invocation")
    print("=" * 68)
    print(f"  Command: {' '.join(TEACHER_CMD)}")
    print(f"  Timeout: {TEACHER_TIMEOUT_S}s per call")
    print(f"  Auth:    OAuth (Claude-CLI login on David's Mac); NO API key")
    print(f"  Label:   {TEACHER_MODEL_LABEL}")
    print()
    print("Data directories (all under data/ → gitignored):")
    print(f"  Task:    {TASK_DIR}/")
    print(f"  Concept: {CONCEPT_DIR}/")
    print(f"  Context: {CONTEXT_DIR}/")
    print(f"  Pool:    {POOL_DIR}/")
    print()
    print("Source corpora:")
    print(f"  Task    ← {CONTRACT_SQUAD_ITEMS} (WP-14 seed, frozen bdc404d760819e19)")
    print(f"  Concept ← {WSD_INSTANCES} (153k WSD rows)")
    print(f"  Context ← HF {HF_PAWS_ID} + {HF_MNLI_ID}")
    print()
    print("Hard caps (bounded teacher-call budget):")
    print(f"  per-level  = {HARD_CAP_PER_LVL} calls")
    print(f"  shared-pool= {HARD_CAP_POOL} calls  (200 passages × 3 levels + safety)")


# ── Main entry ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-teacher-cmd", action="store_true",
                    help="U1 audit: show teacher invocation + provenance")
    ap.add_argument("--smoke-task",       action="store_true",
                    help="U2 smoke: 20 items on contract_squad")
    ap.add_argument("--run-task",         action="store_true",
                    help="U2 full: bounded ~400 items")
    ap.add_argument("--smoke-concept",    action="store_true",
                    help="U3 smoke: 20 WSD rows")
    ap.add_argument("--run-concept",      action="store_true",
                    help="U3 full: bounded ~300 items")
    ap.add_argument("--smoke-context",    action="store_true",
                    help="U4 smoke: 10 PAWS + 10 MNLI")
    ap.add_argument("--small-full-context", action="store_true",
                    help="U4 small-full (CW gate): 60 PAWS + 20 MNLI")
    ap.add_argument("--run-context",      action="store_true",
                    help="U4 full (GATED): bounded ~400 items — only after CW go")
    ap.add_argument("--shared-pool-smoke", action="store_true",
                    help="U5 smoke: 5 passages × 4 sources × 3 levels = 60 calls")
    ap.add_argument("--shared-pool",      action="store_true",
                    help="U5 full: 50 passages × 4 sources × 3 levels = 600 calls")
    args = ap.parse_args()

    if args.list_teacher_cmd:
        cmd_list_teacher_cmd()
        return 0
    if args.smoke_task:    cmd_smoke_task()
    if args.run_task:      cmd_run_task()
    if args.smoke_concept: cmd_smoke_concept()
    if args.run_concept:   cmd_run_concept()
    if args.smoke_context: cmd_smoke_context()
    if getattr(args, "small_full_context", False): cmd_small_full_context()
    if args.run_context:   cmd_run_context()
    if getattr(args, "shared_pool_smoke", False): cmd_shared_pool_smoke()
    if args.shared_pool:   cmd_shared_pool()

    if not any([args.list_teacher_cmd, args.smoke_task, args.run_task,
                args.smoke_concept, args.run_concept, args.smoke_context,
                getattr(args, "small_full_context", False),
                args.run_context, args.shared_pool]):
        print(f"[banner] WP-ST-16 dataset builder — teacher: {TEACHER_MODEL_LABEL}")
        print(f"[banner] --list-teacher-cmd shows the teacher invocation + provenance")
        print(f"[banner] --smoke-{{task,concept,context}} then --run-*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
