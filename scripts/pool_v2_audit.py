#!/usr/bin/env python3
"""WP-ST-18 U2: Leakage audit + strip on pool v2.

Compute per-token level-conditional distribution over framing.prompt tokens.
Strip (mask) any token whose  P(tok|level) − max_over_other_levels ≥ CEILING.
Iterate to fixpoint. Record what was stripped.

Outputs:
  data/ece_shared_pool_v2/audit.json        — per-token stats, iteration log
  data/ece_shared_pool_v2/stripped.jsonl    — each framing with a "stripped_prompt" field
  data/ece_shared_pool_v2/audit_gate.json   — D2 gate + strip non-degenerate check
"""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = ROOT / "data" / "ece_shared_pool_v2"

LEVELS = ("task", "concept", "context")
CEILING_DEFAULT = 0.30
MIN_COUNT = 3   # ignore tokens seen < MIN_COUNT times in the pool

_TOK_RE = re.compile(r"[a-z0-9']+")

def tokens(s: str) -> list[str]:
    return _TOK_RE.findall(s.lower())


def token_level_stats(records: list[dict], banned: set[str]) -> dict:
    """Return {tok: {"task":p_t, "concept":p_c, "context":p_x, "max_delta":d}}"""
    per_level_counts: dict[str, Counter] = {l: Counter() for l in LEVELS}
    per_level_n: dict[str, int] = {l: 0 for l in LEVELS}
    for r in records:
        lvl = r["level"]
        toks = [t for t in tokens(r["framing"]["prompt"]) if t not in banned]
        per_level_n[lvl] += 1
        seen = set(toks)  # per-record presence (not TF) — smoother signal
        per_level_counts[lvl].update(seen)

    # union of tokens
    vocab: set[str] = set()
    for c in per_level_counts.values():
        vocab.update(c.keys())

    stats = {}
    for t in vocab:
        p = {l: per_level_counts[l][t] / max(per_level_n[l], 1) for l in LEVELS}
        # min-count filter across levels
        total = sum(per_level_counts[l][t] for l in LEVELS)
        if total < MIN_COUNT:
            continue
        # compute per-level lift (p_lvl − max_other)
        deltas = {}
        for l in LEVELS:
            other = max(p[o] for o in LEVELS if o != l)
            deltas[l] = p[l] - other
        best_lvl = max(deltas, key=deltas.get)
        stats[t] = {"p": p, "delta": deltas, "leaked_level": best_lvl,
                    "max_delta": deltas[best_lvl], "total_count": total}
    return stats


def strip_iterate(records: list[dict], ceiling: float, max_iters: int = 15) -> dict:
    """Iterate: at each pass, find tokens with max_delta ≥ ceiling and ban them
    from the tokenization; recompute stats; repeat until no token exceeds ceiling
    (or max_iters). Return {banned: [...], log: [{iter, removed, residual_max}]}"""
    banned: set[str] = set()
    log = []
    for it in range(max_iters):
        stats = token_level_stats(records, banned)
        if not stats:
            log.append({"iter": it, "removed": [], "residual_max": 0.0, "vocab_size": 0})
            break
        # find tokens above ceiling
        offenders = [(t, s["max_delta"], s["leaked_level"], s["total_count"])
                     for t, s in stats.items() if s["max_delta"] >= ceiling]
        residual_max = max((s["max_delta"] for s in stats.values()), default=0.0)
        if not offenders:
            log.append({"iter": it, "removed": [], "residual_max": residual_max,
                        "vocab_size": len(stats)})
            break
        # remove all offenders in this pass (batch)
        offenders.sort(key=lambda x: -x[1])
        new_bans = [t for t, *_ in offenders]
        banned.update(new_bans)
        log.append({"iter": it, "removed": offenders[:30],
                    "n_removed": len(new_bans),
                    "residual_max": residual_max,
                    "vocab_size": len(stats)})
    return {"banned": sorted(banned), "log": log,
            "final_residual_max": log[-1]["residual_max"] if log else 1.0}


def stripped_prompt(prompt: str, banned: set[str]) -> str:
    """Replace banned tokens with __STRIP__ marker (preserves position)."""
    out_toks = []
    for m in _TOK_RE.finditer(prompt.lower()):
        t = m.group()
        out_toks.append("__STRIP__" if t in banned else t)
    return " ".join(out_toks)


def survivable_check(records: list[dict], banned: set[str]) -> dict:
    """Strip non-degenerate: at least K tokens survive per framing (median)."""
    survivor_counts = []
    zero_surv = 0
    for r in records:
        toks = tokens(r["framing"]["prompt"])
        n_surv = sum(1 for t in toks if t not in banned)
        survivor_counts.append(n_surv)
        if n_surv == 0:
            zero_surv += 1
    survivor_counts.sort()
    median = survivor_counts[len(survivor_counts) // 2] if survivor_counts else 0
    return {"n_records": len(survivor_counts),
            "median_surviving_tokens": median,
            "mean_surviving_tokens": sum(survivor_counts) / max(len(survivor_counts), 1),
            "min_surviving_tokens": min(survivor_counts) if survivor_counts else 0,
            "max_surviving_tokens": max(survivor_counts) if survivor_counts else 0,
            "n_zero_survivors": zero_surv,
            "non_degenerate": zero_surv == 0 and median >= 3}


def cmd_run(args):
    items_path = POOL_DIR / "items.jsonl"
    with open(items_path) as f:
        records = [json.loads(line) for line in f]
    print(f"[U2] loaded {len(records)} records; ceiling={args.ceiling}")

    result = strip_iterate(records, ceiling=args.ceiling, max_iters=args.max_iters)
    banned = set(result["banned"])
    print(f"[U2] banned {len(banned)} tokens across {len(result['log'])} iterations")
    print(f"[U2] final residual max_delta = {result['final_residual_max']:.4f}")

    survive = survivable_check(records, banned)
    print(f"[U2] survivor stats: median={survive['median_surviving_tokens']}, "
          f"min={survive['min_surviving_tokens']}, "
          f"n_zero={survive['n_zero_survivors']}, non_degen={survive['non_degenerate']}")

    # write stripped view
    strip_path = POOL_DIR / "stripped.jsonl"
    with open(strip_path, "w") as out:
        for r in records:
            r2 = dict(r)
            r2["stripped_prompt"] = stripped_prompt(r["framing"]["prompt"], banned)
            out.write(json.dumps(r2, ensure_ascii=False) + "\n")

    # audit + gate
    audit = {"ceiling": args.ceiling, "n_banned": len(banned),
             "banned_first30": sorted(banned)[:30],
             "iteration_log": result["log"],
             "final_residual_max": result["final_residual_max"],
             "survivor_stats": survive}
    (POOL_DIR / "audit.json").write_text(json.dumps(audit, indent=2))

    d2_pass = result["final_residual_max"] < args.ceiling
    non_degen = survive["non_degenerate"]
    gate = {
        "D2_ceiling": args.ceiling,
        "D2_residual_max": result["final_residual_max"],
        "D2_pass": d2_pass,
        "strip_non_degenerate": non_degen,
        "gate_pass": d2_pass and non_degen,
        "n_banned": len(banned),
        "survivor_median": survive["median_surviving_tokens"],
        "n_zero_survivors": survive["n_zero_survivors"],
    }
    (POOL_DIR / "audit_gate.json").write_text(json.dumps(gate, indent=2))
    print("\n[U2] GATE:")
    print(json.dumps(gate, indent=2))
    return 0 if gate["gate_pass"] else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ceiling", type=float, default=CEILING_DEFAULT)
    r.add_argument("--max-iters", type=int, default=15)
    args = ap.parse_args()
    if args.cmd == "run":
        import sys; sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
