"""Aggregate multi-seed results and compute paired deltas (U7).

Reads data/processed/results.json (keyed by {config}_seed{N}).
Emits papers/results_robust.md with mean ± std + per-seed paired deltas.

Good-direction convention:
  unsafe_accept_rate  ↓  → negative delta is good
  over_reject_rate    ↓  → negative delta is good
  boundary_acc        ↑  → positive delta is good
  concept/context/task acc ↑ → positive delta is good

Success criterion: same-sign (good-direction) in ≥4/5 seeds.
"""
import argparse
import json
import math
import os
import sys

SEEDS = [0, 1, 2, 3, 4]

MAIN_CONFIGS = ["baseline_lm", "cbt_textonly", "cbt_v0"]
SHUFFLE_CONFIG = "cbt_v0_concept_contract_shuffled"
SHUFFLE_LABEL = "cbt_v0_shuffled (within-level; concept-contract only)"

# (metric_key, good_direction, display_name)
METRICS = [
    ("lm_loss",           "↓", "LM Loss"),
    ("boundary_acc",      "↑", "Boundary Acc"),
    ("unsafe_accept_rate","↓", "Unsafe Accept Rate"),
    ("over_reject_rate",  "↓", "Over Reject Rate"),
]
LEVEL_METRICS = ["concept", "context", "task"]


def get_val(results, config, seed, metric):
    key = f"{config}_seed{seed}"
    r = results.get(key)
    if r is None:
        return None
    if metric in r:
        return r[metric]
    pl = r.get("per_level_acc", {})
    if metric in pl:
        return pl[metric]
    return None


def mean_std(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, math.sqrt(var)


def fmt(x, digits=3):
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def sign_check(deltas, good_dir):
    """Return (n_good_sign, passes) where passes iff n_good_sign >= 4."""
    good_sign = 1 if good_dir == "↑" else -1
    valid = [d for d in deltas if d is not None]
    n_good = sum(1 for d in valid if d * good_sign > 0)
    return n_good, len(valid), n_good >= 4


def build_report(results, frozen_hash, majority):
    lines = []
    lines.append("# CBT-v0 Multi-Seed Robustness Report\n")
    lines.append(f"_Seeds: {SEEDS} | Frozen dataset hash: `{frozen_hash}` | "
                 f"Majority-class boundary acc: {fmt(majority)}_\n")

    # --- 1. Mean ± std table ---
    lines.append("## 1. Per-Config Mean ± Std (seeds 0–4)\n")
    all_configs = MAIN_CONFIGS + [SHUFFLE_CONFIG]
    display_names = {c: c for c in MAIN_CONFIGS}
    display_names[SHUFFLE_CONFIG] = SHUFFLE_LABEL

    all_metrics = METRICS + [(f"level_{lv}", "↑", lv.capitalize() + " Acc")
                              for lv in LEVEL_METRICS]
    metric_keys_for_table = [m[0] for m in all_metrics]

    # Header
    header = "| Config | " + " | ".join(f"{name} {dir_}" for _, dir_, name in all_metrics) + " |"
    sep    = "|---|" + "|".join(["---"] * len(all_metrics)) + "|"
    lines.append(header); lines.append(sep)

    for cfg in all_configs:
        row_vals = []
        for mk, _, _ in all_metrics:
            real_mk = mk.replace("level_", "")
            vals = [get_val(results, cfg, s, real_mk) for s in SEEDS]
            m, sd = mean_std(vals)
            if m is None:
                row_vals.append("—")
            else:
                row_vals.append(f"{fmt(m)} ±{fmt(sd)}")
        label = display_names.get(cfg, cfg)
        lines.append(f"| {label} | " + " | ".join(row_vals) + " |")
    lines.append("")

    # --- 2. Per-seed paired deltas ---
    lines.append("## 2. Per-Seed Paired Deltas\n")
    lines.append("### 2a. cbt_v0 vs cbt_textonly (contract injection effect)\n")

    delta_defs_main = [
        ("unsafe_accept_rate", "cbt_v0", "cbt_textonly", "↓",
         "Δunsafe_accept = cbt_v0 − cbt_textonly (negative = good)"),
        ("over_reject_rate",   "cbt_v0", "cbt_textonly", "↓",
         "Δover_reject = cbt_v0 − cbt_textonly (negative = good)"),
        ("concept",            "cbt_v0", "cbt_textonly", "↑",
         "Δconcept_acc = cbt_v0 − cbt_textonly (positive = good)"),
    ]

    for mk, cfg_a, cfg_b, good_dir, label in delta_defs_main:
        deltas = []
        for s in SEEDS:
            va = get_val(results, cfg_a, s, mk)
            vb = get_val(results, cfg_b, s, mk)
            deltas.append(va - vb if va is not None and vb is not None else None)
        n_good, n_valid, passes = sign_check(deltas, good_dir)
        status = "PASS ✓" if passes else "FAIL ✗"
        delta_strs = [fmt(d, 4) if d is not None else "—" for d in deltas]
        lines.append(f"**{label}**")
        lines.append(f"  Seeds: {' | '.join(f's{s}={delta_strs[i]}' for i, s in enumerate(SEEDS))}")
        lines.append(f"  Good-direction ({good_dir}): {n_good}/{n_valid} seeds — **{status}** "
                     f"(criterion: ≥4/5)\n")

    lines.append("### 2b. cbt_v0 vs cbt_v0_shuffled (concept-contract signal test)\n")

    delta_defs_shuf = [
        ("unsafe_accept_rate", "cbt_v0", SHUFFLE_CONFIG, "↓",
         "Δshuffle_unsafe_accept = cbt_v0 − cbt_v0_shuffled (negative = good)"),
        ("concept",            "cbt_v0", SHUFFLE_CONFIG, "↑",
         "Δshuffle_concept_acc = cbt_v0 − cbt_v0_shuffled (positive = good)"),
    ]

    for mk, cfg_a, cfg_b, good_dir, label in delta_defs_shuf:
        deltas = []
        for s in SEEDS:
            va = get_val(results, cfg_a, s, mk)
            vb = get_val(results, cfg_b, s, mk)
            deltas.append(va - vb if va is not None and vb is not None else None)
        n_good, n_valid, passes = sign_check(deltas, good_dir)
        status = "PASS ✓" if passes else "FAIL ✗"
        if n_valid == 0:
            status = "PENDING (no shuffled results yet)"
        delta_strs = [fmt(d, 4) if d is not None else "—" for d in deltas]
        lines.append(f"**{label}**")
        lines.append(f"  Seeds: {' | '.join(f's{s}={delta_strs[i]}' for i, s in enumerate(SEEDS))}")
        lines.append(f"  Good-direction ({good_dir}): {n_good}/{n_valid} seeds — **{status}** "
                     f"(criterion: ≥4/5)\n")

    # --- 3. Structural notes ---
    lines.append("## 3. Structural Notes\n")
    lines.append("- **Task level** has trivial surface cue (extra clause; length + n-gram = 1.000). "
                 "Task acc is not a measure of semantic boundary understanding.")
    lines.append("- **Context/task contracts** are single-valued (`role-preserve`, `facts-only`) → "
                 "within-level shuffle is a no-op there; ablation tests ONLY concept-contract signal.")
    lines.append("- **baseline_lm** has no boundary head — LM loss shown for reference only.")
    lines.append("- CBT-v1 gated until U8 claim write-up passes review.\n")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="papers/results_robust.md")
    args = ap.parse_args()

    res_path = os.path.join(args.data, "results.json")
    if not os.path.exists(res_path):
        print(f"ERROR: {res_path} not found — run train_compare.py first")
        sys.exit(1)

    d = json.load(open(res_path))
    results = d.get("results", {})
    majority = d.get("majority", 0.0)

    # Get frozen hash
    frozen_path = os.path.join(args.data, "frozen_dataset_hash.json")
    frozen_hash = "unknown"
    if os.path.exists(frozen_path):
        frozen_hash = json.load(open(frozen_path)).get("frozen_hash", "unknown")

    # Report completeness
    expected_main = [f"{c}_seed{s}" for c in MAIN_CONFIGS for s in SEEDS]
    expected_shuf = [f"{SHUFFLE_CONFIG}_seed{s}" for s in SEEDS]
    have_main = [k for k in expected_main if k in results]
    have_shuf = [k for k in expected_shuf if k in results]
    print(f"Main sweep: {len(have_main)}/15 runs complete")
    print(f"Shuffle:    {len(have_shuf)}/5 runs complete")
    if len(have_main) < 15:
        missing = [k for k in expected_main if k not in results]
        print(f"Missing main: {missing[:5]}{'...' if len(missing)>5 else ''}")

    report = build_report(results, frozen_hash, majority)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nWrote: {args.out}")
    print(report)


if __name__ == "__main__":
    main()
