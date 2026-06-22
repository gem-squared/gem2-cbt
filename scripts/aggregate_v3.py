"""WP-ST-3 U7 aggregation — g3b (v3-hard decisive) + g6 (v3-easy only).

Usage:
  python scripts/aggregate_v3.py          # checks both splits, writes papers/results_v3.md
  python scripts/aggregate_v3.py --check  # just report how many seeds are done, no writes
"""
import argparse, json, math, os, sys
from collections import defaultdict

CONFIGS = [
    "heuristic_baseline",
    "text_only_nn",
    "gloss_similarity_baseline",
    "text_contract",
    "cbt",
    "easy_random_contract",
    "hard_same_lemma_random",
    "contract_only",
    "target_word_only",
]
N_SEEDS = 10
SPLITS  = ["v3_hard", "v3_easy"]

# Gate thresholds (per WP)
GATE_MIN_SEEDS   = 8   # must win in ≥8/10 seeds
GATE_EFFECT_SIZE = 0.03  # mean delta ≥ +0.03

def load_results(split):
    """Load all per-seed result files for a split. Returns (majority_acc, all_results_dict)."""
    base = f"data/processed_v3/{split}"
    all_results = {}
    majority = None
    for seed in range(N_SEEDS):
        path = os.path.join(base, f"results_v3_{split}_seed{seed}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        if majority is None:
            majority = d.get("majority")
        for k, v in d.get("results", {}).items():
            all_results[k] = v
    return majority, all_results


def seeds_done(split):
    """Return count of seeds that have ALL configs complete."""
    base = f"data/processed_v3/{split}"
    complete = 0
    for seed in range(N_SEEDS):
        path = os.path.join(base, f"results_v3_{split}_seed{seed}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        results = d.get("results", {})
        seed_keys = [k for k in results if k.endswith(f"_seed{seed}")]
        if len(seed_keys) == len(CONFIGS):
            complete += 1
    return complete


def cohen_d(deltas):
    """Cohen's d from list of per-seed deltas."""
    n = len(deltas)
    if n < 2:
        return float("nan")
    mean = sum(deltas) / n
    var  = sum((x - mean)**2 for x in deltas) / (n - 1)
    std  = math.sqrt(var) if var > 0 else 1e-9
    return mean / std


def aggregate_config(config, results, seeds=range(N_SEEDS)):
    """Return dict of accs, mean, std, aurocs for a config across seeds."""
    accs, aurocs = [], []
    for s in seeds:
        key = f"{config}_seed{s}"
        if key not in results:
            continue
        r = results[key]
        if r is None:
            continue
        accs.append(r.get("boundary_acc", float("nan")))
        auc = r.get("auroc")
        if auc is not None:
            aurocs.append(auc)
    if not accs:
        return None
    mean  = sum(accs) / len(accs)
    var   = sum((x - mean)**2 for x in accs) / max(len(accs)-1, 1)
    std   = math.sqrt(var)
    amean = sum(aurocs) / len(aurocs) if aurocs else None
    return {"mean": mean, "std": std, "n": len(accs), "accs": accs,
            "auroc_mean": amean, "aurocs": aurocs}


def gate_verdict(cfg_a_stats, cfg_b_stats):
    """Gate: cfg_a > cfg_b.
    Returns (wins, mean_delta, cohen_d, PASS/FAIL)."""
    if cfg_a_stats is None or cfg_b_stats is None:
        return None, None, None, "MISSING_DATA"
    # Paired comparison (same seeds)
    n = min(cfg_a_stats["n"], cfg_b_stats["n"])
    if n < GATE_MIN_SEEDS:
        return n, None, None, f"INSUFFICIENT ({n}/10)"
    deltas = [a - b for a, b in zip(cfg_a_stats["accs"][:n], cfg_b_stats["accs"][:n])]
    wins = sum(1 for d in deltas if d > 0)
    mean_delta = sum(deltas) / n
    cd = cohen_d(deltas)
    passed = (wins >= GATE_MIN_SEEDS) and (mean_delta >= GATE_EFFECT_SIZE)
    return wins, mean_delta, cd, "PASS ✓" if passed else "FAIL ✗"


def format_gate_row(name, wins, delta, cd, verdict):
    if wins is None:
        return f"| {name} | — | — | — | {verdict} |"
    cd_str = f"{cd:.2f}" if cd == cd else "nan"
    return f"| {name} | {wins}/10 | {delta:+.3f} | {cd_str} | {verdict} |"


def write_results_md(results_by_split, majority_by_split, out_path):
    lines = ["# WP-ST-3 Ecological Validity — Results v3\n"]
    lines.append(f"**Encoder:** all-MiniLM-L6-v2 (384-dim, frozen)  \n")
    lines.append(f"**Splits:** v3-hard (canonical hash `25fc21e245581f64`) · v3-easy (canonical hash `fba8e3f6236ae5a8`)  \n")
    lines.append(f"**Seeds:** 0–9 · **Epochs:** 5 · **Architecture:** MLP 1536→512→256→1  \n\n")

    for split in SPLITS:
        res    = results_by_split[split]
        maj    = majority_by_split[split]
        lines.append(f"---\n## {split}\n\n")
        lines.append(f"**MFS baseline (majority):** {maj:.3f}  \n\n")

        # ── Accuracy table ──
        lines.append("### Accuracy (mean ± std, 10 seeds)\n\n")
        lines.append("| Config | Mean Acc | ±Std | AUROC mean |\n")
        lines.append("|---|---|---|---|\n")
        stats = {}
        for cfg in CONFIGS:
            s = aggregate_config(cfg, res)
            stats[cfg] = s
            if s:
                auc_str = f"{s['auroc_mean']:.3f}" if s["auroc_mean"] is not None else "N/A"
                lines.append(f"| {cfg} | {s['mean']:.3f} | ±{s['std']:.3f} | {auc_str} |\n")
            else:
                lines.append(f"| {cfg} | — | — | — |\n")
        lines.append("\n")

        # ── Gate table ──
        lines.append("### Gate Verdicts\n\n")
        lines.append("| Gate | Seeds winning | Mean Δ | Cohen's d | Verdict |\n")
        lines.append("|---|---|---|---|---|\n")

        if split == "v3_hard":
            lines.append("**NOTE: g6 (cbt > text_only) is NOT reported for v3-hard — text_only_nn is zero-shot-degenerate on unseen lemmas (by-construction failure, not ecological signal).**\n\n")
            lines.append("| Gate | Seeds winning | Mean Δ | Cohen's d | Verdict |\n")
            lines.append("|---|---|---|---|---|\n")
            # g3a: cbt > easy_random (sanity floor)
            wins, delta, cd, v = gate_verdict(stats["cbt"], stats["easy_random_contract"])
            lines.append(format_gate_row("g3a: cbt > easy_random", wins, delta, cd, v) + "\n")
            # g3b: cbt > hard_same_lemma_random (DECISIVE)
            wins, delta, cd, v = gate_verdict(stats["cbt"], stats["hard_same_lemma_random"])
            lines.append(format_gate_row("**g3b (DECISIVE): cbt > hard_same_lemma**", wins, delta, cd, v) + "\n")
            # parsimony: cbt > gloss_sim
            wins, delta, cd, v = gate_verdict(stats["cbt"], stats["gloss_similarity_baseline"])
            lines.append(format_gate_row("parsimony: cbt > gloss_sim", wins, delta, cd, v) + "\n")
            # target_word_only check (must NOT match cbt)
            wins, delta, cd, v = gate_verdict(stats["cbt"], stats["target_word_only"])
            lines.append(format_gate_row("parsimony: cbt > target_word_only", wins, delta, cd, v) + "\n")

        else:  # v3_easy
            # g6: cbt > text_only_nn (WEAK baseline, +0.069 over MFS)
            wins, delta, cd, v = gate_verdict(stats["cbt"], stats["text_only_nn"])
            lines.append(format_gate_row("g6: cbt > text_only_nn (weak frozen baseline)", wins, delta, cd, v) + "\n")
            # g3a sanity
            wins, delta, cd, v = gate_verdict(stats["cbt"], stats["easy_random_contract"])
            lines.append(format_gate_row("g3a: cbt > easy_random", wins, delta, cd, v) + "\n")
            # g3b on easy
            wins, delta, cd, v = gate_verdict(stats["cbt"], stats["hard_same_lemma_random"])
            lines.append(format_gate_row("g3b: cbt > hard_same_lemma", wins, delta, cd, v) + "\n")
            # parsimony
            wins, delta, cd, v = gate_verdict(stats["cbt"], stats["gloss_similarity_baseline"])
            lines.append(format_gate_row("parsimony: cbt > gloss_sim", wins, delta, cd, v) + "\n")

        lines.append("\n")

    # ── Summary ──
    lines.append("---\n## Summary\n\n")
    # Compute decisive gates for summary
    r_hard = results_by_split["v3_hard"]
    r_easy = results_by_split["v3_easy"]
    s_hard = {c: aggregate_config(c, r_hard) for c in CONFIGS}
    s_easy = {c: aggregate_config(c, r_easy) for c in CONFIGS}
    _, g3b_delta, _, g3b_v = gate_verdict(s_hard["cbt"], s_hard["hard_same_lemma_random"])
    _, pars_delta, _, pars_v = gate_verdict(s_hard["cbt"], s_hard["gloss_similarity_baseline"])
    _, g6_delta, _, g6_v = gate_verdict(s_easy["cbt"], s_easy["text_only_nn"])
    _, tw_delta, _, tw_v = gate_verdict(s_easy["cbt"], s_easy["target_word_only"])

    lines.append(f"**v3-hard g3b (DECISIVE):** cbt > hard_same_lemma → {g3b_v} (Δ{g3b_delta:+.3f})  \n")
    lines.append(f"**v3-hard parsimony (cbt > gloss_sim):** {pars_v} (Δ{pars_delta:+.3f})  \n")
    g6_d_str = f"{g6_delta:+.3f}" if g6_delta is not None else "N/A"
    lines.append(f"**v3-easy g6 (cbt > text_only_nn):** {g6_v} (Δ{g6_d_str})  \n")
    lines.append(f"**v3-easy parsimony (cbt > target_word_only):** {tw_v}  \n\n")
    lines.append("**text_only_nn on v3-hard: NOT reported as g6 — zero-shot degenerate by lemma-split construction.**  \n\n")
    lines.append("*See papers/claim_v3.md for bounded claim + CBT-v1 gate decision.*\n")

    os.makedirs("papers", exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"Written: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report progress only, no writes")
    args = ap.parse_args()

    print("=== WP-ST-3 Aggregation Status ===")
    all_complete = True
    results_by_split = {}
    majority_by_split = {}
    for split in SPLITS:
        done = seeds_done(split)
        print(f"{split}: {done}/10 seeds complete")
        if done < N_SEEDS:
            all_complete = False
        maj, res = load_results(split)
        results_by_split[split] = res
        majority_by_split[split] = maj or 0.0
        # Quick stats per config
        for cfg in CONFIGS:
            s = aggregate_config(cfg, res)
            if s:
                print(f"  {cfg}: n={s['n']} seeds, mean_acc={s['mean']:.3f}±{s['std']:.3f}")

    if args.check:
        if all_complete:
            print("\nAll seeds complete — ready for aggregation.")
        else:
            print("\nSweep still running.")
        return

    if not all_complete:
        print("\nNot all seeds complete. Run with --check to monitor, or rerun when done.")
        # Partial aggregation for monitoring
        print("Running partial aggregation for available seeds...")

    out_path = "papers/results_v3.md"
    write_results_md(results_by_split, majority_by_split, out_path)
    print("\nDone. Review papers/results_v3.md before proceeding to claim_v3.md (U8).")


if __name__ == "__main__":
    main()
