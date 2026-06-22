"""WP-ST-4: HPIC complex/phase abstain-gate A/B test.

Does HPIC's complex geometry give a better selective-risk (risk–coverage) curve
than plain thresholds — especially on the CONFLICTING-evidence regime?

Usage:
  python scripts/hpic_gate_ab.py --generate    # U1: build + freeze data
  python scripts/hpic_gate_ab.py --smoke        # U4a: smoke (seed 0 only, small N)
  python scripts/hpic_gate_ab.py --sweep        # U4b: full sweep seeds 0-9
  python scripts/hpic_gate_ab.py --aggregate    # U5: aggregate + write results_hpic_gate.md
  python scripts/hpic_gate_ab.py --all          # U1+U4b+U5 (generate if needed, full run)
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────
N_INSTANCES   = 3000   # per seed: 1000 CLEAR + 1000 INSUFF + 1000 CONFLICT
K_SOURCES     = 5      # evidence sources per instance
N_SEEDS       = 10
N_SMOKE       = 300    # 100 per regime, smoke only

DATA_DIR      = "data/hpic_gate"
RESULTS_DIR   = "data/hpic_gate"
PAPERS_DIR    = "papers"
FROZEN_FILE   = os.path.join(DATA_DIR, "frozen_task_hash.json")
RESULTS_FILE  = os.path.join(RESULTS_DIR, "results_hpic_gate_seed{}.json")

# Generator config — this is what gets hashed (NOT the specific random instances)
GENERATOR_CONFIG = {
    "n_instances": N_INSTANCES,
    "k_sources": K_SOURCES,
    "n_seeds": N_SEEDS,
    "clear_strong_p": [0.75, 0.95],
    "clear_weak_p": [0.40, 0.60],
    "insuff_p": [0.35, 0.65],
    "conflict_strong_p": [0.75, 0.95],
    "conflict_weak_p": [0.05, 0.25],
    "rho_formula": "abs(2p-1)",
    "version": "v1",
}

GATES = ["linear", "max_prob", "hpic_point", "hpic_interval"]

# Gate verdict thresholds (WP-ST-4 U5)
GATE_MIN_SEEDS   = 8
GATE_EFFECT_SIZE = 0.03  # mean delta ≥ +0.03

# CLEAR coverage operating point for g_B conflict-recall comparison
CLEAR_COVERAGE_TARGET = 0.80


# ── U1: Data generator ─────────────────────────────────────────────────────

def generate_instances(rng, n_total):
    """
    Generate synthetic 1-axis ⊥-gate instances.

    Returns list of dicts with:
      p_k: (K,) probabilities toward class A per source
      rho_k: (K,) pre-registered strength = |2p_k-1|
      regime: 'clear' | 'insuff' | 'conflict'
      label: 0 (A) or 1 (B) for clear; -1 for abstain regimes
      should_abstain: bool
    """
    n_each = n_total // 3
    instances = []

    # ── CLEAR regime ──
    for _ in range(n_each):
        label = rng.integers(0, 2)   # 0=A, 1=B
        p = np.zeros(K_SOURCES)
        # 4 aligned strong sources
        for i in range(4):
            p_raw = rng.uniform(0.75, 0.95)
            p[i] = p_raw if label == 0 else (1.0 - p_raw)
        # 1 weak source near 0.5
        p[4] = rng.uniform(0.40, 0.60)
        rho = np.abs(2 * p - 1)
        instances.append({"p": p, "rho": rho, "regime": "clear",
                          "label": label, "should_abstain": False})

    # ── INSUFFICIENT regime ──
    for _ in range(n_each):
        p = rng.uniform(0.35, 0.65, size=K_SOURCES)
        rho = np.abs(2 * p - 1)
        instances.append({"p": p, "rho": rho, "regime": "insuff",
                          "label": -1, "should_abstain": True})

    # ── CONFLICTING regime (both symmetric and asymmetric) ──
    for _ in range(n_total - 2 * n_each):
        n_a = int(rng.integers(2, K_SOURCES))   # 2 or 3
        n_b = K_SOURCES - n_a                    # 3 or 2
        p = np.zeros(K_SOURCES)
        for i in range(n_a):
            p[i] = rng.uniform(0.75, 0.95)       # strong toward A
        for i in range(n_a, K_SOURCES):
            p[i] = rng.uniform(0.05, 0.25)       # strong toward B
        rho = np.abs(2 * p - 1)
        instances.append({"p": p, "rho": rho, "regime": "conflict",
                          "label": -1, "should_abstain": True})

    rng.shuffle(instances)
    return instances


def instances_to_jsonl(instances, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for inst in instances:
            d = {
                "p": inst["p"].tolist(),
                "rho": inst["rho"].tolist(),
                "regime": inst["regime"],
                "label": int(inst["label"]),
                "should_abstain": inst["should_abstain"],
            }
            f.write(json.dumps(d) + "\n")


def load_jsonl(path):
    instances = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["p"]   = np.array(d["p"])
            d["rho"] = np.array(d["rho"])
            instances.append(d)
    return instances


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def config_hash():
    """Hash the GENERATOR CONFIG (not specific random instances).
    Seeds provide sampling variance; the config is frozen."""
    h = hashlib.sha256(
        json.dumps(GENERATOR_CONFIG, sort_keys=True).encode()
    ).hexdigest()[:16]
    return h


def freeze_hash(h):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FROZEN_FILE, "w") as f:
        json.dump({"frozen_hash": h, "config": GENERATOR_CONFIG}, f, indent=2)
    print(f"[freeze] config hash locked: {h}")


def assert_frozen():
    if not os.path.exists(FROZEN_FILE):
        raise RuntimeError(f"FROZEN HASH MISSING: {FROZEN_FILE}")
    frozen = json.load(open(FROZEN_FILE))["frozen_hash"]
    current = config_hash()
    if current != frozen:
        raise RuntimeError(
            f"config hash mismatch: frozen={frozen} current={current} — "
            "generator config changed since freeze")
    return current


def cmd_generate():
    """U1: freeze generator config hash. Data generated per-seed in cmd_sweep."""
    os.makedirs(DATA_DIR, exist_ok=True)
    h = config_hash()
    freeze_hash(h)
    print(f"Generator config frozen. Each sweep seed generates its own {N_INSTANCES} instances.")
    print(f"Config: K={K_SOURCES} sources, regimes=CLEAR/INSUFF/CONFLICT (1/3 each)")
    return h


# ── U2: Gate implementations ────────────────────────────────────────────────

def theta(p):
    """HPIC angle: arccos(2p-1), p in [0,1] → θ in [0°, 180°]."""
    return np.arccos(np.clip(2 * p - 1, -1.0, 1.0))


def gate_certainty(inst, gate):
    """
    Compute certainty score for one instance under the given gate.
    Higher certainty = gate is more confident → should commit.
    Abstain when certainty < τ.
    """
    p   = inst["p"]   # (K,)
    rho = inst["rho"] # (K,)

    if gate == "linear":
        # score = Σρ(2p-1); certainty = |score|
        score = np.sum(rho * (2 * p - 1))
        return abs(score)

    elif gate == "max_prob":
        # Weighted log-odds combination (naive Bayes)
        log_odds = np.sum(rho * np.log(np.clip(p, 1e-9, 1-1e-9) /
                                        np.clip(1 - p, 1e-9, 1-1e-9)))
        p_a = 1.0 / (1.0 + math.exp(-log_odds))
        return max(p_a, 1.0 - p_a)

    elif gate == "hpic_point":
        # Z = Σρ·e^{iθ}; certainty = |Re Z| / (|Z| + ε)
        # This is cos(Arg Z) — normalized reliability ratio
        th = theta(p)
        re_z = np.sum(rho * np.cos(th))
        im_z = np.sum(rho * np.sin(th))
        mag  = math.sqrt(re_z**2 + im_z**2) + 1e-9
        return abs(re_z) / mag

    elif gate == "hpic_interval":
        # Angular dispersion criterion.
        # mean_θ = mean of source angles; std_θ = angular spread.
        # certainty = |mean_θ - π/2| / (std_θ + ε)
        # Abstain when interval [mean_θ - std_θ, mean_θ + std_θ] straddles π/2.
        # (ρ-weighted version: use ρ as weights)
        th    = theta(p)
        w     = rho + 1e-9
        w_sum = w.sum()
        mean_th = np.sum(w * th) / w_sum
        var_th  = np.sum(w * (th - mean_th)**2) / w_sum
        std_th  = math.sqrt(var_th)
        return abs(mean_th - math.pi / 2) / (std_th + 0.1)

    else:
        raise ValueError(f"Unknown gate: {gate}")


# ── U3: Selective-prediction evaluation harness ─────────────────────────────

def compute_gate_scores(instances, gate):
    """Return array of certainty scores for all instances."""
    return np.array([gate_certainty(inst, gate) for inst in instances])


def risk_coverage_curve(certainties, instances):
    """
    Sweep τ from high to low (more permissive → higher coverage).

    Risk = P(error | committed):
      - CLEAR committed + wrong class  → error
      - CLEAR committed + correct      → not error
      - INSUFF/CONFLICT committed       → always error (should have abstained)
    Coverage = fraction committed.

    Returns (coverages, risks) arrays for AUC computation.
    """
    # Sort by certainty descending: as we lower threshold, we include more
    idx = np.argsort(certainties)[::-1]
    sorted_cert = certainties[idx]
    sorted_inst = [instances[i] for i in idx]

    # For CLEAR instances, we need to know the gate's predicted class too.
    # Use sign of linear score as pseudo-label (gate-agnostic; the label
    # evaluation is about WHETHER to abstain, not which class to pick).
    # Actually: for CLEAR instances that the gate commits on, we evaluate
    # accuracy using the correct label and the gate's implicit class prediction.
    # The "class prediction" for all gates is: p_A = sum(rho*(2p-1)) > 0 → A.
    labels  = np.array([inst["label"] for inst in instances])
    regimes = np.array([inst["regime"] for inst in instances])

    # Gate predicted class (linear score direction, same for all gates):
    def predicted_label(inst):
        score = np.sum(inst["rho"] * (2 * inst["p"] - 1))
        return 0 if score >= 0 else 1  # 0=A, 1=B

    sorted_pred = np.array([predicted_label(inst) for inst in sorted_inst])
    sorted_labels  = labels[idx]
    sorted_regimes = regimes[idx]

    N = len(instances)
    coverages, risks = [0.0], [0.0]

    n_committed = 0
    n_errors    = 0

    for i in range(N):
        inst_regime = sorted_regimes[i]
        if inst_regime == "clear":
            wrong = (sorted_pred[i] != sorted_labels[i])
            n_errors += int(wrong)
        else:  # insuff or conflict: committed = automatic error
            n_errors += 1
        n_committed += 1

        # Only record curve points at unique certainty boundaries to save memory
        # (record at each point for small N; decimation optional)
        coverage = n_committed / N
        risk     = n_errors / n_committed
        coverages.append(coverage)
        risks.append(risk)

    return np.array(coverages), np.array(risks)


def auc_trapezoidal(coverages, risks):
    """AUC of risk–coverage curve. Lower is better."""
    return float(np.trapz(risks, coverages))


def conflict_abstain_recall_at_clear_coverage(certainties, instances,
                                               target_clear_coverage):
    """
    At the threshold τ where CLEAR coverage ≈ target_clear_coverage:
    what fraction of CONFLICT instances does the gate abstain on?

    Returns (conflict_abstain_recall, actual_clear_coverage).
    """
    regimes = np.array([inst["regime"] for inst in instances])
    clear_mask    = regimes == "clear"
    conflict_mask = regimes == "conflict"

    if conflict_mask.sum() == 0 or clear_mask.sum() == 0:
        return float("nan"), float("nan")

    # Sweep τ: find where CLEAR coverage ≈ target
    thresholds = np.sort(np.unique(certainties[clear_mask]))[::-1]

    best_tau  = thresholds[0]
    best_diff = 1.0

    for tau in thresholds:
        clear_committed = (certainties[clear_mask] >= tau).mean()
        diff = abs(clear_committed - target_clear_coverage)
        if diff < best_diff:
            best_diff = diff
            best_tau  = tau

    actual_clear_cov = (certainties[clear_mask] >= best_tau).mean()
    conflict_abstain = (certainties[conflict_mask] < best_tau).mean()
    return float(conflict_abstain), float(actual_clear_cov)


def insuff_abstain_recall_at_tau(certainties, instances, tau):
    regimes = np.array([inst["regime"] for inst in instances])
    mask = regimes == "insuff"
    if mask.sum() == 0:
        return float("nan")
    return float((certainties[mask] < tau).mean())


def evaluate_gate(instances, gate):
    """Run full evaluation for one gate on one set of instances. Returns dict."""
    certainties = compute_gate_scores(instances, gate)
    coverages, risks = risk_coverage_curve(certainties, instances)
    rc_auc = auc_trapezoidal(coverages, risks)

    conflict_recall, actual_clear_cov = conflict_abstain_recall_at_clear_coverage(
        certainties, instances, CLEAR_COVERAGE_TARGET)

    # Also record at fixed τ=median for sanity
    tau_median = float(np.median(certainties))
    insuff_recall = insuff_abstain_recall_at_tau(certainties, instances, tau_median)

    return {
        "rc_auc":              rc_auc,
        "conflict_abstain_recall": conflict_recall,
        "actual_clear_coverage": actual_clear_cov,
        "insuff_abstain_recall_median_tau": insuff_recall,
        "n_instances": len(instances),
        "n_clear":    sum(1 for i in instances if i["regime"] == "clear"),
        "n_insuff":   sum(1 for i in instances if i["regime"] == "insuff"),
        "n_conflict": sum(1 for i in instances if i["regime"] == "conflict"),
    }


# ── U4: Sweep ───────────────────────────────────────────────────────────────

def cmd_sweep(seeds=None, smoke=False):
    """Run all gates × seeds. Each seed generates its own fresh instances."""
    assert_frozen()

    if smoke:
        seeds = [0]
        n_inst = N_SMOKE
        print(f"[smoke] seed 0, {n_inst} instances")
    else:
        seeds = list(range(N_SEEDS)) if seeds is None else seeds
        n_inst = N_INSTANCES

    for seed in seeds:
        res_path = RESULTS_FILE.format(seed)
        if os.path.exists(res_path) and not smoke:
            print(f"[sweep] seed {seed} already done, skipping")
            continue

        # Each seed generates its own fresh random instances → real sampling variance
        rng = np.random.default_rng(seed + 1000)   # +1000 to avoid collision with generation seed
        instances = generate_instances(rng, n_inst)

        result = {"seed": seed, "smoke": smoke, "n_instances": n_inst, "gates": {}}
        for gate in GATES:
            metrics = evaluate_gate(instances, gate)
            result["gates"][gate] = metrics
            print(f"  seed={seed} gate={gate:<15} rc_auc={metrics['rc_auc']:.4f}  "
                  f"conflict_recall={metrics['conflict_abstain_recall']:.3f}")

        if not smoke:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(res_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[sweep] seed {seed} written → {res_path}")

    if smoke:
        print("[smoke] PASS — all gates ran without error")


# ── U5: Aggregate + verdict ─────────────────────────────────────────────────

def load_all_results():
    """Load per-seed result files. Returns list of result dicts."""
    results = []
    for seed in range(N_SEEDS):
        path = RESULTS_FILE.format(seed)
        if os.path.exists(path):
            results.append(json.load(open(path)))
    return results


def cohen_d(deltas):
    n = len(deltas)
    if n < 2:
        return float("nan")
    mean = sum(deltas) / n
    var  = sum((x - mean)**2 for x in deltas) / (n - 1)
    std  = math.sqrt(var) if var > 0 else 1e-9
    return mean / std


def paired_gate_verdict(results, metric_key, gate_a, gate_b,
                         min_seeds=GATE_MIN_SEEDS, effect_floor=GATE_EFFECT_SIZE,
                         higher_is_better=True):
    """gate_a > gate_b verdict. Returns (wins, mean_delta, cohen_d, verdict_str)."""
    deltas = []
    for r in results:
        va = r["gates"][gate_a][metric_key]
        vb = r["gates"][gate_b][metric_key]
        if math.isnan(va) or math.isnan(vb):
            continue
        d = (va - vb) if higher_is_better else (vb - va)
        deltas.append(d)

    n = len(deltas)
    if n < min_seeds:
        return n, float("nan"), float("nan"), f"INSUFFICIENT ({n}/{N_SEEDS})"

    wins     = sum(1 for d in deltas if d > 0)
    mean_d   = sum(deltas) / n
    cd       = cohen_d(deltas)
    passed   = (wins >= min_seeds) and (mean_d >= effect_floor)
    verdict  = "PASS ✓" if passed else "FAIL ✗"
    return wins, mean_d, cd, verdict


def cmd_aggregate():
    results = load_all_results()
    n_done = len(results)
    print(f"Seeds loaded: {n_done}/{N_SEEDS}")
    if n_done < N_SEEDS:
        print(f"WARNING: only {n_done} seeds complete")

    # Per-gate mean±std across seeds
    gate_stats = {}
    for gate in GATES:
        rc_aucs     = [r["gates"][gate]["rc_auc"]              for r in results]
        c_recalls   = [r["gates"][gate]["conflict_abstain_recall"] for r in results]
        gate_stats[gate] = {
            "rc_auc_mean":  sum(rc_aucs)/len(rc_aucs),
            "rc_auc_std":   math.sqrt(sum((x-sum(rc_aucs)/len(rc_aucs))**2
                                          for x in rc_aucs) / max(len(rc_aucs)-1,1)),
            "conflict_recall_mean": sum(c_recalls)/len(c_recalls),
            "conflict_recall_std":  math.sqrt(sum((x-sum(c_recalls)/len(c_recalls))**2
                                                   for x in c_recalls) / max(len(c_recalls)-1,1)),
        }

    # ── Gate verdicts ──
    # g_A: any HPIC variant > {linear, max_prob} on risk–coverage AUC
    #   (lower AUC is better, so we check baseline - hpic > 0.03 → higher_is_better=False for AUC)
    hpic_variants = ["hpic_point", "hpic_interval"]
    baselines     = ["linear", "max_prob"]

    # For AUC: hpic lower is better → compute (baseline - hpic) for wins
    g_a_verdicts = {}
    for hpic in hpic_variants:
        for base in baselines:
            key = f"{hpic}_vs_{base}"
            wins, md, cd, v = paired_gate_verdict(
                results, "rc_auc", hpic, base, higher_is_better=False)
            g_a_verdicts[key] = (wins, md, cd, v)

    # g_B: HPIC > baselines on conflict abstain recall (higher is better)
    g_b_verdicts = {}
    for hpic in hpic_variants:
        for base in baselines:
            key = f"{hpic}_vs_{base}"
            wins, md, cd, v = paired_gate_verdict(
                results, "conflict_abstain_recall", hpic, base,
                higher_is_better=True)
            g_b_verdicts[key] = (wins, md, cd, v)

    # Determine overall g_A and g_B pass
    g_a_pass = any(v[3] == "PASS ✓" for v in g_a_verdicts.values())
    g_b_pass = any(v[3] == "PASS ✓" for v in g_b_verdicts.values())

    # ── Write results_hpic_gate.md ──
    lines = []
    lines.append("# WP-ST-4: HPIC ⊥-Gate A/B Test — Results\n\n")
    lines.append(f"**Seeds:** 0–{n_done-1} ({n_done}/{N_SEEDS} complete)  \n")
    lines.append(f"**N instances:** {N_INSTANCES} (test set 30%)  "
                 f"**K sources:** {K_SOURCES}  "
                 f"**ρ_k = |2p_k−1|** (pre-registered)  \n")
    lines.append(f"**Regimes:** CLEAR / INSUFF / CONFLICT (equal thirds)  \n\n")

    lines.append("---\n## Per-Gate Summary (mean±std, 10 seeds)\n\n")
    lines.append("| Gate | RC-AUC (↓ better) | Conflict-Abstain-Recall (↑ better) |\n")
    lines.append("|---|---|---|\n")
    for gate in GATES:
        s = gate_stats[gate]
        lines.append(f"| {gate} | {s['rc_auc_mean']:.4f}±{s['rc_auc_std']:.4f} | "
                     f"{s['conflict_recall_mean']:.3f}±{s['conflict_recall_std']:.3f} |\n")
    lines.append("\n")

    lines.append("---\n## Gate g_A: HPIC > baselines on Risk–Coverage AUC (lower = better for AUC)\n\n")
    lines.append("| Comparison | Seeds winning | Mean Δ (baseline−HPIC) | Cohen's d | Verdict |\n")
    lines.append("|---|---|---|---|---|\n")
    for key, (wins, md, cd, v) in g_a_verdicts.items():
        cd_str = f"{cd:.2f}" if not math.isnan(cd) else "nan"
        md_str = f"{md:+.4f}" if not math.isnan(md) else "nan"
        lines.append(f"| {key} | {wins}/{N_SEEDS} | {md_str} | {cd_str} | {v} |\n")
    lines.append(f"\n**g_A overall: {'PASS ✓' if g_a_pass else 'FAIL ✗'}**\n\n")

    lines.append("---\n## Gate g_B (DECISIVE): HPIC > baselines on Conflict-Regime Abstain Recall\n\n")
    lines.append(f"Operating point: CLEAR coverage ≈ {CLEAR_COVERAGE_TARGET:.0%}  \n\n")
    lines.append("| Comparison | Seeds winning | Mean Δ | Cohen's d | Verdict |\n")
    lines.append("|---|---|---|---|---|\n")
    for key, (wins, md, cd, v) in g_b_verdicts.items():
        cd_str = f"{cd:.2f}" if not math.isnan(cd) else "nan"
        md_str = f"{md:+.4f}" if not math.isnan(md) else "nan"
        lines.append(f"| {key} | {wins}/{N_SEEDS} | {md_str} | {cd_str} | {v} |\n")
    lines.append(f"\n**g_B overall: {'PASS ✓' if g_b_pass else 'FAIL ✗'}**\n\n")

    lines.append("---\n## Verdict\n\n")
    if g_b_pass:
        lines.append("**g_B PASS** — HPIC complex geometry outperforms plain thresholds on "
                     "conflict-regime abstain recall. HPIC ⊥-gate adopted for CBT's "
                     "accept/reject/⊥ decision point.\n\n")
    else:
        lines.append("**g_B FAIL** — No HPIC variant outperforms plain thresholds on "
                     "conflict-regime abstain recall by meaningful margin. "
                     "HPIC ⊥-gate REJECTED; CBT uses plain confidence threshold.\n\n")

    if g_a_pass:
        lines.append("**g_A PASS** — HPIC also improves overall risk–coverage AUC.\n\n")
    else:
        lines.append("**g_A FAIL** — HPIC does not improve overall risk–coverage AUC.\n\n")

    lines.append("*See papers/claim_hpic_gate.md for bounded claim + adoption decision.*\n")

    out_path = os.path.join(PAPERS_DIR, "results_hpic_gate.md")
    os.makedirs(PAPERS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"Written: {out_path}")

    # Print summary to stdout
    print("\n=== AGGREGATE VERDICT ===")
    for gate in GATES:
        s = gate_stats[gate]
        print(f"  {gate:<15} rc_auc={s['rc_auc_mean']:.4f}  "
              f"conflict_recall={s['conflict_recall_mean']:.3f}")
    print(f"\ng_A (RC-AUC): {'PASS' if g_a_pass else 'FAIL'}")
    print(f"g_B (conflict recall): {'PASS' if g_b_pass else 'FAIL'}")

    return g_a_pass, g_b_pass, gate_stats, g_a_verdicts, g_b_verdicts


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate",  action="store_true")
    ap.add_argument("--smoke",     action="store_true")
    ap.add_argument("--sweep",     action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--all",       action="store_true")
    args = ap.parse_args()

    if args.all:
        args.generate = args.sweep = args.aggregate = True

    if args.generate or not os.path.exists(os.path.join(DATA_DIR, "train.jsonl")):
        cmd_generate()

    if args.smoke:
        cmd_sweep(smoke=True)

    if args.sweep:
        cmd_sweep(smoke=False)

    if args.aggregate:
        cmd_aggregate()


if __name__ == "__main__":
    main()
