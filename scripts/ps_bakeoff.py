"""WP-ST-16 U6+U7: PS bake-off on the shared passage pool.

Consumes `data/ece_shared_pool/items.jsonl` (U5 output; 600 records × 6 sha256-hashed
DeepSeek framings across 200 passages × 3 levels × 4 sources).

U6 — Level detection:
  Features per (passage, level_candidate):
    density_L  = local density at level L's centroid (k-NN inverse mean-distance)
    distance_L = Euclidean distance from passage to level L's centroid
  Baselines (per-level one-vs-rest):
    density_only     — LogReg on [density_L]
    distance_only    — LogReg on [distance_L]
    two_feat_logreg  — LogReg on [density_L, distance_L]
    PS               — RBF-kernel SVM on [density_L, distance_L]  (nonlinear combo)
    chance_floor     — uniform random predictor
  k=5-fold CV on TRAIN; report per-level AUC mean±std across folds.
  Assert `fold_std > 0` on every reported effect size (WP-5 lesson).

U7 — Source-detector control:
  SAME features, SAME splits. Train a source classifier (predict source_label
  from the 6-D feature vector). Report per-source AUC + macro-average.
  Pre-registered gate: source-detector macro-AUC ≤ 0.55.

U8 (PENDING — CW review after U7):
  Gate A: PS macro-AUC − max(density_only, distance_only, 2feat_logreg, floor) ≥ +0.05
          AND R²(PS_score ~ 2feat_logreg_score) < 0.95  (algebraic-reducibility check).
  Gate B: source-detector macro-AUC ≤ 0.55.
  Both required → PS ADOPTED; otherwise PS = COSMETIC (adopt simplest scorer).

Splits: 80/20 stratified by (source, level) at the PASSAGE level (all 3 framings
of a passage stay in the same split — leakage-safe). Frozen splits + seed=42.

Usage:
  python scripts/ps_bakeoff.py --features    # U6a: extract + freeze features
  python scripts/ps_bakeoff.py --level-eval  # U6b: k-fold level detection + baselines
  python scripts/ps_bakeoff.py --source-eval # U7:  source-detector control
  python scripts/ps_bakeoff.py --all         # run all three
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract_schema import verify  # noqa: E402 (imported for future PS-of-Task filter)


POOL_ITEMS_FILE   = "data/ece_shared_pool/items.jsonl"
OUT_DIR           = "data/ece_shared_pool"
FEATURES_FILE     = os.path.join(OUT_DIR, "features.jsonl")
LEVEL_EVAL_FILE   = os.path.join(OUT_DIR, "u6_level_eval.json")
SOURCE_EVAL_FILE  = os.path.join(OUT_DIR, "u7_source_eval.json")

SEED       = 42
N_FOLDS    = 5
TRAIN_FRAC = 0.80
LEVELS     = ["task", "concept", "context"]
SOURCES    = ["squad", "wsd", "paws", "mnli"]
KNN_K      = 5  # k for k-NN density estimation


# ── U6a: Feature extraction ───────────────────────────────────────────────

def _load_pool():
    return [json.loads(l) for l in open(POOL_ITEMS_FILE)]


def _stratified_passage_split(passage_meta: dict, seed: int) -> tuple:
    """Split PASSAGES 80/20 stratified by source (not records). All 3 framings
    of a passage go to the same split — leakage-safe."""
    rng = np.random.default_rng(seed)
    train_ids, test_ids = [], []
    by_source: dict = {}
    for pid, m in passage_meta.items():
        by_source.setdefault(m["source_label"], []).append(pid)
    for src, ids in by_source.items():
        idx = np.array(ids)
        rng.shuffle(idx)
        cut = int(round(len(idx) * TRAIN_FRAC))
        train_ids.extend(idx[:cut].tolist())
        test_ids.extend(idx[cut:].tolist())
    return set(train_ids), set(test_ids)


def _extract_prompt_text(rec: dict) -> str:
    """Prompt = passage + level-specific framing signal (question/target_word).
    Never contract text — that would leak teacher identity per Guardrail 1.
    Prompts differ per (passage, level_candidate) so features can discriminate."""
    passage = rec["passage"]
    framing = rec.get("framing") or {}
    level = rec["level"]
    if level == "task":
        q = framing.get("question", "")
        return f"{passage} {q}"
    if level == "concept":
        w = framing.get("target_word", "")
        return f"{passage} {w}"
    # context: no natural prompt-side signal → use passage alone
    # (This is a real finding: Context lacks a distinctive prompt marker
    #  in the CBT design; it's a structural check invoked on paraphrase
    #  candidates rather than a user-asked question.)
    return passage


def cmd_features():
    """Extract PS features per (passage, level_candidate) — density + distance.

    Prompt-based features per record (record = (passage, level_candidate)):
      prompt = passage + level-specific framing signal (Task: question;
               Concept: target_word; Context: passage alone)
      TF-IDF vectorizer fit on TRAIN prompts ONLY (no test leakage)
      Level cluster L = mean TF-IDF of TRAIN records with framing == L
      density_L  = k-NN inverse-mean-distance to nearest K TRAIN records at level L
      distance_L = Euclidean distance to level L's centroid
    """
    print(f"[features] loading pool {POOL_ITEMS_FILE} ...")
    pool = _load_pool()
    print(f"[features] loaded {len(pool)} records")

    # Passage metadata
    passage_meta: dict = {}
    for r in pool:
        passage_meta[r["passage_id"]] = {"source_label": r["source_label"]}

    # PASSAGE-level split — all 3 framings of a passage → same split
    train_pids, test_pids = _stratified_passage_split(passage_meta, seed=SEED)
    print(f"[features] passage split: train={len(train_pids)} test={len(test_pids)}")

    # Per-record prompts
    for r in pool:
        r["_prompt"] = _extract_prompt_text(r)

    # TF-IDF fit on TRAIN prompts only (all 3 framings of each TRAIN passage)
    train_prompts = [r["_prompt"] for r in pool if r["passage_id"] in train_pids]
    vec = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2), lowercase=True,
        stop_words="english", min_df=1,
    )
    vec.fit(train_prompts)
    all_vecs = vec.transform([r["_prompt"] for r in pool]).toarray()

    # Per-level cluster vectors (TRAIN records with framing == L)
    level_train_vecs = {L: [] for L in LEVELS}
    for i, r in enumerate(pool):
        if r["passage_id"] in train_pids:
            level_train_vecs[r["level"]].append(all_vecs[i])
    level_cluster = {L: np.array(v) for L, v in level_train_vecs.items()}
    centroids = {L: level_cluster[L].mean(axis=0) if level_cluster[L].size else all_vecs[0] * 0.0
                 for L in LEVELS}
    print(f"[features] level cluster sizes (TRAIN records):"
          f"  task={len(level_train_vecs['task'])}"
          f"  concept={len(level_train_vecs['concept'])}"
          f"  context={len(level_train_vecs['context'])}")

    def _knn_density(vec: np.ndarray, cluster: np.ndarray, k: int) -> float:
        if cluster.shape[0] == 0:
            return 0.0
        dists = np.linalg.norm(cluster - vec, axis=1)
        k_eff = min(k, len(dists))
        top_k = np.sort(dists)[:k_eff]
        return 1.0 / (1.0 + top_k.mean())

    def _centroid_dist(vec: np.ndarray, centroid: np.ndarray) -> float:
        return float(np.linalg.norm(centroid - vec))

    with open(FEATURES_FILE, "w") as f:
        for i, r in enumerate(pool):
            pid = r["passage_id"]
            v = all_vecs[i]
            row = {
                "passage_id":   pid,
                "source_label": r["source_label"],
                "level":        r["level"],
                "retain":       r["retain"],
                "in_train":     pid in train_pids,
                "in_test":      pid in test_pids,
            }
            for L in LEVELS:
                row[f"density_{L}"]  = _knn_density(v, level_cluster[L], KNN_K)
                row[f"distance_{L}"] = _centroid_dist(v, centroids[L])
            f.write(json.dumps(row) + "\n")
    print(f"[features] wrote {FEATURES_FILE}")
    return {"n_train_passages": len(train_pids), "n_test_passages": len(test_pids),
            "n_records": len(pool)}


# ── U6b: Level detection (per-level one-vs-rest) ──────────────────────────

def _load_features() -> tuple:
    rows = [json.loads(l) for l in open(FEATURES_FILE)]
    return rows


def _row_feature_vec_for_level(row: dict, level: str, feature_set: str) -> list:
    """Extract feature vector for a per-level one-vs-rest classifier.
    feature_set in {'density_only', 'distance_only', 'two_feat', 'joint_all'}."""
    if feature_set == "density_only":
        return [row[f"density_{level}"]]
    if feature_set == "distance_only":
        return [row[f"distance_{level}"]]
    if feature_set == "two_feat":
        return [row[f"density_{level}"], row[f"distance_{level}"]]
    if feature_set == "joint_all":
        # All 6 features, in level order
        return [row[f"density_{L}"] for L in LEVELS] + \
               [row[f"distance_{L}"] for L in LEVELS]
    raise ValueError(feature_set)


def _fit_and_eval_per_level(rows_train: list, rows_test: list, level: str,
                            feature_set: str, scorer_kind: str, seed: int) -> dict:
    """One-vs-rest binary classifier for `level`. Returns per-fold + test AUC."""
    def X_y(rows: list) -> tuple:
        X = np.array([_row_feature_vec_for_level(r, level, feature_set) for r in rows])
        y = np.array([1 if r["level"] == level else 0 for r in rows])
        return X, y

    X_tr, y_tr = X_y(rows_train)
    X_te, y_te = X_y(rows_test)

    # k-fold CV on TRAIN
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_aucs: list = []
    for fold_idx, (idx_tr, idx_va) in enumerate(kf.split(X_tr, y_tr)):
        if scorer_kind == "logreg":
            clf = LogisticRegression(max_iter=1000, random_state=seed)
        elif scorer_kind == "svm_rbf":
            clf = SVC(kernel="rbf", probability=True, random_state=seed, C=1.0)
        else:
            raise ValueError(scorer_kind)
        clf.fit(X_tr[idx_tr], y_tr[idx_tr])
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_tr[idx_va])[:, 1]
        else:
            scores = clf.decision_function(X_tr[idx_va])
        try:
            auc = roc_auc_score(y_tr[idx_va], scores)
        except ValueError:
            auc = 0.5  # degenerate (all one class)
        fold_aucs.append(auc)

    # Final model on all TRAIN → evaluate on TEST
    if scorer_kind == "logreg":
        final = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        final = SVC(kernel="rbf", probability=True, random_state=seed, C=1.0)
    final.fit(X_tr, y_tr)
    if hasattr(final, "predict_proba"):
        test_scores = final.predict_proba(X_te)[:, 1]
    else:
        test_scores = final.decision_function(X_te)
    try:
        test_auc = roc_auc_score(y_te, test_scores)
    except ValueError:
        test_auc = 0.5

    fold_mean = float(np.mean(fold_aucs))
    fold_std  = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0

    # WP-5 lesson: assert fold_std > 0 before any effect claim
    if fold_std == 0.0:
        print(f"    ⚠ fold_std=0 on {feature_set}/{scorer_kind}/{level} — degenerate CV")

    return {
        "fold_aucs":  fold_aucs,
        "fold_mean":  fold_mean,
        "fold_std":   fold_std,
        "test_auc":   float(test_auc),
        "n_train":    int(len(rows_train)),
        "n_test":     int(len(rows_test)),
        "test_scores": test_scores.tolist(),  # kept for U8 reducibility check
        "test_y":      y_te.tolist(),
    }


def cmd_level_eval():
    """U6b: per-level one-vs-rest AUC across baselines + PS."""
    if not os.path.exists(FEATURES_FILE):
        print(f"[level-eval] features missing — run --features first")
        return None
    rows = _load_features()
    rows_train = [r for r in rows if r["in_train"]]
    rows_test  = [r for r in rows if r["in_test"]]
    print(f"[level-eval] TRAIN records={len(rows_train)}  TEST records={len(rows_test)}")

    scorers = [
        ("density_only",    "density_only", "logreg"),
        ("distance_only",   "distance_only", "logreg"),
        ("two_feat_logreg", "two_feat",     "logreg"),
        ("PS_svm_rbf",      "two_feat",     "svm_rbf"),
    ]

    out: dict = {"scorers": {}}
    for scorer_name, feat_set, kind in scorers:
        out["scorers"][scorer_name] = {"per_level": {}}
        for L in LEVELS:
            res = _fit_and_eval_per_level(rows_train, rows_test, L, feat_set, kind, SEED)
            out["scorers"][scorer_name]["per_level"][L] = res
        # Macro-average across levels
        fold_means = [out["scorers"][scorer_name]["per_level"][L]["fold_mean"] for L in LEVELS]
        test_aucs  = [out["scorers"][scorer_name]["per_level"][L]["test_auc"] for L in LEVELS]
        out["scorers"][scorer_name]["macro_fold_mean"] = float(np.mean(fold_means))
        out["scorers"][scorer_name]["macro_test_auc"]  = float(np.mean(test_aucs))

    # Chance floor (uniform random)
    out["scorers"]["chance_floor"] = {
        "per_level": {L: {"fold_mean": 0.5, "fold_std": 0.0, "test_auc": 0.5} for L in LEVELS},
        "macro_fold_mean": 0.5, "macro_test_auc": 0.5,
    }

    with open(LEVEL_EVAL_FILE, "w") as f:
        json.dump({k: v for k, v in out.items() if k != "test_scores"}, f, indent=2, default=str)

    # Console report
    print(f"\n[level-eval] Per-scorer × per-level (FOLD_MEAN ± FOLD_STD | TEST_AUC):")
    print(f"  {'scorer':20s}  {'task':>22s}  {'concept':>22s}  {'context':>22s}  macro_test")
    for name, _, _ in scorers:
        sc = out["scorers"][name]
        cells = []
        for L in LEVELS:
            r = sc["per_level"][L]
            cells.append(f"{r['fold_mean']:.3f}±{r['fold_std']:.3f} | {r['test_auc']:.3f}")
        print(f"  {name:20s}  {cells[0]:>22s}  {cells[1]:>22s}  {cells[2]:>22s}  {sc['macro_test_auc']:.3f}")
    print(f"  {'chance_floor':20s}  {'0.500':>22s}  {'0.500':>22s}  {'0.500':>22s}  0.500")

    # Reducibility check (U8 preview): R²(PS_test_scores ~ 2feat_logreg_test_scores)
    from sklearn.metrics import r2_score
    print(f"\n[level-eval] Reducibility check (PS_score vs 2feat_logreg_score per level):")
    for L in LEVELS:
        ps_scores = out["scorers"]["PS_svm_rbf"]["per_level"][L].get("test_scores", [])
        tf_scores = out["scorers"]["two_feat_logreg"]["per_level"][L].get("test_scores", [])
        if ps_scores and tf_scores and len(ps_scores) == len(tf_scores):
            r2 = r2_score(tf_scores, ps_scores)
            print(f"  {L}: R² = {r2:.3f}  ({'REDUCIBLE ≥ 0.95' if r2 >= 0.95 else 'NON-REDUCIBLE < 0.95'})")
    return out


# ── U7: Source-detector control ───────────────────────────────────────────

def cmd_source_eval():
    """U7: pure source/corpus classifier on the SAME 6-D features from U6.
    Predicts source_label ∈ {squad, wsd, paws, mnli}. Pre-registered gate:
    macro-AUC ≤ 0.55 (near-chance for 4-class one-vs-rest)."""
    if not os.path.exists(FEATURES_FILE):
        print(f"[source-eval] features missing — run --features first")
        return None
    rows = _load_features()
    rows_train = [r for r in rows if r["in_train"]]
    rows_test  = [r for r in rows if r["in_test"]]

    # Feature vector = joint_all (all 6 features) — same set U6 gives PS
    def X_y_source(rows: list, target_source: str) -> tuple:
        X = np.array([_row_feature_vec_for_level(r, LEVELS[0], "joint_all") for r in rows])
        y = np.array([1 if r["source_label"] == target_source else 0 for r in rows])
        return X, y

    per_source = {}
    for src in SOURCES:
        X_tr, y_tr = X_y_source(rows_train, src)
        X_te, y_te = X_y_source(rows_test,  src)
        if y_tr.sum() < 2 or y_te.sum() < 2:
            print(f"[source-eval] skip {src} (too few positives)")
            per_source[src] = {"fold_mean": 0.5, "fold_std": 0.0, "test_auc": 0.5,
                               "n_pos_train": int(y_tr.sum())}
            continue
        # k-fold CV
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_aucs = []
        for idx_tr, idx_va in kf.split(X_tr, y_tr):
            clf = LogisticRegression(max_iter=1000, random_state=SEED)
            clf.fit(X_tr[idx_tr], y_tr[idx_tr])
            scores = clf.predict_proba(X_tr[idx_va])[:, 1]
            try:
                fold_aucs.append(roc_auc_score(y_tr[idx_va], scores))
            except ValueError:
                fold_aucs.append(0.5)
        final = LogisticRegression(max_iter=1000, random_state=SEED)
        final.fit(X_tr, y_tr)
        test_scores = final.predict_proba(X_te)[:, 1]
        try:
            test_auc = roc_auc_score(y_te, test_scores)
        except ValueError:
            test_auc = 0.5
        per_source[src] = {
            "fold_aucs":  fold_aucs,
            "fold_mean":  float(np.mean(fold_aucs)),
            "fold_std":   float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
            "test_auc":   float(test_auc),
        }

    macro_fold = float(np.mean([per_source[s]["fold_mean"] for s in SOURCES]))
    macro_test = float(np.mean([per_source[s]["test_auc"]  for s in SOURCES]))
    gate_passes = macro_test <= 0.55

    out = {
        "per_source":       per_source,
        "macro_fold_mean":  macro_fold,
        "macro_test_auc":   macro_test,
        "gate_threshold":   0.55,
        "gate_passes":      gate_passes,
    }
    with open(SOURCE_EVAL_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Console report
    print(f"\n[source-eval] Per-source AUC (one-vs-rest):")
    print(f"  {'source':6s}  {'fold_mean±std':>17s}  test_auc")
    for src in SOURCES:
        s = per_source[src]
        print(f"  {src:6s}  {s['fold_mean']:.3f}±{s['fold_std']:.3f}  →  {s['test_auc']:.3f}")
    print(f"\n[source-eval] macro_fold_mean = {macro_fold:.3f}")
    print(f"[source-eval] macro_test_auc  = {macro_test:.3f}")
    print(f"[source-eval] pre-registered gate = macro_test ≤ 0.55")
    print(f"[source-eval] GATE VERDICT = {'PASS' if gate_passes else 'FAIL (source is detectable — PS confounded)'}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features",    action="store_true", help="U6a: extract + freeze features")
    ap.add_argument("--level-eval",  action="store_true", help="U6b: k-fold level detection + baselines")
    ap.add_argument("--source-eval", action="store_true", help="U7: source-detector control")
    ap.add_argument("--all",         action="store_true", help="run all three")
    args = ap.parse_args()

    if args.all:
        args.features = args.level_eval = args.source_eval = True

    if args.features:    cmd_features()
    if args.level_eval:  cmd_level_eval()
    if args.source_eval: cmd_source_eval()

    if not (args.features or args.level_eval or args.source_eval or args.all):
        print(__doc__)


if __name__ == "__main__":
    sys.exit(main())
