#!/usr/bin/env python3
"""WP-ST-18 U3–U7: features → correlation → 3-way bake-off → source-detector → verdict.

CRITICAL: features are built from framing.prompt (QUERY), NOT the shared
passage. Passage-disjoint train/test split (all 3 framings of a passage on
the same side). No test tuning; k-fold w/ StratifiedGroupKFold.

Subcommands:
  features            build TF-IDF + per-level centroids on TRAIN passages
  corr                per-level corr(density_L, distance_L)
  bakeoff             3-way + baselines with passage-disjoint k-fold
  source              source-detector on same features (Gate B ≤ 0.55)
  neg_control         shuffle level labels; every scorer must collapse to chance
  verdict             mechanical A/B/C emission gated by all guards

Reads: data/ece_shared_pool_v2/items.jsonl
Writes: data/ece_shared_pool_v2/{features.jsonl, corr.json, bakeoff.json,
        source.json, neg_control.json, verdict.json}
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = ROOT / "data" / "ece_shared_pool_v2"

LEVELS = ("task", "concept", "context")
LVL2IDX = {l: i for i, l in enumerate(LEVELS)}

# ── I/O ───────────────────────────────────────────────────────────────────

def load_records() -> list[dict]:
    with open(POOL_DIR / "items.jsonl") as f:
        return [json.loads(line) for line in f]


def deterministic_passage_split(passage_ids: list[str], seed: int = 42,
                                test_frac: float = 0.2):
    """Passage-disjoint split. All 3 framings of a passage go to the same side."""
    uniq = sorted(set(passage_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_test = int(round(len(uniq) * test_frac))
    test_p = set(uniq[i] for i in perm[:n_test])
    train_p = set(uniq[i] for i in perm[n_test:])
    return train_p, test_p


# ── FEATURES (from QUERY, not passage) ────────────────────────────────────

def cmd_features(args):
    """Build features from the U2-STRIPPED query text (not raw).
    The stripped view has 23 leaked-marker tokens masked as __STRIP__.
    Passage is NEVER passed to the vectorizer."""
    stripped_path = POOL_DIR / "stripped.jsonl"
    if not stripped_path.exists():
        raise SystemExit("U2 stripped.jsonl missing — run pool_v2_audit.py first")
    with open(stripped_path) as f:
        records = [json.loads(line) for line in f]
    print(f"[U3] loaded {len(records)} STRIPPED records from U2")

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Passage-disjoint split (fit on TRAIN passages only)
    train_p, test_p = deterministic_passage_split(
        [r["passage_id"] for r in records], seed=args.seed, test_frac=args.test_frac)
    print(f"[U3] passage-disjoint split: {len(train_p)} train / {len(test_p)} test")

    def prompt_text(r):
        # STRIPPED query text — passage NEVER included by design.
        # Strip __STRIP__ marker tokens (removed leaked-level markers per U2).
        s = r.get("stripped_prompt", r["framing"]["prompt"])
        return s.replace("__strip__", "").replace("__STRIP__", "")

    train_texts = [prompt_text(r) for r in records if r["passage_id"] in train_p]
    train_labels = [LVL2IDX[r["level"]] for r in records if r["passage_id"] in train_p]

    # Fit TF-IDF on TRAIN queries only
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                          min_df=2, max_df=0.95, sublinear_tf=True)
    X_tr = vec.fit_transform(train_texts)
    print(f"[U3] TF-IDF vocab size = {len(vec.vocabulary_)}")

    # Per-level centroids in TF-IDF space (mean of train vectors labeled L)
    centroids = np.zeros((len(LEVELS), X_tr.shape[1]))
    for l_idx in range(len(LEVELS)):
        mask = np.array([lab == l_idx for lab in train_labels])
        if mask.sum() > 0:
            centroids[l_idx] = np.asarray(X_tr[mask].mean(axis=0)).flatten()
    # L2-normalize centroids for cosine
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1
    centroids_n = centroids / norms

    # ASSERT passage-token contribution ≈ 0 by construction:
    # we NEVER passed the passage to the vectorizer. Sanity-check: none of the
    # vocab terms should exclusively come from passages. (Vocab is fit on
    # train_texts = queries.) We record vocab overlap with passage tokens
    # as an audit statistic.
    passage_toks = set()
    for r in records:
        if r["passage_id"] in train_p:
            for m in re.finditer(r"[a-z0-9']+", r["passage"].lower()):
                passage_toks.add(m.group())
    vocab_terms = set(vec.vocabulary_.keys())
    passage_overlap = sum(1 for t in vocab_terms
                          if any(pt in t for pt in passage_toks))
    print(f"[U3] vocab-passage-overlap sanity: {passage_overlap}/{len(vocab_terms)} "
          f"(informational only — features come from QUERIES, not passages)")

    # Featurize ALL records with TRAIN-fit vectorizer + TRAIN-fit centroids
    all_texts = [prompt_text(r) for r in records]
    X_all = vec.transform(all_texts)
    X_all_dense = np.asarray(X_all.todense())

    # DISTANCE (geometry) = 1 - cosine similarity to class-L centroid
    xnorm = np.linalg.norm(X_all_dense, axis=1, keepdims=True)
    xnorm[xnorm == 0] = 1
    Xn = X_all_dense / xnorm
    cos_sim = Xn @ centroids_n.T  # (N, 3)
    dist = 1.0 - cos_sim  # (N, 3), one distance per level

    # DENSITY (mass) = class-conditional Naive-Bayes-style log-likelihood
    # under a Laplace-smoothed unigram language model per level, using the
    # SAME term vocabulary as the TF-IDF vectorizer but with RAW-COUNT
    # class-conditional probabilities. Fit on TRAIN queries only.
    #
    # Log-likelihood of x under level L:
    #   log P(x | L) = sum_t log P(t | L)
    # where P(t | L) = (count(t, L) + 1) / (sum_t' count(t', L) + V)
    #
    # This is a MASS-based feature (how likely x's tokens are under class L's
    # unigram model) — orthogonal to the GEOMETRY-based cosine distance.
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 2),
                         min_df=2, max_df=0.95,
                         vocabulary=vec.vocabulary_)
    C_tr = cv.fit_transform(train_texts)  # (N_tr, V) raw counts on TRAIN
    V = C_tr.shape[1]
    # per-level total counts
    log_p_tL = np.zeros((len(LEVELS), V))
    for l_idx in range(len(LEVELS)):
        mask = np.array([lab == l_idx for lab in train_labels])
        if mask.sum() == 0:
            log_p_tL[l_idx] = np.log(1.0 / V)
            continue
        cnt = np.asarray(C_tr[mask].sum(axis=0)).flatten() + 1.0  # Laplace
        p = cnt / cnt.sum()
        log_p_tL[l_idx] = np.log(p)

    # Featurize all records
    C_all = cv.transform(all_texts)   # (N, V) raw counts
    # log P(x | L) = sum over tokens t: count(t, x) * log P(t | L)
    dens = np.zeros((len(records), len(LEVELS)))
    C_all_arr = np.asarray(C_all.todense())
    for l_idx in range(len(LEVELS)):
        dens[:, l_idx] = C_all_arr @ log_p_tL[l_idx]  # (N,)
    # normalize by token-count to avoid length dominating (average log-lik per token)
    tok_counts = C_all_arr.sum(axis=1, keepdims=True)
    tok_counts[tok_counts == 0] = 1
    dens = dens / tok_counts  # (N, 3), average log-likelihood per token

    feats_path = POOL_DIR / "features.jsonl"
    with open(feats_path, "w") as f:
        for i, r in enumerate(records):
            feat = {
                "passage_id": r["passage_id"],
                "source_label": r["source_label"],
                "level": r["level"],
                "in_train": r["passage_id"] in train_p,
                "density":  {LEVELS[k]: float(dens[i, k]) for k in range(3)},
                "distance": {LEVELS[k]: float(dist[i, k]) for k in range(3)},
            }
            f.write(json.dumps(feat) + "\n")

    # Freeze-hash the features (query-based + passage-disjoint)
    h = hashlib.sha256()
    with open(feats_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    frozen = h.hexdigest()[:16]

    manifest = {
        "wp": "WP-ST-18", "unit": "U3",
        "vectorizer": "TfidfVectorizer(lowercase, ngrams 1-2, min_df=2, "
                      "max_df=0.95, sublinear_tf, TRAIN-passages-only fit)",
        "centroid_space": "per-level TF-IDF centroid, L2-normalized for cosine",
        "features_per_level": {"density": "avg log P(token | level=L) under class-L Laplace-smoothed unigram LM (MASS-based)",
                               "distance": "1 - cosine(x TF-IDF, centroid_L) (GEOMETRY-based)"},
        "passage_disjoint": True,
        "seed": args.seed, "test_frac": args.test_frac,
        "n_train_passages": len(train_p), "n_test_passages": len(test_p),
        "vocab_size": len(vec.vocabulary_),
        "vocab_passage_overlap_terms": passage_overlap,
        "vocab_passage_overlap_frac": passage_overlap / max(len(vocab_terms), 1),
        "features_hash": frozen,
    }
    (POOL_DIR / "features_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[U3] wrote {len(records)} feature vectors")
    print(f"[U3] features_hash = {frozen}")

    # cache centroids + vectorizer vocab for downstream reproducibility
    np.savez(POOL_DIR / "features_centroids.npz",
             centroids=centroids, centroids_norm=centroids_n)
    (POOL_DIR / "features_vocab.json").write_text(
        json.dumps({"vocab_size": len(vec.vocabulary_),
                    "sample_terms": list(vec.vocabulary_.keys())[:50]}))
    return 0


def load_features(name: str = "features.jsonl") -> list[dict]:
    with open(POOL_DIR / name) as f:
        return [json.loads(line) for line in f]


def cmd_residualize(args):
    """Retry #2 on U3: residualize source signal out of the 6-dim features.

    Fit LogReg source classifier on TRAIN features; get 4-class source proba
    on ALL records; regress each of the 6 level features on the source proba
    (linear fit on TRAIN, apply to all); write residuals to features_r.jsonl.
    Preserves passage-disjoint split; frozen hash written."""
    from sklearn.linear_model import LogisticRegression, LinearRegression

    feats = load_features("features.jsonl")
    N = len(feats)
    print(f"[U3-r] loaded {N} feature records for residualization")

    src_labels = sorted(set(f["source_label"] for f in feats))
    s2i = {s: i for i, s in enumerate(src_labels)}
    y_src = np.array([s2i[f["source_label"]] for f in feats])

    X = feats_to_matrix(feats, "ps")  # (N, 6) — 3 density + 3 distance
    train_mask = np.array([f["in_train"] for f in feats])

    # 1) fit source classifier on TRAIN
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X[train_mask], y_src[train_mask])
    S = clf.predict_proba(X)   # (N, 4)

    # 2) residualize each of the 6 features by regressing on S (TRAIN-fit)
    X_res = np.zeros_like(X)
    per_feat_r2 = []
    for j in range(X.shape[1]):
        reg = LinearRegression()
        reg.fit(S[train_mask], X[train_mask, j])
        pred = reg.predict(S)
        X_res[:, j] = X[:, j] - pred
        per_feat_r2.append(float(reg.score(S[train_mask], X[train_mask, j])))
    print(f"[U3-r] per-feature R² of source→feature regression: "
          f"{[round(r, 3) for r in per_feat_r2]}")

    # 3) write residualized features.jsonl (same schema)
    out_path = POOL_DIR / "features_r.jsonl"
    with open(out_path, "w") as f:
        for i, r in enumerate(feats):
            out = dict(r)
            # 6-dim vector layout: [dens_task, dens_concept, dens_context,
            #                        dist_task, dist_concept, dist_context]
            out["density"]  = {LEVELS[k]: float(X_res[i, k])     for k in range(3)}
            out["distance"] = {LEVELS[k]: float(X_res[i, k + 3]) for k in range(3)}
            f.write(json.dumps(out) + "\n")

    # 4) freeze-hash + manifest update
    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    frozen = h.hexdigest()[:16]

    old_man = json.loads((POOL_DIR / "features_manifest.json").read_text())
    man = dict(old_man)
    man["retry"] = 2
    man["retry_kind"] = "source-residualization"
    man["source_residualize"] = {
        "sources": src_labels,
        "per_feature_r2_train": per_feat_r2,
        "note": ("Each of 6 features residualized by LinReg on 4-class source proba "
                 "produced by LogReg fit on TRAIN features. Residual = f - E[f | source_proba]."),
    }
    man["features_hash_residualized"] = frozen
    (POOL_DIR / "features_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"[U3-r] wrote features_r.jsonl; residualized_hash = {frozen}")
    return 0


def feats_to_matrix(feats: list[dict], mode: str) -> np.ndarray:
    """mode: density_only | distance_only | ps | 2feat"""
    N = len(feats)
    if mode == "density_only":
        return np.array([[f["density"][l] for l in LEVELS] for f in feats])
    if mode == "distance_only":
        return np.array([[f["distance"][l] for l in LEVELS] for f in feats])
    if mode == "ps":
        return np.array([[f["density"][l] for l in LEVELS] +
                         [f["distance"][l] for l in LEVELS] for f in feats])
    if mode == "2feat":
        # class-agnostic aggregates: mean density, mean distance
        return np.array([[np.mean([f["density"][l] for l in LEVELS]),
                          np.mean([f["distance"][l] for l in LEVELS])]
                         for f in feats])
    raise ValueError(mode)


# ── U4: correlation ───────────────────────────────────────────────────────

def cmd_corr(args):
    feats = load_features(args.features_file)
    out = {}
    for l in LEVELS:
        d = np.array([f["density"][l] for f in feats])
        r = np.array([f["distance"][l] for f in feats])
        if d.std() == 0 or r.std() == 0:
            out[l] = {"corr": None, "reason": "constant feature"}
        else:
            out[l] = {"corr": float(np.corrcoef(d, r)[0, 1])}
    (POOL_DIR / "corr.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # gate: all finite
    all_finite = all(v.get("corr") is not None and np.isfinite(v["corr"])
                     for v in out.values())
    print(f"[U4] all_finite = {all_finite}")
    return 0 if all_finite else 1


# ── U5: 3-way + baselines ─────────────────────────────────────────────────

def stratified_group_kfold(passage_ids: list[str], labels: list[int], k: int, seed: int):
    """Passage-groups → k folds. Each fold's test set is disjoint by passage.
    We use sklearn.model_selection.StratifiedGroupKFold when levels are stratified."""
    from sklearn.model_selection import GroupKFold
    # Since each passage has all 3 levels equally, stratification by label
    # is auto-satisfied when we group by passage. Use plain GroupKFold.
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(np.zeros(len(passage_ids)), labels, groups=passage_ids):
        yield tr, te


def true_discrimination(y_true, y_pred):
    """Row-level TRUE discrimination: per row, correct class predicted."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    per_class = {}
    for l_idx, l in enumerate(LEVELS):
        mask = y_true == l_idx
        if mask.sum() > 0:
            per_class[l] = {"recall": float((y_pred[mask] == l_idx).mean()),
                            "n": int(mask.sum())}
        else:
            per_class[l] = {"recall": None, "n": 0}
    macro = float(np.mean([per_class[l]["recall"] for l in LEVELS
                           if per_class[l]["recall"] is not None]))
    return {"per_class_recall": per_class, "macro_recall": macro,
            "accuracy": float((y_true == y_pred).mean())}


def macro_auc(y_true, proba):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        return None


def _fit_predict_one_scorer(X_tr, y_tr, X_te, mode):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_te)
    y_hat_tr = clf.predict(X_tr)
    try:
        proba_te = clf.predict_proba(X_te)
        proba_tr = clf.predict_proba(X_tr)
    except Exception:
        proba_te = None; proba_tr = None
    return y_hat, y_hat_tr, proba_te, proba_tr


def cmd_bakeoff(args):
    feats = load_features(args.features_file)
    labels = [LVL2IDX[f["level"]] for f in feats]
    passage_ids = [f["passage_id"] for f in feats]
    print(f"[U5] N={len(feats)} feats; k={args.k}")

    scorers = ["density_only", "distance_only", "ps", "2feat"]
    results = {}

    for mode in scorers:
        X = feats_to_matrix(feats, mode)
        fold_train_recall, fold_test_recall = [], []
        fold_train_auc, fold_test_auc = [], []
        fold_train_acc, fold_test_acc = [], []

        for fi, (tr, te) in enumerate(
                stratified_group_kfold(passage_ids, labels, args.k, args.seed)):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = np.array(labels)[tr], np.array(labels)[te]
            y_hat, y_hat_tr, proba_te, proba_tr = _fit_predict_one_scorer(
                X_tr, y_tr, X_te, mode)

            r_tr = true_discrimination(y_tr, y_hat_tr)
            r_te = true_discrimination(y_te, y_hat)
            fold_train_recall.append(r_tr["macro_recall"])
            fold_test_recall.append(r_te["macro_recall"])
            fold_train_acc.append(r_tr["accuracy"])
            fold_test_acc.append(r_te["accuracy"])
            if proba_te is not None:
                fold_train_auc.append(macro_auc(y_tr, proba_tr))
                fold_test_auc.append(macro_auc(y_te, proba_te))

        def summ(xs):
            xs = [x for x in xs if x is not None]
            if not xs:
                return {"mean": None, "std": None, "n": 0}
            return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "n": len(xs)}

        results[mode] = {
            "train_recall": summ(fold_train_recall),
            "test_recall":  summ(fold_test_recall),
            "train_auc":    summ(fold_train_auc),
            "test_auc":     summ(fold_test_auc),
            "train_acc":    summ(fold_train_acc),
            "test_acc":     summ(fold_test_acc),
            "fold_std_test_recall": float(np.std(fold_test_recall)),
        }

    # chance floor: 1/3
    results["chance_floor"] = {"macro_recall_expected": 1.0 / 3.0}

    # fold_std > 0 assertion (WP-5 lesson)
    fold_std_ok = all(r["fold_std_test_recall"] > 0.0
                      for k, r in results.items() if k != "chance_floor")

    # Gate A margin: PS beats each of {density_only, distance_only, 2feat} by ≥ 0.05
    ps_test = results["ps"]["test_recall"]["mean"]
    marg = {}
    for base in ("density_only", "distance_only", "2feat"):
        marg[f"ps_over_{base}"] = ps_test - results[base]["test_recall"]["mean"]
    ps_beats_parts = (marg["ps_over_density_only"] >= 0.05
                      and marg["ps_over_distance_only"] >= 0.05
                      and marg["ps_over_2feat"] >= 0.05)

    # reducibility R² of PS predictions from density_only + distance_only
    # (would use residual analysis; here we proxy: max(density_only, distance_only)
    # test_recall vs ps test_recall. If ps ≈ max(part), ps is reducible.)
    max_part = max(results["density_only"]["test_recall"]["mean"],
                   results["distance_only"]["test_recall"]["mean"])
    reducibility_margin = ps_test - max_part  # >= 0.05 → non-reducible

    summary = {
        "k_folds": args.k,
        "seed": args.seed,
        "results": results,
        "fold_std_gt_zero_all": fold_std_ok,
        "gate_A_margin_check": marg,
        "ps_beats_parts_by_0.05": bool(ps_beats_parts),
        "reducibility_margin_ps_minus_best_part": float(reducibility_margin),
        "ps_non_reducible_by_0.05": reducibility_margin >= 0.05,
    }
    (POOL_DIR / "bakeoff.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if fold_std_ok else 1


# ── U6: source-detector-at-chance ─────────────────────────────────────────

def cmd_source(args):
    feats = load_features(args.features_file)
    passage_ids = [f["passage_id"] for f in feats]
    src_labels = sorted(set(f["source_label"] for f in feats))
    s2i = {s: i for i, s in enumerate(src_labels)}
    y = np.array([s2i[f["source_label"]] for f in feats])

    X = feats_to_matrix(feats, "ps")  # same features as the main experiment
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    fold_auc = []
    for fi, (tr, te) in enumerate(
            stratified_group_kfold(passage_ids, y.tolist(), args.k, args.seed)):
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        try:
            auc = float(roc_auc_score(y[te], proba, multi_class="ovr",
                                      average="macro"))
        except Exception:
            auc = None
        fold_auc.append(auc)
    fold_auc = [a for a in fold_auc if a is not None]
    macro = float(np.mean(fold_auc)) if fold_auc else None
    std = float(np.std(fold_auc)) if fold_auc else None
    gate_B_pass = (macro is not None) and (macro <= 0.55)
    out = {"sources": src_labels, "k_folds": args.k,
           "fold_aucs": fold_auc, "macro_auc": macro, "std": std,
           "gate_B_threshold": 0.55, "gate_B_pass": gate_B_pass}
    (POOL_DIR / "source.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if gate_B_pass else 2  # 2 = auto-HALT signal


# ── U7 neg-control: shuffle labels, expect chance ─────────────────────────

def cmd_neg_control(args):
    feats = load_features(args.features_file)
    passage_ids = [f["passage_id"] for f in feats]
    labels = [LVL2IDX[f["level"]] for f in feats]
    # SHUFFLE labels by passage-group (preserve group-per-passage grouping to
    # avoid a trivial pass): pick a random level per PASSAGE, assigned to all
    # 3 framings of that passage. This mirrors the group structure.
    rng = np.random.default_rng(args.seed + 1)
    passages_uniq = sorted(set(passage_ids))
    shuffled_map = {p: int(rng.integers(0, 3)) for p in passages_uniq}
    y_shuf = np.array([shuffled_map[p] for p in passage_ids])

    scorers = ["density_only", "distance_only", "ps", "2feat"]
    results = {}
    from sklearn.linear_model import LogisticRegression
    for mode in scorers:
        X = feats_to_matrix(feats, mode)
        recalls = []
        for fi, (tr, te) in enumerate(
                stratified_group_kfold(passage_ids, y_shuf.tolist(), args.k, args.seed)):
            try:
                clf = LogisticRegression(max_iter=2000, solver="lbfgs")
                clf.fit(X[tr], y_shuf[tr])
                y_hat = clf.predict(X[te])
                r = true_discrimination(y_shuf[te], y_hat)
                recalls.append(r["macro_recall"])
            except Exception as e:
                recalls.append(None)
        recalls = [r for r in recalls if r is not None]
        results[mode] = {"macro_recall_mean": float(np.mean(recalls)) if recalls else None,
                         "macro_recall_std":  float(np.std(recalls)) if recalls else None,
                         "n_folds": len(recalls)}
    # chance = 1/3. LEAK detection: any scorer with mean_recall > 0.45 (chance + 0.12)
    LEAK_CEIL = 0.45
    leaks = {k: v for k, v in results.items()
             if v["macro_recall_mean"] is not None and v["macro_recall_mean"] > LEAK_CEIL}
    out = {"shuffled_by_passage_group": True,
           "chance_expected": 1.0 / 3.0,
           "leak_ceiling": LEAK_CEIL,
           "scorer_results": results,
           "leaks": leaks,
           "leak_free": len(leaks) == 0}
    (POOL_DIR / "neg_control.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["leak_free"] else 2


# ── U7 verdict ────────────────────────────────────────────────────────────

def cmd_verdict(args):
    u1_gate  = json.loads((POOL_DIR / "manifest.json").read_text())["gate"]
    u2_gate  = json.loads((POOL_DIR / "audit_gate.json").read_text())
    u3_man   = json.loads((POOL_DIR / "features_manifest.json").read_text())
    u5       = json.loads((POOL_DIR / "bakeoff.json").read_text())
    u6       = json.loads((POOL_DIR / "source.json").read_text())
    neg      = json.loads((POOL_DIR / "neg_control.json").read_text())

    guards = {
        "guard_neg_control_leak_free": bool(neg["leak_free"]),
        "guard_D1_pass": bool(u1_gate["d1"]["all_pass"]),
        "guard_D2_pass": bool(u2_gate["D2_pass"]),
        "guard_D2_strip_non_degenerate": bool(u2_gate["strip_non_degenerate"]),
        "guard_fold_std_gt_zero": bool(u5["fold_std_gt_zero_all"]),
        "guard_passage_disjoint": bool(u3_man["passage_disjoint"]),
        "guard_features_from_query_not_passage": True,  # by-construction
    }
    all_pass = all(guards.values())

    if not all_pass:
        verdict = {
            "verdict": "AUTO_HALT",
            "reason": "one or more guards failed",
            "guards": guards,
            "note": "Do NOT emit A/B/C. Ping CW/David."
        }
        (POOL_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))
        print(json.dumps(verdict, indent=2))
        return 2

    # Guards all pass → emit A/B/C
    ps_test    = u5["results"]["ps"]["test_recall"]["mean"]
    dens_test  = u5["results"]["density_only"]["test_recall"]["mean"]
    dist_test  = u5["results"]["distance_only"]["test_recall"]["mean"]
    two_test   = u5["results"]["2feat"]["test_recall"]["mean"]
    chance     = 1.0 / 3.0

    # "Beat chance" — pre-reg: mean test_recall ≥ chance + 0.05 (i.e., ≥ 0.383)
    BEAT_CHANCE_MARGIN = 0.05
    beat_chance = {
        "density_only": dens_test >= chance + BEAT_CHANCE_MARGIN,
        "distance_only": dist_test >= chance + BEAT_CHANCE_MARGIN,
        "ps":            ps_test  >= chance + BEAT_CHANCE_MARGIN,
        "2feat":         two_test >= chance + BEAT_CHANCE_MARGIN,
    }
    any_beat_chance = any(beat_chance.values())
    ps_beats_parts = u5["ps_beats_parts_by_0.05"]
    non_reducible = u5["ps_non_reducible_by_0.05"]
    gate_B_pass = u6["gate_B_pass"]

    if any_beat_chance and ps_beats_parts and non_reducible and gate_B_pass:
        outcome = "A"
        rationale = "PS beats parts by ≥ 0.05 AND non-reducible AND Gate B pass"
    elif any_beat_chance:
        outcome = "B"
        rationale = "some scorer beats chance but PS geometry cosmetic (parts explain PS)"
    else:
        outcome = "C"
        rationale = "no scorer beats chance on held-out"

    verdict = {
        "verdict": outcome,
        "rationale": rationale,
        "guards": guards,
        "scorers_test_recall": {"density_only": dens_test, "distance_only": dist_test,
                                "ps": ps_test, "2feat": two_test, "chance": chance},
        "beat_chance_margin_0.05": beat_chance,
        "ps_beats_parts_by_0.05": ps_beats_parts,
        "ps_non_reducible_by_0.05": non_reducible,
        "gate_B_source_auc": u6["macro_auc"], "gate_B_pass": gate_B_pass,
    }
    (POOL_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("features")
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--test-frac", type=float, default=0.2)

    c = sub.add_parser("corr")
    c.add_argument("--features-file", default="features.jsonl")

    b = sub.add_parser("bakeoff")
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--features-file", default="features.jsonl")

    s = sub.add_parser("source")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--features-file", default="features.jsonl")

    n = sub.add_parser("neg_control")
    n.add_argument("--k", type=int, default=5)
    n.add_argument("--seed", type=int, default=42)
    n.add_argument("--features-file", default="features.jsonl")

    r = sub.add_parser("residualize")

    v = sub.add_parser("verdict")

    args = ap.parse_args()
    fn = {"features": cmd_features, "corr": cmd_corr, "bakeoff": cmd_bakeoff,
          "source": cmd_source, "neg_control": cmd_neg_control, "verdict": cmd_verdict,
          "residualize": cmd_residualize}
    sys.exit(fn[args.cmd](args))


if __name__ == "__main__":
    main()
