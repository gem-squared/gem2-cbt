#!/usr/bin/env python3
"""WP-ST-13 U5 — Partition-ALIGNMENT diagnostic (no-LLM).

David's question: is Concept/Context/Task a meaningful (natural) partition axis,
or does the data's own structure cut across it (MoE lesson: learned > hand-designed)?

Operationalization: do FREELY-learned clusters of the items recover the C/C/T labels?
  - high alignment  -> C/C/T is a natural axis of the data
  - alignment ~ random-label floor -> C/C/T is not recovered by the data structure

ANTI-by-construction guard: the only C/C/T-labeled data available (WP-2 processed_v2)
is SYNTHETIC and template-generated per level (context='Base:..Test:..',
task='Source:..Query/Output:..', concept=short symbol statements). So raw alignment
trivially recovers the TEMPLATE. We therefore report BOTH:
  (1) raw           — with surface markers (upper bound = template recovery)
  (2) stripped      — level-marker keywords removed (does deeper lexical structure survive?)
  (3) random floor  — labels shuffled (NMI/ARI should -> 0)
  (4) marker-only    — supervised probe on JUST marker-presence (how much is pure template)

HONEST SCOPE: synthetic template data CANNOT settle whether C/C/T is a *natural* axis
in real text — it only shows how the synthetic generator's structure relates to C/C/T.
The LLM handling comparison (U6, CC) + ideally a natural-text replication are required.
"""
import json, re, sys, os
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.model_selection import cross_val_score

DATA = "data/processed_v2/train.jsonl"
OUT  = "data/ccts_axis/alignment_results.json"
SEED = 0
LEVELS = ["concept", "context", "task"]

# level-marker keywords to strip for the anti-template test
MARKERS = ["Base:", "Test:", "Source:", "Query:", "Output:", "Response:",
           "Base", "Test", "Source", "Query", "Output", "Response"]

def load():
    rows = [json.loads(l) for l in open(DATA)]
    texts = [r["text"] for r in rows]
    levels = [r["level"] for r in rows]
    return texts, levels

def strip_markers(t):
    s = t
    for m in MARKERS:
        s = s.replace(m, " ")
    return re.sub(r"\s+", " ", s).strip()

def cluster_align(texts, y_idx, tag, seed=SEED):
    """TF-IDF -> KMeans(k=3) -> alignment vs true labels (purity/NMI/ARI)."""
    X = TfidfVectorizer(max_features=4000, stop_words=None).fit_transform(texts)
    km = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(X)
    c = km.labels_
    # purity: best cluster->label assignment
    pur = 0
    for cl in set(c):
        mask = c == cl
        if mask.sum():
            pur += Counter(np.array(y_idx)[mask]).most_common(1)[0][1]
    pur /= len(y_idx)
    return {"tag": tag, "purity": round(pur, 4),
            "NMI": round(nmi(y_idx, c), 4), "ARI": round(ari(y_idx, c), 4)}

def supervised(texts, y_idx, tag):
    X = TfidfVectorizer(max_features=4000).fit_transform(texts)
    acc = cross_val_score(LogisticRegression(max_iter=400), X, y_idx, cv=5).mean()
    return {"tag": tag, "cv_accuracy": round(float(acc), 4)}

def marker_only(texts, y_idx):
    """Probe: can JUST level-marker presence predict the level? (=how much is pure template)"""
    feats = np.array([[1.0 if m in t else 0.0 for m in
                       ["Base", "Test", "Source", "Query", "Output", "Response"]]
                      for t in texts])
    acc = cross_val_score(LogisticRegression(max_iter=400), feats, y_idx, cv=5).mean()
    return round(float(acc), 4)

def main():
    os.makedirs("data/ccts_axis", exist_ok=True)
    texts, levels = load()
    y = np.array([LEVELS.index(l) for l in levels])
    rng = np.random.default_rng(SEED)
    y_shuf = rng.permutation(y)
    stripped = [strip_markers(t) for t in texts]

    res = {
        "n": len(texts), "level_dist": dict(Counter(levels)),
        "cluster_raw":      cluster_align(texts, y, "raw_with_markers"),
        "cluster_stripped": cluster_align(stripped, y, "markers_stripped"),
        "cluster_random_floor": cluster_align(texts, y_shuf, "random_label_floor"),
        "supervised_raw":      supervised(texts, y, "raw"),
        "supervised_stripped": supervised(stripped, y, "stripped"),
        "marker_only_cv_accuracy": marker_only(texts, y),
        "majority_floor": round(max(Counter(levels).values()) / len(levels) / 0 if False
                                else max(Counter(levels).values()) / len(texts), 4),
        "seed": SEED, "source": DATA,
        "note": "synthetic template data; raw alignment is by-construction. "
                "decisive contrast = stripped vs random_floor.",
    }
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
