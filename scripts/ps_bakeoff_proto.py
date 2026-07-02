#!/usr/bin/env python3
"""WP-ST-17 PROTOTYPE (CW guardrail probe, NOT the final deliverable).

De-risks the PS bake-off before CC builds the full harness. Answers ONE question:
after a leakage audit strips the fixture scaffolding that near-deterministically tags
the level, is there ANY real signal left for a density+distance PS to beat simpler
baselines on -- or does everything saturate (WP-10 trap)?

Deterministic, CPU only, no LLM, no network. Fits everything on TRAIN, applies to TEST.
"""
from __future__ import annotations
import json, re, math, collections, statistics, sys, os

DATA = None
for c in ("/sessions/clever-laughing-meitner/mnt/gem2-CBT",
          os.path.expanduser("~/GEM-Squared-Universe/gem2-CBT")):
    if os.path.exists(os.path.join(c, "data/processed_v2/train.jsonl")):
        DATA = c; break
assert DATA, "processed_v2 not found"
LEVELS = ["task", "context", "concept"]
SEP_CEILING = 0.30          # leakage audit: strip tokens whose (P(t|L)-max_other) exceeds this
random_seed = 0

def load(split):
    return [json.loads(l) for l in open(f"{DATA}/data/processed_v2/{split}.jsonl")]

def toks(s): return re.findall(r"[a-z0-9]+", s.lower())

# ---------- leakage audit: find near-deterministic level-indicator tokens on TRAIN ----------
def audit_leak(rows):
    df = {L: collections.Counter() for L in LEVELS}; n = {L: 0 for L in LEVELS}
    for r in rows:
        L = r["level"]; n[L] += 1
        for t in set(toks(r["text"])): df[L][t] += 1
    allt = set().union(*[set(df[L]) for L in LEVELS])
    leak = set()
    sep = {}
    for t in allt:
        ps = {L: df[L][t]/n[L] for L in LEVELS}
        top = max(ps.values()); other = sorted(ps.values())[-2]
        s = top - other
        sep[t] = s
        if s > SEP_CEILING:
            leak.add(t)
    # also strip contract-name vocabulary (leaks verbatim per audit)
    return leak, sep

def strip_view(text, leak):
    return " ".join(t for t in toks(text) if t not in leak)

# ---------- features: density (close-set overlap) + distance (centroid) fit on TRAIN ----------
def build_features(train_texts, train_levels):
    # per-level close-set = top tokens by lift on TRAIN; centroid = mean tf vector (bag of that set)
    df = {L: collections.Counter() for L in LEVELS}; n = {L: 0 for L in LEVELS}
    for txt, L in zip(train_texts, train_levels):
        n[L] += 1
        for t in set(txt.split()): df[L][t] += 1
    allt = set().union(*[set(df[L]) for L in LEVELS]) or {"__none__"}
    closeset = {}
    for L in LEVELS:
        lift = []
        for t in allt:
            pL = df[L][t]/max(n[L],1)
            po = max((df[O][t]/max(n[O],1)) for O in LEVELS if O != L)
            lift.append((pL-po, t))
        lift.sort(reverse=True)
        closeset[L] = {t for _, t in lift[:30]}
    # centroid in the union-closeset space
    space = sorted(set().union(*closeset.values())) or ["__none__"]
    idx = {t: i for i, t in enumerate(space)}
    def vec(txt):
        v = [0.0]*len(space); ct = collections.Counter(txt.split())
        for t, c in ct.items():
            if t in idx: v[idx[t]] = c
        nrm = math.sqrt(sum(x*x for x in v)) or 1.0
        return [x/nrm for x in v]
    cent = {}
    for L in LEVELS:
        vs = [vec(txt) for txt, lv in zip(train_texts, train_levels) if lv == L] or [[0.0]*len(space)]
        cent[L] = [sum(col)/len(vs) for col in zip(*vs)]
    def feats(txt, L):
        tk = txt.split(); cs = closeset[L]
        density = (sum(1 for t in tk if t in cs)/len(tk)) if tk else 0.0
        v = vec(txt); c = cent[L]
        dist = math.sqrt(sum((a-b)**2 for a, b in zip(v, c)))   # euclidean to centroid
        prox = 1.0/(1.0+dist)                                    # higher = closer
        return density, prox
    return feats

# ---------- scorers (all consume the SAME (density,prox) pair) ----------
def scorers():
    return {
        "density_only":   lambda d, p: d,
        "distance_only":  lambda d, p: p,
        "density+dist":   lambda d, p: (d*p) ** 0.5,   # geometric mean = the PS candidate
        # logistic-regression weights are fit per-fold below (needs training) -> handled separately
    }

def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg: return float("nan")
    wins = ties = 0
    negs = sorted(neg)
    import bisect
    for s in pos:
        lo = bisect.bisect_left(negs, s); hi = bisect.bisect_right(negs, s)
        wins += lo; ties += (hi-lo)
    return (wins + 0.5*ties)/(len(pos)*len(neg))

def fit_logreg(X, y, iters=300, lr=0.5):
    w = [0.0, 0.0]; b = 0.0
    for _ in range(iters):
        gw = [0.0, 0.0]; gb = 0.0
        for xi, yi in zip(X, y):
            z = w[0]*xi[0]+w[1]*xi[1]+b
            pr = 1/(1+math.exp(-max(-30,min(30,z))))
            e = pr-yi
            gw[0]+=e*xi[0]; gw[1]+=e*xi[1]; gb+=e
        m=len(X); w[0]-=lr*gw[0]/m; w[1]-=lr*gw[1]/m; b-=lr*gb/m
    return lambda d,p: 1/(1+math.exp(-max(-30,min(30,w[0]*d+w[1]*p+b)))), (w,b)

def evaluate(view_name, train, test, leak):
    tr_txt = [strip_view(r["text"], leak) if view_name=="STRIPPED" else " ".join(toks(r["text"])) for r in train]
    te_txt = [strip_view(r["text"], leak) if view_name=="STRIPPED" else " ".join(toks(r["text"])) for r in test]
    tr_lv = [r["level"] for r in train]; te_lv = [r["level"] for r in test]
    feats = build_features(tr_txt, tr_lv)
    print(f"\n===== VIEW: {view_name} =====")
    for L in LEVELS:
        # one-vs-rest: does PS(L) rank level==L above others
        te_feat = [feats(t, L) for t in te_txt]
        y = [1 if lv == L else 0 for lv in te_lv]
        # simple scorers
        row = {}
        for name, fn in scorers().items():
            row[name] = auc([fn(d, p) for d, p in te_feat], y)
        # logreg fit on TRAIN one-vs-rest
        tr_feat = [feats(t, L) for t in tr_txt]
        ytr = [1 if lv == L else 0 for lv in tr_lv]
        lrfn, _ = fit_logreg(tr_feat, ytr)
        row["logreg2"] = auc([lrfn(d, p) for d, p in te_feat], y)
        print(f"  {L:8} | " + " ".join(f"{k}={row[k]:.3f}" for k in
              ["density_only","distance_only","density+dist","logreg2"]))
    return

if __name__ == "__main__":
    train, test = load("train"), load("test")
    leak, sep = audit_leak(train)
    top = sorted(sep.items(), key=lambda kv: -kv[1])[:12]
    print(f"leakage audit: {len(leak)} tokens exceed sep>{SEP_CEILING}")
    print("  top leak tokens (sep):", ", ".join(f"{t}({s:.2f})" for t, s in top))
    evaluate("RAW", train, test, leak)
    evaluate("STRIPPED", train, test, leak)
    print("\n[read] RAW ~1.0 across the board = leakage confirmed. "
          "STRIPPED tells us if any real level signal survives.")
