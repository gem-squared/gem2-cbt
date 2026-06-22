"""U4 heuristic separability gate for Dataset v2.

Checks whether binary labels are trivially predictable from surface cues.
Gate threshold: ≤ 0.65 accuracy per level on EACH classifier.

Classifiers (no neural training required):
  1. length-only: optimal threshold on text length
  2. char-bigram:  logistic regression on char bigram bag-of-words (L-BFGS, PyTorch)
  3. contract-id:  logistic regression on one-hot contract ID

Informational: contract-id very high → trivially ID-separable (bad);
               bigram very high → surface-separable, contract unnecessary (bad).

On PASS: freezes canonical v2 hash to data_dir/frozen_dataset_hash.json.
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbt.dataset import read_jsonl
from cbt.fingerprint import compute_dataset_hash, freeze_dataset_hash

LEVELS = ["concept", "context", "task"]
GATE_THRESH = 0.65


def by_level(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["level"]].append(r)
    return d


def length_accuracy(train_rows, test_rows):
    """Accuracy of the best length-threshold classifier on test."""
    lens_tr = [len(r["text"]) for r in train_rows]
    labs_tr = [r["label"] for r in train_rows]
    lens_te = [len(r["text"]) for r in test_rows]
    labs_te = [r["label"] for r in test_rows]
    # majority class baseline
    maj = sum(labs_te) / len(labs_te)
    best_acc = max(maj, 1 - maj)
    for t in sorted(set(lens_tr)):
        for flip in (False, True):
            preds = [(1 if l > t else 0) ^ int(flip) for l in lens_te]
            acc = sum(p == y for p, y in zip(preds, labs_te)) / len(labs_te)
            best_acc = max(best_acc, acc)
    return best_acc


def _logreg(X_tr, y_tr, X_te, y_te, epochs=10):
    """L-BFGS logistic regression; returns test accuracy."""
    model = nn.Linear(X_tr.shape[1], 1, bias=True)
    opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=20)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(X_tr).squeeze(-1), y_tr)
        loss.backward()
        return loss

    for _ in range(epochs):
        opt.step(closure)

    with torch.no_grad():
        preds = (model(X_te).squeeze(-1) > 0).long()
        return (preds == y_te.long()).float().mean().item()


def bigram_accuracy(train_rows, test_rows, top_k=400):
    """Test accuracy of char-bigram logistic regression."""
    counts = Counter()
    for r in train_rows:
        t = r["text"].lower()
        for i in range(len(t) - 1):
            counts[t[i:i+2]] += 1
    vocab = {bg: i for i, (bg, _) in enumerate(counts.most_common(top_k))}
    if not vocab:
        return 0.5

    def feat(rows):
        X = torch.zeros(len(rows), len(vocab))
        y = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
        for i, r in enumerate(rows):
            t = r["text"].lower()
            for j in range(len(t) - 1):
                bg = t[j:j+2]
                if bg in vocab:
                    X[i, vocab[bg]] += 1
        norms = X.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return X / norms, y

    X_tr, y_tr = feat(train_rows)
    X_te, y_te = feat(test_rows)
    return _logreg(X_tr, y_tr, X_te, y_te)


def contract_id_accuracy(train_rows, test_rows):
    """Test accuracy of contract-ID logistic regression."""
    all_contracts = sorted(set(r["contract"] for r in train_rows + test_rows))
    c2i = {c: i for i, c in enumerate(all_contracts)}
    V = len(all_contracts)

    def feat(rows):
        X = torch.zeros(len(rows), V)
        y = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
        for i, r in enumerate(rows):
            X[i, c2i[r["contract"]]] = 1.0
        return X, y

    X_tr, y_tr = feat(train_rows)
    X_te, y_te = feat(test_rows)
    return _logreg(X_tr, y_tr, X_te, y_te)


def run_gate(data_dir):
    print(f"\n=== U4 Heuristic Separability Gate: {data_dir} ===\n")
    train_rows = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    test_rows  = read_jsonl(os.path.join(data_dir, "test.jsonl"))
    tr_lv = by_level(train_rows)
    te_lv = by_level(test_rows)

    gate_pass = True
    probe = {}
    header = f"{'level':12s} | {'length':>7} | {'bigram':>7} | {'contract_id':>11} | {'len_gate':>8} | {'bg_gate':>7}"
    print(header)
    print("-" * len(header))
    for lv in LEVELS:
        tr, te = tr_lv[lv], te_lv[lv]
        la = length_accuracy(tr, te)
        ba = bigram_accuracy(tr, te)
        ca = contract_id_accuracy(tr, te)
        gl = "PASS" if la <= GATE_THRESH else "FAIL"
        gb = "PASS" if ba <= GATE_THRESH else "FAIL"
        if la > GATE_THRESH or ba > GATE_THRESH:
            gate_pass = False
        print(f"  {lv:12s} | {la:7.3f} | {ba:7.3f} | {ca:11.3f} | {gl:>8} | {gb:>7}")
        probe[f"length_{lv}"] = round(la, 4)
        probe[f"bigram_{lv}"] = round(ba, 4)
        probe[f"contract_id_{lv}"] = round(ca, 4)

    print()
    if gate_pass:
        print(f"GATE: PASS (all levels ≤ {GATE_THRESH})")
    else:
        print(f"GATE: FAIL — surface-separable level(s) exceed {GATE_THRESH}")
        print("Action required: harden v2 generator and regenerate before freeze.")
    return gate_pass, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--force-freeze", action="store_true",
                    help="freeze even if gate fails (for debugging only)")
    ap.add_argument("--just-gate", action="store_true",
                    help="run gate only, do not freeze")
    args = ap.parse_args()

    gate_pass, probe = run_gate(args.data)

    if args.just_gate:
        sys.exit(0 if gate_pass else 1)

    if not gate_pass and not args.force_freeze:
        sys.exit(1)

    h = compute_dataset_hash(args.data)
    path = freeze_dataset_hash(
        args.data, h,
        notes="U4 heuristic gate PASS — length/bigram ≤ 0.65 all levels",
        probe_results=probe)
    print(f"\nFrozen hash : {h}")
    print(f"Freeze file : {path}")
    print("FREEZE_OK")


if __name__ == "__main__":
    main()
