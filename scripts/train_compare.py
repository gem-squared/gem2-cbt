"""Train baseline vs CBT-v0 and produce a comparison table.

Usage:
  python scripts/train_compare.py --epochs 20 --batch 128 --lam 0.5
"""
import argparse, json, os, sys, time, math
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

torch.set_num_threads(max(1, os.cpu_count() or 1))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbt.tokenizer import CharTokenizer, ContractVocab, LEVELS
from cbt.dataset import CBTDataset, read_jsonl
from cbt.model import CBT, CBTConfig
from cbt.losses import total_loss
from cbt.fingerprint import (compute_dataset_hash, ckpt_path, write_manifest,
                             assert_frozen_hash)

CONFIGS = {
    # name                              use_pixel  use_boundary
    "baseline_lm":                      (False,    False),
    "cbt_textonly":                     (False,    True),
    "cbt_v0":                           (True,     True),
    # Within-level concept-contract shuffle ablation (v1 — INVALID control; kept for reference).
    "cbt_v0_concept_contract_shuffled": (True,     True),
}

# Display labels for report tables
CONFIG_LABELS = {
    "cbt_v0_concept_contract_shuffled": "cbt_v0_shuffled (within-level; concept-contract only)",
}

SWEEP_CONFIGS = ["baseline_lm", "cbt_textonly", "cbt_v0"]  # main sweep (U5)
SHUFFLE_CONFIGS = ["cbt_v0_concept_contract_shuffled"]      # ablation sweep (U6)

# --------------------------------------------------------------------------
# V2 ABLATION LADDER CONFIGS
# Each entry: (use_level, use_contract, use_boundary, special_mode)
# special_mode: None | "random_contract" | "contract_only" | "heuristic"
# --------------------------------------------------------------------------
CONFIGS_V2 = {
    # name              use_level  use_contract  use_boundary  special_mode
    "text_only":        (False,    False,        True,         None),
    "text_level":       (True,     False,        True,         None),
    "text_contract":    (False,    True,         True,         None),
    "cbt_v0":           (True,     True,         True,         None),
    "random_contract":  (True,     True,         True,         "random_contract"),
    "contract_only":    (False,    True,         True,         "contract_only"),
    "heuristic_baseline":(False,   False,        False,        "heuristic"),
}

V2_TRAINED_CONFIGS = [k for k, v in CONFIGS_V2.items() if v[3] != "heuristic"]
V2_SWEEP_ORDER = ["text_only", "text_level", "text_contract", "cbt_v0",
                  "random_contract", "contract_only", "heuristic_baseline"]


def set_seed(s):
    torch.manual_seed(s)


def make_shuffled_concept_contract_mapping(all_rows, seed):
    """Within-level contract permutation for concept level only.
    Context (role-preserve) and task (facts-only) each have one contract →
    shuffle is a no-op there. Returns {original_contract: shuffled_contract}."""
    import random as _random
    rng = _random.Random(seed)
    concept_contracts = sorted(set(r["contract"] for r in all_rows
                                   if r["level"] == "concept"))
    shuffled = concept_contracts[:]
    rng.shuffle(shuffled)
    return dict(zip(concept_contracts, shuffled))


def apply_contract_mapping(rows, mapping):
    """Apply contract permutation to rows; unmapped contracts pass through."""
    return [{**r, "contract": mapping.get(r["contract"], r["contract"])}
            for r in rows]


def make_random_contract_rows(rows, seed):
    """Per-example random contract assignment (v2 valid control).
    Properties: (a) within-level, (b) example-wise independent, (c) deterministic by seed,
    (d) label-preserving — original label unchanged.
    Prefers a contract different from the true one when alternatives exist.
    """
    import random as _random
    rng = _random.Random(seed)
    # Build per-level contract pools
    level_contracts = defaultdict(set)
    for r in rows:
        level_contracts[r["level"]].add(r["contract"])
    level_contracts = {lv: sorted(cs) for lv, cs in level_contracts.items()}
    result = []
    for r in rows:
        pool = level_contracts[r["level"]]
        others = [c for c in pool if c != r["contract"]]
        new_c = rng.choice(others) if others else r["contract"]
        result.append({**r, "contract": new_c})  # label PRESERVED
    return result


def _compute_auroc(probs, labels):
    """Trapezoidal AUROC. Returns None if only one class present."""
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0 or not probs:
        return None
    sorted_pairs = sorted(zip(probs, labels), key=lambda x: -x[0])
    tp, fp, prev_tpr, prev_fpr, auc = 0, 0, 0.0, 0.0, 0.0
    for _, lab in sorted_pairs:
        if lab == 1: tp += 1
        else: fp += 1
        tpr = tp / n_pos; fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_tpr, prev_fpr = tpr, fpr
    return auc


def evaluate(model, loader, device, forward_fn=None):
    """Evaluate model. Returns per-level acc/F1/error-rates + AUROC + confusion.
    forward_fn(model, batch) overrides default forward call.
    AUROC is None when there's no probability output (heuristic path)."""
    model.eval()
    tot_lm, tot_tok = 0.0, 0
    correct, n = 0, 0
    tp, fp, tn, fn = 0, 0, 0, 0
    false_accept, n_incompat = 0, 0
    false_reject, n_compat = 0, 0
    # per_level[lv] = dict of counts
    per_level = defaultdict(lambda: dict(c=0, n=0, tp=0, fp=0, tn=0, fn=0,
                                         fa=0, ni=0, fr=0, nc=0))
    probs_all, labs_all = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            if forward_fn is not None:
                lm_logits, bnd_logits = forward_fn(model, b)
            else:
                lm_logits, bnd_logits = model(
                    b["input_ids"], attn_pad=b["attn_pad"],
                    level=b["level"], contract=b["contract"])
            B, T, V = lm_logits.shape
            tgt = b["lm_targets"]
            mask = tgt != -100
            ce = torch.nn.functional.cross_entropy(
                lm_logits.view(B * T, V), tgt.view(B * T),
                ignore_index=-100, reduction="sum")
            tot_lm += ce.item(); tot_tok += int(mask.sum().item())
            if bnd_logits is not None:
                prob = torch.softmax(bnd_logits, dim=-1)[:, 1]  # P(compatible)
                pred = bnd_logits.argmax(-1)
                lab = b["label"]
                correct += int((pred == lab).sum().item()); n += lab.numel()
                probs_all.extend(prob.cpu().tolist())
                labs_all.extend(lab.cpu().tolist())
                incompat_m = lab == 0; compat_m = lab == 1
                n_incompat += int(incompat_m.sum().item())
                n_compat   += int(compat_m.sum().item())
                false_accept += int(((pred == 1) & incompat_m).sum().item())
                false_reject += int(((pred == 0) & compat_m).sum().item())
                _tp = int(((pred == 1) & compat_m).sum().item())
                _fp = int(((pred == 1) & incompat_m).sum().item())
                _tn = int(((pred == 0) & incompat_m).sum().item())
                _fn = int(((pred == 0) & compat_m).sum().item())
                tp += _tp; fp += _fp; tn += _tn; fn += _fn
                for lv, c_ok, pr, lb in zip(b["level"].tolist(),
                                            (pred == lab).tolist(),
                                            pred.tolist(), lab.tolist()):
                    pld = per_level[lv]
                    pld["c"] += int(c_ok); pld["n"] += 1
                    if lb == 1:
                        pld["nc"] += 1
                        if pr == 1: pld["tp"] += 1
                        else:       pld["fn"] += 1; pld["fr"] += 1
                    else:
                        pld["ni"] += 1
                        if pr == 0: pld["tn"] += 1
                        else:       pld["fp"] += 1; pld["fa"] += 1
    lm_ppl_loss = tot_lm / max(tot_tok, 1)
    res = {"lm_loss": lm_ppl_loss}
    if n > 0:
        res["boundary_acc"] = correct / n
        res["unsafe_accept_rate"] = (false_accept / n_incompat) if n_incompat else None
        res["over_reject_rate"]   = (false_reject / n_compat) if n_compat else None
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        res["f1"] = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        res["auroc"] = _compute_auroc(probs_all, labs_all)
        res["confusion"] = [[tn, fp], [fn, tp]]
        pl_acc, pl_ua, pl_or, pl_f1 = {}, {}, {}, {}
        for lv, pld in sorted(per_level.items()):
            lvn = LEVELS[lv]
            pl_acc[lvn] = pld["c"] / pld["n"] if pld["n"] else None
            pl_ua[lvn]  = pld["fa"] / pld["ni"] if pld["ni"] else None
            pl_or[lvn]  = pld["fr"] / pld["nc"] if pld["nc"] else None
            tp2, fp2, fn2 = pld["tp"], pld["fp"], pld["fn"]
            p2 = tp2/(tp2+fp2) if (tp2+fp2) else 0.0
            r2 = tp2/(tp2+fn2) if (tp2+fn2) else 0.0
            pl_f1[lvn] = 2*p2*r2/(p2+r2) if (p2+r2) else 0.0
        res["per_level_acc"] = pl_acc
        res["per_level_unsafe_accept"] = pl_ua
        res["per_level_over_reject"]   = pl_or
        res["per_level_f1"] = pl_f1
    return res


def check_seed_variance(tok, cv, train_ds):
    """Verify --seed affects init + shuffle but not data split (which is file-based)."""
    from cbt.model import CBT, CBTConfig
    cfg = CBTConfig(vocab_size=tok.vocab_size, block_size=train_ds.block,
                    n_layer=2, n_head=2, n_embd=96, n_contracts=cv.size)
    set_seed(0); m0 = CBT(cfg)
    set_seed(1); m1 = CBT(cfg)
    p0 = next(m0.parameters()).data.flatten()[:8]
    p1 = next(m1.parameters()).data.flatten()[:8]
    assert not torch.allclose(p0, p1), "SEED_CHECK FAIL: seeds 0 and 1 give identical init weights"
    set_seed(0); idx0 = torch.randperm(len(train_ds))[:5].tolist()
    set_seed(1); idx1 = torch.randperm(len(train_ds))[:5].tolist()
    assert idx0 != idx1, "SEED_CHECK FAIL: seeds 0 and 1 give identical shuffle order"
    print(f"seed_check: PASS | init differs seed0 vs seed1 | "
          f"shuffle[0]={idx0[:3]} shuffle[1]={idx1[:3]} | "
          f"data split = file-based (unchanged by train seed)")


def train_one(name, use_pixel, use_boundary, data, args, device, dataset_hash):
    """Resumable trainer. Trains toward args.epochs total, but stops early when
    args.time_budget seconds elapse, checkpointing so a later call can resume.
    Returns (result_dict_or_None, done_bool, epochs_done)."""
    tok, cv, train_ds, test_ds = data
    cfg = CBTConfig(
        vocab_size=tok.vocab_size, block_size=train_ds.block,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, n_contracts=cv.size,
        use_pixel=use_pixel, use_boundary=use_boundary)
    set_seed(args.seed)
    model = CBT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ckpt = ckpt_path(args.data, dataset_hash, name, args.seed)
    epochs_done = 0
    if os.path.exists(ckpt) and not args.fresh:
        st = torch.load(ckpt, map_location=device)
        if st.get("epochs_done", 0) > 0:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            epochs_done = st["epochs_done"]
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    t0 = time.time()
    while epochs_done < args.epochs and (time.time() - t0) < args.time_budget:
        model.train()
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            lm_logits, bnd_logits = model(
                b["input_ids"], attn_pad=b["attn_pad"],
                level=b["level"], contract=b["contract"])
            loss = total_loss(lm_logits, b["lm_targets"], bnd_logits,
                              b["label"], lam=args.lam)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        epochs_done += 1
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epochs_done": epochs_done}, ckpt)
    write_manifest(args.data, dataset_hash, name, args)
    done = epochs_done >= args.epochs
    if not done:
        print(f"[{name}] resumed -> {epochs_done}/{args.epochs} epochs "
              f"({time.time()-t0:.1f}s); rerun to continue")
        return None, False, epochs_done
    test_loader = DataLoader(test_ds, batch_size=256)
    res = evaluate(model, test_loader, device)
    res["params"] = model.num_params(); res["epochs"] = epochs_done
    print(f"[{name}] DONE {res}")
    return res, True, epochs_done


def heuristic_evaluate(test_ds, device):
    """Majority-class heuristic baseline: always predict the majority label.
    AUROC = None (no probability output). Confusion matrix and per-level F1 computed."""
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=256)
    labels, levels = [], []
    for b in loader:
        labels.extend(b["label"].tolist())
        levels.extend(b["level"].tolist())
    n = len(labels)
    pos = sum(labels)
    maj_label = 1 if pos >= n - pos else 0
    # Confusion for majority-class predictor
    if maj_label == 1:
        tp = pos; fp = n - pos; tn = fn = 0
    else:
        tn = n - pos; fn = pos; tp = fp = 0
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    # Per-level majority-class accuracy + F1
    pl_acc, pl_f1 = {}, {}
    for lv_id in range(len(LEVELS)):
        lv_labs = [l for l, lev in zip(labels, levels) if lev == lv_id]
        if not lv_labs:
            continue
        lv_pos = sum(lv_labs)
        lv_maj = 1 if lv_pos >= len(lv_labs) - lv_pos else 0
        lv_n = len(lv_labs)
        if lv_maj == 1:
            lv_tp = lv_pos; lv_fp = lv_n - lv_pos; lv_fn = 0
        else:
            lv_tp = 0; lv_fp = 0; lv_fn = lv_pos
        lv_acc = (lv_tp + (lv_n - lv_pos if lv_maj == 0 else 0)) / lv_n
        # Simpler: majority-class accuracy = max(lv_pos, lv_n-lv_pos) / lv_n
        lv_acc = max(lv_pos, lv_n - lv_pos) / lv_n
        lv_prec = lv_tp / (lv_tp + lv_fp) if (lv_tp + lv_fp) else 0.0
        lv_rec  = lv_tp / (lv_tp + lv_fn) if (lv_tp + lv_fn) else 0.0
        pl_acc[LEVELS[lv_id]] = lv_acc
        pl_f1[LEVELS[lv_id]]  = 2*lv_prec*lv_rec/(lv_prec+lv_rec) if (lv_prec+lv_rec) else 0.0
    return {"lm_loss": None, "boundary_acc": acc, "f1": f1, "auroc": None,
            "confusion": [[tn, fp], [fn, tp]],
            "unsafe_accept_rate": None, "over_reject_rate": None,
            "per_level_acc": pl_acc, "per_level_f1": pl_f1,
            "per_level_unsafe_accept": {}, "per_level_over_reject": {},
            "params": 0, "epochs": 0, "heuristic": True, "majority_label": maj_label}


def train_one_v2(name, use_level, use_contract, use_boundary, special_mode,
                 data, args, device, dataset_hash):
    """v2 variant of train_one: uses independent use_level/use_contract flags.
    special_mode: None | 'random_contract' | 'contract_only' | 'heuristic'
    """
    tok, cv, train_ds, test_ds = data
    if special_mode == "heuristic":
        res = heuristic_evaluate(test_ds, device)
        print(f"[{name}] HEURISTIC {res}")
        return res, True, 0

    cfg = CBTConfig(
        vocab_size=tok.vocab_size, block_size=train_ds.block,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, n_contracts=cv.size,
        use_level=use_level, use_contract=use_contract, use_boundary=use_boundary)
    set_seed(args.seed)
    model = CBT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ckpt = ckpt_path(args.data, dataset_hash, name, args.seed)
    epochs_done = 0
    if os.path.exists(ckpt) and not args.fresh:
        st = torch.load(ckpt, map_location=device)
        if st.get("epochs_done", 0) > 0:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            epochs_done = st["epochs_done"]
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    t0 = time.time()
    while epochs_done < args.epochs and (time.time() - t0) < args.time_budget:
        model.train()
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            ids = b["input_ids"]
            if special_mode == "contract_only":
                # zero out text: only contract embedding contributes
                ids = torch.zeros_like(ids)
            lm_logits, bnd_logits = model(
                ids, attn_pad=b["attn_pad"],
                level=b["level"], contract=b["contract"])
            loss = total_loss(lm_logits, b["lm_targets"], bnd_logits,
                              b["label"], lam=args.lam)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        epochs_done += 1
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epochs_done": epochs_done}, ckpt)
    write_manifest(args.data, dataset_hash, name, args)
    done = epochs_done >= args.epochs
    if not done:
        print(f"[{name}] resumed -> {epochs_done}/{args.epochs} epochs "
              f"({time.time()-t0:.1f}s); rerun to continue")
        return None, False, epochs_done
    test_loader = DataLoader(test_ds, batch_size=256)

    # For contract_only eval: zero out input_ids
    def _eval_forward(model, b):
        ids = b["input_ids"]
        if special_mode == "contract_only":
            ids = torch.zeros_like(ids)
        return model(ids, attn_pad=b["attn_pad"],
                     level=b["level"], contract=b["contract"])

    res = evaluate(model, test_loader, device, forward_fn=_eval_forward)
    res["params"] = model.num_params(); res["epochs"] = epochs_done
    print(f"[{name}] DONE {res}")
    return res, True, epochs_done


def build_table(results, maj, args):
    """Build comparison table. results keyed by {config}_seed{seed}."""
    def fmt(x):
        return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    seed = args.seed
    order = [c for c in CONFIGS if f"{c}_seed{seed}" in results]
    lines = []
    lines.append("# CBT-v0 — baseline vs CBT comparison\n")
    lines.append(f"_Config: epochs={args.epochs}, batch={args.batch}, lr={args.lr}, "
                 f"lambda={args.lam}, n_layer={args.n_layer}, n_head={args.n_head}, "
                 f"n_embd={args.n_embd}, seed={seed}_\n")
    lines.append(f"_Majority-class boundary acc (reference) = {maj:.3f}._\n")
    lines.append("| Model | LM loss ↓ | Boundary acc ↑ | Unsafe Accept Rate ↓ | "
                 "Over Reject Rate ↓ | concept | context | task | params |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name in order:
        r = results[f"{name}_seed{seed}"]
        pl = r.get("per_level_acc", {})
        label = CONFIG_LABELS.get(name, name)
        lines.append(
            "| {m} | {lm} | {ba} | {ua} | {or_} | {c} | {cx} | {tk} | {p} |".format(
                m=label, lm=fmt(r["lm_loss"]),
                ba=fmt(r.get("boundary_acc")),
                ua=fmt(r.get("unsafe_accept_rate")),
                or_=fmt(r.get("over_reject_rate")),
                c=fmt(pl.get("concept")), cx=fmt(pl.get("context")),
                tk=fmt(pl.get("task")), p=r["params"]))
    lines.append("\n**Read:** `baseline_lm` has no boundary head (acc shown as —; "
                 "compare against majority-class reference). `cbt_textonly` adds a "
                 "boundary head but no contract/level injection. `cbt_v0` adds "
                 "semantic-pixel (level+contract) injection.\n"
                 "**Unsafe Accept Rate** = incompatible predicted compatible (↓ good). "
                 "**Over Reject Rate** = compatible predicted incompatible (↓ good).\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", choices=["v1", "v2"],
                    help="v1=original configs, v2=ablation ladder on dataset_v2")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--only", default=None, help="train a single config by name")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=2)
    ap.add_argument("--n_embd", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_budget", type=float, default=38.0,
                    help="max seconds to train per call before checkpointing")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
    ap.add_argument("--out", default="papers/results.md")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows = read_jsonl(os.path.join(args.data, "train.jsonl"))
    test_rows = read_jsonl(os.path.join(args.data, "test.jsonl"))
    all_text = [r["text"] for r in train_rows + test_rows]
    tok = CharTokenizer.from_texts(all_text)
    cv = ContractVocab([r["contract"] for r in train_rows + test_rows])
    max_len = max(len(tok.encode(t)) for t in all_text) + 1
    block = min(256, max_len)
    train_ds = CBTDataset(train_rows, tok, cv, block)
    test_ds = CBTDataset(test_rows, tok, cv, block)
    print(f"device={device} vocab={tok.vocab_size} contracts={cv.size} "
          f"block={block} train={len(train_ds)} test={len(test_ds)} "
          f"trunc(train/test)={train_ds.n_trunc}/{test_ds.n_trunc}")

    maj = sum(r["label"] for r in test_rows)
    maj = max(maj, len(test_rows) - maj) / len(test_rows)
    print(f"majority-class boundary acc (reference) = {maj:.3f}")

    data = (tok, cv, train_ds, test_ds)
    dataset_hash = assert_frozen_hash(args.data)  # FAIL FAST if dataset changed
    print(f"dataset_hash={dataset_hash} (frozen hash verified)")
    check_seed_variance(tok, cv, train_ds)

    if args.version == "v2":
        # V2 BRANCH: ablation ladder with random_contract
        rand_train_rows = make_random_contract_rows(train_rows, args.seed)
        rand_test_rows  = make_random_contract_rows(test_rows, args.seed)
        rand_train_ds = CBTDataset(rand_train_rows, tok, cv, block)
        rand_test_ds  = CBTDataset(rand_test_rows,  tok, cv, block)
        random_contract_data = (tok, cv, rand_train_ds, rand_test_ds)

        res_path = os.path.join(args.data, "results_v2.json")
        results = {}
        if os.path.exists(res_path):
            try:
                results = json.load(open(res_path)).get("results", {})
            except Exception:
                results = {}

        to_run = [args.only] if args.only else V2_SWEEP_ORDER
        all_done = True
        n_configs = len(CONFIGS_V2)
        for name in to_run:
            if name not in CONFIGS_V2:
                print(f"WARNING: unknown v2 config '{name}', skipping")
                continue
            key = f"{name}_seed{args.seed}"
            if key in results:
                print(f"[{name}] already done (seed {args.seed}), skipping")
                continue
            ul, uc, ub, special = CONFIGS_V2[name]
            run_data = random_contract_data if special == "random_contract" else data
            res, done, _ = train_one_v2(name, ul, uc, ub, special,
                                         run_data, args, device, dataset_hash)
            if done:
                results[key] = res
            else:
                all_done = False
                break

        with open(res_path, "w") as f:
            json.dump({"majority": maj, "results": results, "args": vars(args),
                       "version": "v2"}, f, indent=2)
        seed_keys = [k for k in results if k.endswith(f"_seed{args.seed}")]
        print(f"\nv2 results: {len(seed_keys)}/{n_configs} configs done for seed {args.seed}")
        print("\nALL_DONE" if all_done and len(seed_keys) == n_configs else
              "\nNOT_DONE (rerun same command to continue)")
        return

    # V1 BRANCH (unchanged)
    shuf_mapping = make_shuffled_concept_contract_mapping(
        train_rows + test_rows, args.seed)
    shuf_train_ds = CBTDataset(
        apply_contract_mapping(train_rows, shuf_mapping), tok, cv, block)
    shuf_test_ds = CBTDataset(
        apply_contract_mapping(test_rows, shuf_mapping), tok, cv, block)
    shuffled_data = (tok, cv, shuf_train_ds, shuf_test_ds)

    res_path = os.path.join(args.data, "results.json")
    results = {}
    if os.path.exists(res_path):
        try:
            results = json.load(open(res_path)).get("results", {})
        except Exception:
            results = {}

    to_run = [args.only] if args.only else list(CONFIGS.keys())
    all_done = True
    for name in to_run:
        up, ub = CONFIGS[name]
        run_data = shuffled_data if name in SHUFFLE_CONFIGS else data
        res, done, _ = train_one(name, up, ub, run_data, args, device, dataset_hash)
        if done:
            results[f"{name}_seed{args.seed}"] = res  # seed-aware, non-clobbering
        else:
            all_done = False
            break  # finish this config across calls before moving on

    with open(res_path, "w") as f:
        json.dump({"majority": maj, "results": results, "args": vars(args)}, f, indent=2)
    table = build_table(results, maj, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    seed_keys = [k for k in results if k.endswith(f"_seed{args.seed}")]
    print("\nALL_DONE" if all_done and len(seed_keys) == len(CONFIGS) else
          "\nNOT_DONE (rerun same command to continue)")


if __name__ == "__main__":
    main()
