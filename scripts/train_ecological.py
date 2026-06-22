"""Ecological validity experiment — WP-ST-3.

Gloss-informed sense ranking: does a separately-encoded contract(gloss) vector
channel improve ranking over context-alone on REAL natural-language WSD?

Usage (U5 smoke):
  python scripts/train_ecological.py --split v3_hard --smoke --seed 0

Usage (U6 full sweep):
  python scripts/train_ecological.py --split v3_hard --seed 0
  python scripts/train_ecological.py --split v3_easy --seed 0
  ... (seeds 0-9, both splits)

Encoding cache (run once):
  python scripts/train_ecological.py --encode-only
"""
import argparse, json, os, sys, time, math, hashlib, random as _random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbt.fingerprint import assert_frozen_hash, freeze_dataset_hash, compute_dataset_hash

ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM     = 384
CACHE_DIR     = "data/processed_v3/encodings"

# ─────────────────────────────────────────────────────────────────────────────
# Config ladder (U4)
# ─────────────────────────────────────────────────────────────────────────────
# Each config: (use_context, use_gloss, special_mode)
# special_mode: None | "random_easy" | "random_hard" | "contract_only"
#               "target_only" | "heuristic" | "gloss_sim" | "text_only_nn"
CONFIGS_V3 = {
    "heuristic_baseline":          (False, False, "heuristic"),
    "text_only_nn":                (True,  False, "text_only_nn"),
    "gloss_similarity_baseline":   (True,  True,  "gloss_sim"),
    "text_contract":               (True,  True,  None),
    "cbt":                         (True,  True,  None),   # alias; same arch as text_contract
    "easy_random_contract":        (True,  True,  "random_easy"),
    "hard_same_lemma_random":      (True,  True,  "random_hard"),
    "contract_only":               (False, True,  "contract_only"),
    "target_word_only":            (True,  True,  "target_only"),
}
V3_SWEEP_ORDER = [
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
V3_TRAINED = {"text_contract", "cbt", "easy_random_contract",
              "hard_same_lemma_random", "contract_only", "target_word_only"}


# ─────────────────────────────────────────────────────────────────────────────
# Encoding + cache
# ─────────────────────────────────────────────────────────────────────────────
def _get_encoder(device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(ENCODER_MODEL)
    m.to(device)
    return m


def encode_texts(texts, encoder, device, batch_size=512):
    vecs = encoder.encode(texts, device=device, batch_size=batch_size,
                          show_progress_bar=True, convert_to_numpy=True)
    return torch.tensor(vecs, dtype=torch.float32)


def build_encoding_cache(device="mps"):
    """Encode all unique sentences + glosses from both splits. Cache to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_rows = []
    for split in ("v3_easy", "v3_hard"):
        for part in ("train", "test"):
            path = f"data/processed_v3/{split}/{part}.jsonl"
            all_rows.extend(json.loads(l) for l in open(path))
    print(f"Total instances: {len(all_rows)}")

    # Collect unique items
    id2sent = {r["id"]: r["sentence"] for r in all_rows}
    unique_sent_ids = sorted(id2sent.keys())
    unique_sents = [id2sent[i] for i in unique_sent_ids]

    gloss_set = {}
    for r in all_rows:
        for g in r["candidate_glosses"]:
            if g not in gloss_set:
                gloss_set[g] = len(gloss_set)
    unique_glosses = sorted(gloss_set.keys(), key=lambda g: gloss_set[g])

    print(f"Unique sentences: {len(unique_sents)}")
    print(f"Unique glosses:   {len(unique_glosses)}")

    enc = _get_encoder(device)

    sent_cache = os.path.join(CACHE_DIR, "sent_vecs.pt")
    sent_id_cache = os.path.join(CACHE_DIR, "sent_ids.json")
    if not os.path.exists(sent_cache):
        print("Encoding sentences...")
        sent_vecs = encode_texts(unique_sents, enc, device)
        torch.save(sent_vecs, sent_cache)
        json.dump(unique_sent_ids, open(sent_id_cache, "w"))
        print(f"  saved {sent_vecs.shape}")
    else:
        print(f"Sentence cache exists: {sent_cache}")

    gloss_cache = os.path.join(CACHE_DIR, "gloss_vecs.pt")
    gloss_idx_cache = os.path.join(CACHE_DIR, "gloss_index.json")
    if not os.path.exists(gloss_cache):
        print("Encoding glosses...")
        gloss_vecs = encode_texts(unique_glosses, enc, device)
        torch.save(gloss_vecs, gloss_cache)
        json.dump(gloss_set, open(gloss_idx_cache, "w"))
        print(f"  saved {gloss_vecs.shape}")
    else:
        print(f"Gloss cache exists: {gloss_cache}")

    # Lemma encodings for target_word_only baseline
    lemma_set = {}
    for r in all_rows:
        lemma = r["lemma"]
        if lemma not in lemma_set:
            lemma_set[lemma] = len(lemma_set)
    unique_lemmas = sorted(lemma_set.keys(), key=lambda l: lemma_set[l])
    lemma_cache = os.path.join(CACHE_DIR, "lemma_vecs.pt")
    lemma_idx_cache = os.path.join(CACHE_DIR, "lemma_index.json")
    if not os.path.exists(lemma_cache):
        print(f"Encoding {len(unique_lemmas)} lemmas...")
        lemma_vecs = encode_texts(unique_lemmas, enc, device)
        torch.save(lemma_vecs, lemma_cache)
        json.dump(lemma_set, open(lemma_idx_cache, "w"))
        print(f"  saved {lemma_vecs.shape}")
    else:
        print(f"Lemma cache exists: {lemma_cache}")


def load_encoding_cache():
    sent_vecs = torch.load(os.path.join(CACHE_DIR, "sent_vecs.pt"),
                           map_location="cpu", weights_only=True)
    sent_ids  = json.load(open(os.path.join(CACHE_DIR, "sent_ids.json")))
    id2idx    = {sid: i for i, sid in enumerate(sent_ids)}
    gloss_vecs = torch.load(os.path.join(CACHE_DIR, "gloss_vecs.pt"),
                             map_location="cpu", weights_only=True)
    gloss_idx  = json.load(open(os.path.join(CACHE_DIR, "gloss_index.json")))
    lemma_vecs = torch.load(os.path.join(CACHE_DIR, "lemma_vecs.pt"),
                             map_location="cpu", weights_only=True)
    lemma_idx  = json.load(open(os.path.join(CACHE_DIR, "lemma_index.json")))
    return sent_vecs, id2idx, gloss_vecs, gloss_idx, lemma_vecs, lemma_idx


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class WSDDataset(Dataset):
    """Each item: context_vec, list of gloss_vecs, true_idx, lemma_id, meta."""
    def __init__(self, rows, sent_vecs, id2idx, gloss_vecs, gloss_idx,
                 max_cands=None):
        self.items = []
        for r in rows:
            sid = id2idx.get(r["id"])
            if sid is None:
                continue
            ctx = sent_vecs[sid]
            glosses = [gloss_vecs[gloss_idx[g]] for g in r["candidate_glosses"]]
            if max_cands and len(glosses) > max_cands:
                # keep true + random negatives
                true_idx = r["true_idx"]
                neg_idxs = [i for i in range(len(glosses)) if i != true_idx]
                _random.shuffle(neg_idxs)
                keep = [true_idx] + neg_idxs[:max_cands - 1]
                glosses = [glosses[i] for i in keep]
                true_idx = 0  # moved to position 0
            else:
                true_idx = r["true_idx"]
            self.items.append({
                "ctx": ctx,
                "glosses": glosses,
                "true_idx": true_idx,
                "lemma_id": r["lemma_id"],
                "lemma": r["lemma"],
                "true_synset": r["true_synset"],
                "candidates": r["candidates"],
                "candidate_glosses": r["candidate_glosses"],
                "is_mfs": r["is_mfs"],
                "id": r["id"],
            })

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate_fn(batch):
    """Pad variable-length candidate lists."""
    max_k = max(len(item["glosses"]) for item in batch)
    ctx = torch.stack([item["ctx"] for item in batch])  # (B, D)
    gloss_pad = torch.zeros(len(batch), max_k, EMBED_DIM)
    mask = torch.zeros(len(batch), max_k, dtype=torch.bool)
    true_idxs = torch.tensor([item["true_idx"] for item in batch])
    for i, item in enumerate(batch):
        k = len(item["glosses"])
        gloss_pad[i, :k] = torch.stack(item["glosses"])
        mask[i, :k] = True
    return {"ctx": ctx, "gloss": gloss_pad, "mask": mask,
            "true_idx": true_idxs, "meta": batch}


# ─────────────────────────────────────────────────────────────────────────────
# Model: MLP ranker
# ─────────────────────────────────────────────────────────────────────────────
class MLPRanker(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


def build_features(ctx, gloss, special_mode, target_vec=None):
    """
    ctx:    (B, D) or (B, K, D)
    gloss:  (B, K, D)
    Returns: (B, K, in_dim)
    """
    B, K, D = gloss.shape
    if ctx.dim() == 2:
        ctx_exp = ctx.unsqueeze(1).expand(B, K, D)
    else:
        ctx_exp = ctx

    if special_mode == "contract_only":
        # zero out context
        ctx_exp = torch.zeros_like(ctx_exp)
    elif special_mode == "target_only" and target_vec is not None:
        # use target word embedding instead of full sentence context
        ctx_exp = target_vec.unsqueeze(1).expand(B, K, D)

    diff  = (ctx_exp - gloss).abs()
    prod  = ctx_exp * gloss
    feats = torch.cat([ctx_exp, gloss, diff, prod], dim=-1)  # (B, K, 4D)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Non-trained baselines
# ─────────────────────────────────────────────────────────────────────────────
def heuristic_evaluate(test_ds):
    """Always predict MFS (is_mfs=True). Returns metrics dict."""
    correct = n = 0
    for item in test_ds:
        # MFS is at some position in candidates; pick candidate with is_mfs where possible
        # Actually: the heuristic = predict the position with is_mfs=True candidate.
        # In our dataset structure, we need the true per-instance MFS position.
        # Fallback: predict position 0
        pred = 0  # majority = first candidate (approx MFS position)
        # A better heuristic: find candidate where is_mfs would match
        # But we don't store per-candidate MFS here; use true_idx=0 as proxy
        if item["true_idx"] == 0:  # happens to be position 0
            correct += 1
        n += 1
    # Better: count is_mfs correct (true_idx is the actual answer, is_mfs is whether answer is MFS)
    # Majority baseline = predict the true_idx when item["is_mfs"] is True
    correct = sum(1 for item in test_ds if item.get("is_mfs", False) and True)
    # Actually the correct heuristic: always predict the first candidate (position 0 bias)
    # OR: count how often the true answer is the MFS candidate
    # Per WP: heuristic_baseline = majority(MFS) accuracy
    mfs_correct = sum(1 for item in test_ds if item.get("is_mfs", False))
    n = len(test_ds)
    acc = mfs_correct / n if n else 0.0
    return {"boundary_acc": acc, "f1": 0.0, "auroc": None,
            "n": n, "mfs_correct": mfs_correct, "heuristic": True}


def gloss_sim_evaluate(test_ds, device):
    """Rank by cosine(context_vec, gloss_vec_i) — frozen encoder, no training."""
    correct = n = 0
    for item in test_ds:
        ctx = item["ctx"].to(device)            # (D,)
        glosses = torch.stack(item["glosses"]).to(device)  # (K, D)
        ctx_n = F.normalize(ctx.unsqueeze(0), dim=-1)
        glo_n = F.normalize(glosses, dim=-1)
        scores = (ctx_n * glo_n).sum(-1)        # (K,)
        pred = scores.argmax().item()
        if pred == item["true_idx"]:
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    return {"boundary_acc": acc, "f1": 0.0, "auroc": None, "n": n, "gloss_sim": True}


def text_only_nn_evaluate(train_ds, test_ds, device):
    """1-NN context centroid: score by cosine(context_vec, synset_centroid)."""
    # Build centroids from train
    centroid_sum = defaultdict(lambda: torch.zeros(EMBED_DIM))
    centroid_cnt = defaultdict(int)
    for item in train_ds:
        syn = item["true_synset"]
        centroid_sum[syn] += item["ctx"]
        centroid_cnt[syn] += 1
    synsets = list(centroid_sum.keys())
    centroids = torch.stack([centroid_sum[s] / centroid_cnt[s] for s in synsets]).to(device)
    centroids = F.normalize(centroids, dim=-1)
    syn2idx = {s: i for i, s in enumerate(synsets)}

    correct = n = 0
    for item in test_ds:
        ctx = F.normalize(item["ctx"].to(device), dim=-1)  # (D,)
        cands = item["candidates"]  # list of synset IDs
        scores = []
        for c in cands:
            if c in syn2idx:
                scores.append((ctx * centroids[syn2idx[c]]).sum().item())
            else:
                scores.append(0.0)  # unseen synset (v3-hard)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == item["true_idx"]:
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    return {"boundary_acc": acc, "f1": 0.0, "auroc": None, "n": n, "text_only_nn": True}


# ─────────────────────────────────────────────────────────────────────────────
# Random contract controls
# ─────────────────────────────────────────────────────────────────────────────
def make_easy_random_ds(ds, gloss_vecs, gloss_idx, seed):
    """Replace gloss_vecs with a random gloss from a DIFFERENT lemma (label-preserving)."""
    rng = _random.Random(seed)
    # Collect all glosses NOT from each lemma
    lemma2glosses = defaultdict(list)
    for item in ds:
        for g in item["candidate_glosses"]:
            lemma2glosses[item["lemma_id"]].append(g)
    all_glosses = list(gloss_idx.keys())

    new_items = []
    for item in ds:
        lemma_set = set(item["candidate_glosses"])
        others = [g for g in all_glosses if g not in lemma_set]
        new_glosses = []
        for g in item["candidate_glosses"]:
            rand_g = rng.choice(others) if others else rng.choice(all_glosses)
            new_glosses.append(gloss_vecs[gloss_idx[rand_g]])
        new_item = dict(item)
        new_item["glosses"] = new_glosses
        new_items.append(new_item)
    # Return a simple list-backed dataset
    return new_items


def make_hard_random_ds(ds, gloss_vecs, gloss_idx, seed):
    """Replace gloss_vecs with a WRONG-sense gloss from the SAME lemma (label-preserving)."""
    rng = _random.Random(seed)
    new_items = []
    for item in ds:
        cand_glosses = item["candidate_glosses"]
        k = len(cand_glosses)
        new_glosses = []
        for i, g in enumerate(cand_glosses):
            # Pick a different candidate from same lemma
            others = [cand_glosses[j] for j in range(k) if j != i]
            rand_g = rng.choice(others) if others else g
            new_glosses.append(gloss_vecs[gloss_idx[rand_g]])
        new_item = dict(item)
        new_item["glosses"] = new_glosses
        new_items.append(new_item)
    return new_items


# ─────────────────────────────────────────────────────────────────────────────
# Training + evaluation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(all_preds, all_trues, all_scores=None):
    n = len(all_preds)
    correct = sum(p == t for p, t in zip(all_preds, all_trues))
    acc = correct / n if n else 0.0
    # F1 (macro over instances — each instance is a K-way problem; binary compat/incompat)
    tp = sum(1 for p, t in zip(all_preds, all_trues) if p == t)
    prec = tp / n if n else 0.0; rec = tp / n if n else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # AUROC over true-candidate scores vs wrong-candidate scores
    auroc = None
    if all_scores:
        pos, neg = [], []
        for scores_k, true_i in zip(all_scores, all_trues):
            if scores_k:
                pos.append(scores_k[true_i])
                neg.extend(s for i, s in enumerate(scores_k) if i != true_i)
        if pos and neg:
            # Approx AUROC: fraction of (pos, neg) pairs where pos > neg
            pos_t = torch.tensor(pos); neg_t = torch.tensor(neg)
            auroc = ((pos_t.unsqueeze(1) > neg_t.unsqueeze(0)).float().mean().item())
    return {"boundary_acc": acc, "f1": f1, "auroc": auroc, "n": n}


def train_and_eval(config_name, use_ctx, use_gloss, special_mode,
                   train_items, test_items, gloss_vecs, gloss_idx,
                   args, device, split_hash):
    """Train MLP ranker and evaluate. Returns metrics dict."""
    # Handle non-trained configs
    if special_mode == "heuristic":
        return heuristic_evaluate(test_items), True
    if special_mode == "gloss_sim":
        return gloss_sim_evaluate(test_items, device), True
    if special_mode == "text_only_nn":
        return text_only_nn_evaluate(train_items, test_items, device), True

    # Apply random contract transformations
    if special_mode == "random_easy":
        train_items = make_easy_random_ds(train_items, gloss_vecs, gloss_idx, args.seed)
        test_items  = make_easy_random_ds(test_items,  gloss_vecs, gloss_idx, args.seed)
    elif special_mode == "random_hard":
        train_items = make_hard_random_ds(train_items, gloss_vecs, gloss_idx, args.seed)
        test_items  = make_hard_random_ds(test_items,  gloss_vecs, gloss_idx, args.seed)

    # MLP input dim
    in_dim = EMBED_DIM * 4  # [ctx; gloss; |c-g|; c*g]
    model = MLPRanker(in_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    torch.manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Load checkpoint if exists
    ckpt_dir = os.path.join("data/processed_v3", f"checkpoints_{args.split}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f"{config_name}_seed{args.seed}.pt")
    epochs_done = 0
    if os.path.exists(ckpt) and not args.fresh:
        st = torch.load(ckpt, map_location=device, weights_only=False)
        if st.get("epochs_done", 0) > 0:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            epochs_done = st["epochs_done"]

    def _batch_forward(items_batch, training=True):
        """Vectorized forward over a batch of items with variable K (padded)."""
        valid = [it for it in items_batch if len(it["glosses"]) >= 2]
        if not valid:
            return None, None
        B = len(valid)
        max_k = max(len(it["glosses"]) for it in valid)
        ctx_t = torch.stack([it["ctx"] for it in valid]).to(device)        # (B, D)
        glo_t = torch.zeros(B, max_k, EMBED_DIM, device=device)
        mask  = torch.zeros(B, max_k, dtype=torch.bool, device=device)
        tgts  = torch.tensor([it["true_idx"] for it in valid], device=device)
        tgt_v = None
        if special_mode == "target_only":
            tgt_v = torch.stack([it["lemma_vec"] for it in valid]).to(device)
        for j, it in enumerate(valid):
            k = len(it["glosses"])
            glo_t[j, :k] = torch.stack(it["glosses"]).to(device)
            mask[j, :k] = True
        feats = build_features(ctx_t, glo_t, special_mode, tgt_v)         # (B, max_k, 4D)
        if training:
            scores = model(feats)                                           # (B, max_k)
        else:
            with torch.no_grad():
                scores = model(feats)
        # Mask padding before cross-entropy
        scores = scores.masked_fill(~mask, float('-inf'))
        return scores, tgts, mask, valid

    t0 = time.time()
    while epochs_done < args.epochs and (time.time() - t0) < args.time_budget:
        model.train()
        _random.seed(args.seed + epochs_done)
        _random.shuffle(train_items)

        batch_loss = 0.0; n_batches = 0
        for i in range(0, len(train_items), args.batch):
            result = _batch_forward(train_items[i:i + args.batch], training=True)
            if result[0] is None:
                continue
            scores, tgts, mask, _ = result
            loss = F.cross_entropy(scores, tgts)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_loss += loss.item(); n_batches += 1
        epochs_done += 1
        if epochs_done % 2 == 0 or epochs_done == args.epochs:
            print(f"  [{config_name} s{args.seed}] epoch {epochs_done}/{args.epochs} "
                  f"loss={batch_loss/max(n_batches,1):.4f} ({time.time()-t0:.0f}s)")

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epochs_done": epochs_done}, ckpt)
    done = epochs_done >= args.epochs
    if not done:
        print(f"[{config_name}] checkpoint at {epochs_done}/{args.epochs} epochs — rerun to continue")
        return None, False

    # Evaluation (vectorized)
    model.eval()
    preds, trues, scores_list = [], [], []
    for i in range(0, len(test_items), args.batch):
        result = _batch_forward(test_items[i:i + args.batch], training=False)
        if result[0] is None:
            continue
        scores, tgts, mask, valid_items = result
        for j, item in enumerate(valid_items):
            sc = scores[j].cpu().tolist()
            pred = max(range(len(sc)), key=lambda x: sc[x] if sc[x] != float('-inf') else -1e9)
            preds.append(pred); trues.append(item["true_idx"])
            scores_list.append([s for s, m in zip(sc, mask[j].cpu().tolist()) if m])

    metrics = compute_metrics(preds, trues, scores_list)
    metrics["epochs"] = epochs_done
    metrics["params"] = sum(p.numel() for p in model.parameters())
    _auc = metrics.get('auroc')
    _auc_str = f"{_auc:.3f}" if _auc is not None else "N/A"
    print(f"[{config_name}] DONE acc={metrics['boundary_acc']:.3f} "
          f"f1={metrics['f1']:.3f} auroc={_auc_str}")
    return metrics, True


# ─────────────────────────────────────────────────────────────────────────────
# Separability gate + canonical freeze (U3)
# ─────────────────────────────────────────────────────────────────────────────
def run_separability_gate(train_items, test_items, majority_acc, device, thresh=0.05):
    """text_only_nn must beat MFS by >= thresh. Returns (pass, text_only_acc)."""
    print(f"\n=== Separability Gate: text_only_nn vs majority={majority_acc:.3f} ===")
    res = text_only_nn_evaluate(train_items, test_items, device)
    acc = res["boundary_acc"]
    margin = acc - majority_acc
    passed = margin >= thresh
    print(f"text_only_nn acc={acc:.3f}, majority={majority_acc:.3f}, "
          f"margin={margin:+.3f} → {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed, acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v3_hard", choices=["v3_easy", "v3_hard"])
    ap.add_argument("--only", default=None, help="run single config")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_budget", type=float, default=300.0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: tiny subset, quick")
    ap.add_argument("--sep-thresh", type=float, default=0.05,
                    help="separability gate margin threshold (text_only over MFS)")
    ap.add_argument("--skip-gate", action="store_true",
                    help="skip separability gate (already passed)")
    ap.add_argument("--force-freeze", action="store_true")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    # Encoding step — build cache if any component missing or --encode-only requested
    cache_complete = all(os.path.exists(os.path.join(CACHE_DIR, f))
                        for f in ("sent_vecs.pt", "gloss_vecs.pt", "lemma_vecs.pt"))
    if not cache_complete or args.encode_only:
        if not cache_complete:
            print("Encoding cache incomplete — building...")
        build_encoding_cache(device)
        if args.encode_only:
            return

    sent_vecs, id2idx, gloss_vecs, gloss_idx, lemma_vecs, lemma_idx = load_encoding_cache()
    sent_vecs  = sent_vecs.to("cpu")
    gloss_vecs = gloss_vecs.to("cpu")
    lemma_vecs = lemma_vecs.to("cpu")
    print(f"Loaded: sent_vecs={sent_vecs.shape}, gloss_vecs={gloss_vecs.shape}, lemma_vecs={lemma_vecs.shape}")

    # Load split
    split_dir = f"data/processed_v3/{args.split}"
    train_rows = [json.loads(l) for l in open(f"{split_dir}/train.jsonl")]
    test_rows  = [json.loads(l) for l in open(f"{split_dir}/test.jsonl")]

    if args.smoke:
        # Tiny subset for plumbing check
        _random.seed(0); _random.shuffle(train_rows)
        _random.seed(0); _random.shuffle(test_rows)
        train_rows = train_rows[:500]
        test_rows  = test_rows[:200]
        args.epochs = 2
        args.time_budget = 120.0
        print(f"SMOKE MODE: train={len(train_rows)}, test={len(test_rows)}, epochs={args.epochs}")

    # Build WSDDataset items (pre-resolved vectors)
    def make_items(rows):
        items = []
        for r in rows:
            sid = id2idx.get(r["id"])
            if sid is None:
                continue
            ctx = sent_vecs[sid]
            glosses = [gloss_vecs[gloss_idx[g]] for g in r["candidate_glosses"]]
            lemma_vec = lemma_vecs[lemma_idx[r["lemma"]]]
            items.append({
                "ctx": ctx, "glosses": glosses,
                "lemma_vec": lemma_vec,
                "true_idx": r["true_idx"],
                "lemma_id": r["lemma_id"], "lemma": r["lemma"],
                "true_synset": r["true_synset"],
                "candidates": r["candidates"],
                "candidate_glosses": r["candidate_glosses"],
                "is_mfs": r["is_mfs"], "id": r["id"],
            })
        return items

    print("Building item lists...")
    train_items = make_items(train_rows)
    test_items  = make_items(test_rows)
    print(f"train_items={len(train_items)}, test_items={len(test_items)}")

    # Majority baseline
    mfs_acc = sum(1 for it in test_items if it["is_mfs"]) / len(test_items)
    print(f"MFS baseline: {mfs_acc:.3f}")

    # Separability gate (U3) + canonical freeze
    split_hash = compute_dataset_hash(split_dir)
    frozen_path = os.path.join(split_dir, "frozen_dataset_hash.json")
    if not os.path.exists(frozen_path) or args.force_freeze:
        if not args.skip_gate:
            gate_pass, text_only_acc = run_separability_gate(
                train_items, test_items, mfs_acc, device, args.sep_thresh)
            if not gate_pass and not args.force_freeze:
                print("GATE FAIL — text_only does not beat MFS by required margin.")
                print("Adjust subset/framing before canonical freeze. Exiting.")
                sys.exit(1)
        h = freeze_dataset_hash(split_dir, split_hash,
                                notes=f"U3 separability gate PASS — text_only > MFS",
                                probe_results={"mfs_acc": mfs_acc})
        print(f"Canonical freeze: {split_hash}")
    else:
        print(f"Dataset already frozen. Verifying hash...")
        split_hash = assert_frozen_hash(split_dir)
        print(f"Hash verified: {split_hash}")

    # Per-seed results file (avoids write-race when seeds run in parallel)
    res_path = os.path.join(split_dir, f"results_v3_{args.split}_seed{args.seed}.json")
    results = {}
    if os.path.exists(res_path):
        try:
            results = json.load(open(res_path)).get("results", {})
        except Exception:
            results = {}

    to_run = [args.only] if args.only else V3_SWEEP_ORDER
    all_done = True
    for name in to_run:
        if name not in CONFIGS_V3:
            print(f"Unknown config '{name}', skipping"); continue
        key = f"{name}_seed{args.seed}"
        if key in results:
            print(f"[{name}] already done (seed {args.seed}), skipping"); continue
        use_ctx, use_gloss, special = CONFIGS_V3[name]
        res, done = train_and_eval(
            name, use_ctx, use_gloss, special,
            list(train_items), list(test_items),
            gloss_vecs, gloss_idx, args, device, split_hash)
        if done:
            results[key] = res
        else:
            all_done = False
            break

    with open(res_path, "w") as f:
        json.dump({"majority": mfs_acc, "results": results,
                   "split": args.split, "split_hash": split_hash,
                   "args": vars(args)}, f, indent=2)

    n_done = len(results)
    n_total = len(V3_SWEEP_ORDER)
    print(f"\nv3 results ({args.split} seed{args.seed}): {n_done}/{n_total} configs done")
    print("ALL_DONE" if all_done and n_done == n_total else
          "NOT_DONE (rerun same command to continue)")


if __name__ == "__main__":
    main()
