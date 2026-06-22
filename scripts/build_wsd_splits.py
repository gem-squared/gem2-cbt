"""WP-ST-3 U2 — build v3-easy (sentence split) + v3-hard (lemma split), provisional freeze + stats.

Task framing is LOCKED to candidate gloss SCORING/RANKING per instance (model scores the
instance's own candidate glosses, picks best) — NOT global synset classification. Therefore
the majority baseline = always predict the most-frequent sense (candidate idx 0) = is_mfs rate.

CPU-only. NO encoder, NO canonical freeze (deferred to U3 separability gate).
"""
import json, os, hashlib, random
from collections import Counter

SRC = "data/processed_v3/wsd_capped.jsonl"
OUT = "data/processed_v3"


def load():
    return [json.loads(l) for l in open(SRC)]


def sent_id(rec):
    # id format: s{si}_{lemma}_{ncand}_{kept}
    return rec["id"].split("_", 1)[0]


def split_by_key(rows, keyfn, test_frac, seed):
    keys = sorted({keyfn(r) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_test = int(len(keys) * test_frac)
    test_keys = set(keys[:n_test])
    train = [r for r in rows if keyfn(r) not in test_keys]
    test = [r for r in rows if keyfn(r) in test_keys]
    return train, test


def hash_split(train, test):
    h = hashlib.sha256()
    for part in (train, test):
        for r in part:
            h.update(r["id"].encode())
    return h.hexdigest()[:16]


def stats(name, train, test, keyfn, leak_keyfn):
    # leakage
    tr_keys = {leak_keyfn(r) for r in train}
    te_keys = {leak_keyfn(r) for r in test}
    leak = tr_keys & te_keys
    def mfs_rate(rows): return sum(r["is_mfs"] for r in rows) / max(1, len(rows))
    def candstats(rows):
        cs = [len(r["candidates"]) for r in rows]
        return (sum(cs)/max(1,len(cs)), min(cs), max(cs))
    avg, mn, mx = candstats(test)
    lemmas_tr = {r["lemma_id"] for r in train}
    lemmas_te = {r["lemma_id"] for r in test}
    return {
        "split": name,
        "train_n": len(train), "test_n": len(test),
        "leakage_keys": len(leak),
        "majority_MFS_baseline_test": round(mfs_rate(test), 4),
        "majority_MFS_baseline_train": round(mfs_rate(train), 4),
        "test_avg_candidates": round(avg, 2), "test_min_candidates": mn, "test_max_candidates": mx,
        "train_lemmas": len(lemmas_tr), "test_lemmas": len(lemmas_te),
        "lemma_overlap": len(lemmas_tr & lemmas_te),
    }


def write_split(name, train, test):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    for part, rows in [("train", train), ("test", test)]:
        with open(os.path.join(d, f"{part}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def main():
    rows = load()
    print(f"loaded {len(rows)} instances")
    # v3-easy: sentence split (no sentence in both)
    e_tr, e_te = split_by_key(rows, sent_id, test_frac=0.15, seed=0)
    # v3-hard: lemma split (no lemma_id in both)
    h_tr, h_te = split_by_key(rows, lambda r: r["lemma_id"], test_frac=0.15, seed=0)
    write_split("v3_easy", e_tr, e_te)
    write_split("v3_hard", h_tr, h_te)
    s_easy = stats("v3_easy", e_tr, e_te, sent_id, sent_id)
    s_hard = stats("v3_hard", h_tr, h_te, lambda r: r["lemma_id"], lambda r: r["lemma_id"])
    manifest = {
        "source": SRC, "task_framing": "candidate_gloss_ranking (NOT global classification)",
        "freeze": "PROVISIONAL (canonical freeze deferred to U3 separability gate)",
        "v3_easy": {**s_easy, "provisional_hash": hash_split(e_tr, e_te)},
        "v3_hard": {**s_hard, "provisional_hash": hash_split(h_tr, h_te)},
    }
    with open(os.path.join(OUT, "manifest_provisional.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    # split_stats.md
    os.makedirs("papers", exist_ok=True)
    with open("papers/split_stats.md", "w") as f:
        f.write("# WP-ST-3 U2 — split stats (provisional)\n\n")
        f.write("_Task framing: candidate gloss SCORING/RANKING per instance (not global "
                "classification). Majority baseline = predict most-frequent sense (idx 0) = MFS rate._\n\n")
        f.write("| split | train_n | test_n | majority(MFS) test | avg cands | train lemmas | test lemmas | lemma overlap | leakage |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for s in (s_easy, s_hard):
            f.write("| {split} | {train_n} | {test_n} | {majority_MFS_baseline_test} | "
                    "{test_avg_candidates} | {train_lemmas} | {test_lemmas} | {lemma_overlap} | "
                    "{leakage_keys} |\n".format(**s))
        f.write("\n**Provisional hashes:** v3_easy `{}` | v3_hard `{}`\n".format(
            manifest["v3_easy"]["provisional_hash"], manifest["v3_hard"]["provisional_hash"]))
        f.write("\n**Read:** v3-hard (lemma split) is the generalization test the claim weights. "
                "text_only must beat majority(MFS) — measured with the encoder in U3 (NOT here). "
                "Canonical freeze happens in U3 only if that gate passes.\n")
    print("V3_EASY:", s_easy)
    print("V3_HARD:", s_hard)
    print("wrote", os.path.join(OUT, "manifest_provisional.json"), "+ papers/split_stats.md")


if __name__ == "__main__":
    main()
