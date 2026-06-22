"""WP-ST-3 U1 — build the gloss-informed sense-classification dataset from SemCor/WordNet.

Framing (NOT raw gloss-matching): each instance is a sentence with one sense-tagged
target word. text_only must predict the true synset among the lemma's candidate synsets
(context-only WSD). The contract channel (downstream) = the candidate-gloss vectors.

Record (JSONL):
  { id, sentence, lemma, pos, lemma_id, target,
    true_synset, candidates:[synset names], candidate_glosses:[...], true_idx }

Constraints (CONTRACT):
- CEILING PRE-EMPTION: keep only POLYSEMOUS lemmas with >= MIN_SENSES synsets (same POS);
  cap most-frequent-sense (MFS, rank-0 synset) dominance per lemma to <= MFS_CAP fraction.
- Negative sampling is derivable downstream: HARD neg = other candidate (same lemma);
  EASY neg = gloss from a different lemma. (Stored implicitly via candidates list.)

Resumable by sentence range (--start/--count) since SemCor is large; appends to JSONL.
Run full:   python scripts/build_wsd_dataset.py --all
"""
import argparse, json, os, random
import nltk
from nltk.corpus import semcor
from nltk.corpus.reader.wordnet import Lemma
from nltk.corpus import wordnet as wn

OUT_DIR = "data/processed_v3"
RAW = os.path.join(OUT_DIR, "wsd_instances.jsonl")


def _flat(x):
    """Flatten arbitrarily nested lists/strings into a list of str tokens."""
    out = []
    if isinstance(x, str):
        out.append(x)
    elif hasattr(x, "leaves"):
        for leaf in x.leaves():
            out.extend(_flat(leaf))
    elif isinstance(x, (list, tuple)):
        for e in x:
            out.extend(_flat(e))
    else:
        out.append(str(x))
    return out


def iter_instances(start, count):
    """Yield (sentence_text, lemma, pos, true_synset_name, target_surface) per sense-tagged token."""
    sents = semcor.tagged_sents(tag="sem")
    n = len(sents)
    end = n if count < 0 else min(n, start + count)
    for si in range(start, end):
        chunks = sents[si]
        sentence = " ".join(t for ch in chunks for t in _flat(ch))
        for ch in chunks:
            lab = ch.label() if hasattr(ch, "label") else None
            if not isinstance(lab, Lemma):
                continue
            try:
                syn = lab.synset()
            except Exception:
                continue
            lemma = syn.lemmas()[0].name()
            pos = syn.pos()
            surface = " ".join(_flat(ch))
            yield si, sentence, lemma.lower(), pos, syn.name(), surface


def build(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(args.seed)
    mode = "w" if (args.start == 0 or not os.path.exists(RAW)) else "a"
    kept = 0
    seen_lemmas = set()
    with open(RAW, mode) as f:
        for si, sentence, lemma, pos, true_syn, surface in iter_instances(args.start, args.count):
            cands = wn.synsets(lemma, pos=pos)
            if len(cands) < args.min_senses:
                continue
            cand_names = [c.name() for c in cands]
            if true_syn not in cand_names:
                continue
            rec = {
                "id": f"s{si}_{lemma}_{len(cand_names)}_{kept}",
                "sentence": sentence,
                "lemma": lemma,
                "pos": pos,
                "lemma_id": f"{lemma}.{pos}",
                "target": surface,
                "true_synset": true_syn,
                "candidates": cand_names,
                "candidate_glosses": [c.definition() for c in cands],
                "true_idx": cand_names.index(true_syn),
                "is_mfs": cand_names.index(true_syn) == 0,
            }
            f.write(json.dumps(rec) + "\n")
            kept += 1
            seen_lemmas.add(lemma)
    print(f"[build] start={args.start} count={args.count} kept={kept} lemmas={len(seen_lemmas)} -> {RAW}")


def cap_and_stats(args):
    """Second pass: cap MFS dominance per lemma, write capped file + stats."""
    rows = [json.loads(l) for l in open(RAW)]
    by_lemma = {}
    for r in rows:
        by_lemma.setdefault(r["lemma_id"], []).append(r)
    rng = random.Random(args.seed)
    capped = []
    for lid, rs in by_lemma.items():
        mfs = [r for r in rs if r["is_mfs"]]
        non = [r for r in rs if not r["is_mfs"]]
        # cap MFS to at most MFS_CAP fraction of the kept set for this lemma
        if non:
            max_mfs = int(len(non) * args.mfs_cap / max(1e-9, (1 - args.mfs_cap)))
            rng.shuffle(mfs)
            mfs = mfs[:max_mfs]
        capped.extend(mfs + non)
    rng.shuffle(capped)
    out = os.path.join(OUT_DIR, "wsd_capped.jsonl")
    with open(out, "w") as f:
        for r in capped:
            f.write(json.dumps(r) + "\n")
    # stats
    from collections import Counter
    n = len(capped)
    mfs_rate = sum(r["is_mfs"] for r in capped) / max(1, n)
    cand_sizes = Counter(len(r["candidates"]) for r in capped)
    lemmas = set(r["lemma_id"] for r in capped)
    # majority baseline if always predict MFS (idx 0): = mfs_rate
    print(f"[cap] instances={n} lemmas={len(lemmas)} MFS_rate(majority-if-predict-sense0)={mfs_rate:.3f}")
    print(f"[cap] candidate-set sizes (top): {dict(sorted(cand_sizes.items())[:8])}")
    print(f"[cap] avg candidates/instance = {sum(len(r['candidates']) for r in capped)/max(1,n):.2f}")
    print(f"[cap] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=-1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min_senses", type=int, default=3)
    ap.add_argument("--mfs_cap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", action="store_true", help="run MFS-cap + stats pass over wsd_instances.jsonl")
    args = ap.parse_args()
    if args.all:
        args.start, args.count = 0, -1
    if args.cap:
        cap_and_stats(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
