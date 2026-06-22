# CBT v3 Claim — Ecological Validity: Gloss Channel in Zero-Shot WSD

**Experiment:** WP-ST-3 | **Date:** 2026-06-19 | **Status:** VERIFIED
**v3-hard frozen hash:** `25fc21e245581f64` | **v3-easy frozen hash:** `fba8e3f6236ae5a8`
**Seeds:** 0–9 | **Epochs:** 5 | **Encoder:** all-MiniLM-L6-v2 (384-dim, frozen)

---

## Verdict

**v3-hard g3b: PASS — correct contract content (same-lemma sense distinction) is load-bearing
in zero-shot WSD (unseen lemmas). v3-easy g6: FAIL — trained MLP does not beat the
strong 1-NN context centroid on seen lemmas.**

**CBT-v1 gate: NOT PASSED** (g6 condition not met). CBT-v1 architecture remains gated.

---

## Decisive Evidence

### v3-hard g3b: cbt > hard_same_lemma_random (DECISIVE for zero-shot)

| Metric | Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|---|
| boundary_acc | 10/10 | +0.125 | 4.08 | PASS ✓ |

`hard_same_lemma_random` receives a WRONG-sense gloss from the SAME lemma — the same
candidate pool, shuffled. `cbt` (0.332) consistently outperforms it (0.206) across all
10 seeds. The correct sense gloss is the active variable; everything else is held fixed.

This is the ecological analogue of WP-ST-2 g3: correct contract **content** is
load-bearing, not just "having a gloss slot". This holds on REAL natural language with
UNSEEN lemmas (zero-shot).

### v3-hard g3a: cbt > easy_random_contract (sanity floor)

| Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|
| 10/10 | +0.190 | 6.07 | PASS ✓ |

Gloss from a DIFFERENT lemma entirely collapses performance to near-random (0.141).
Correct-content advantage confirmed even against maximally wrong contract.

### v3-hard parsimony: cbt > gloss_similarity_baseline

| Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|
| 10/10 | +0.086 | 2.79 | PASS ✓ |

`gloss_similarity_baseline` ranks by raw cosine(context_vec, gloss_vec) without a
trained head. `cbt` (0.332) beats it (0.245) consistently — the MLP head learns a
meaningful interaction beyond frozen cosine similarity. The trained interaction is
load-bearing, not just the encoder's proximity.

### v3-hard parsimony: cbt > target_word_only

| Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|
| 10/10 | +0.106 | 1.56 | PASS ✓ |

`target_word_only` replaces the full sentence context with just the lemma embedding.
`cbt` (0.332) beats it (0.225) — sentence context is necessary, not just the target
word identity + gloss. The full context contributes.

---

## Null Result: v3-easy g6 FAIL

### v3-easy g6: cbt vs text_only_nn (seen lemmas)

| Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|
| 0/10 | −0.027 | −11.01 | FAIL ✗ |

`text_only_nn` (1-NN context centroid, 0.603) outperforms `cbt` (0.576) on v3-easy
(seen lemmas, sentence split). This is interpretable, not a confound:

- **text_only_nn is non-parametric**: it uses the full training-set synset centroids at
  inference time (nearest-neighbor lookup). For seen lemmas, well-calibrated prototypes
  exist — 1-NN is a strong baseline.
- **cbt is parametric (5 epochs)**: the MLP compresses training signal into 918k
  parameters and doesn't fully recover the 1-NN oracle on seen lemmas within 5 epochs.
- **contract_only = 0.508 ≈ MFS = 0.533**: the gloss channel alone carries ~97% of MFS
  accuracy without any sentence context on v3-easy. The gloss is highly informative
  for seen lemmas — context adds comparatively little on top.
- **target_word_only = 0.578 ≈ cbt = 0.576**: on v3-easy, lemma identity + gloss
  ≈ full sentence + gloss. Sentence context provides minimal marginal value when
  lemmas are well-represented by training prototypes.

**Interpretation**: on seen lemmas, the task reduces largely to gloss-matching (the
gloss alone disambiguates reliably), and 1-NN context lookup provides a ceiling the
5-epoch MLP doesn't reach. This is NOT a failure of the gloss channel — g3b still
passes (correct content beats random) — but it means the MLP's context–gloss
interaction doesn't add over the non-parametric context baseline on this split.

**v3-easy g3b PASS** (cbt=0.576 > hard_same_lemma=0.292, 10/10, Δ+0.284, d=65.72)
confirms correct content is still load-bearing even on seen lemmas.

---

## Scope and Boundaries (EEF: ⊢ grounded claims only)

- ⊢ **Claim**: a separately-encoded contract(gloss) vector channel, trained with a
  shallow MLP, outperforms random-content controls on zero-shot WSD (unseen lemmas,
  v3-hard). Correct sense-disambiguating content is load-bearing.

- ⊢ **Does NOT claim**: gloss channel consistently outperforms strong non-parametric
  context baselines on seen lemmas. The v3-easy g6 null result is part of the record.

- ⊢ **Encoder caveat**: uses a frozen small sentence encoder (MiniLM-L6, 384-dim),
  not a fine-tuned SOTA WSD model. Results are about the gloss-injection mechanism,
  not encoder quality.

- ⊢ **Prior art**: gloss-informed WSD (GlossBERT, BEM, ConSeC) is well-established.
  This is a clean ablation of the injection mechanism, not a new method.

- ⊢ **Task scope**: classification/ranking probe on SemCor WSD. Does NOT transfer to
  text generation, hallucination detection, instruction-following, or equivariance.

- ⊢ **Zero-shot scope**: v3-hard zero-shot property arises from the lemma split
  construction (no overlap). The claim is about this structural zero-shot regime.

---

## CBT-v1 Gate Decision

**Gate condition (per WP-ST-3 U8):**
  (v3-hard g3b PASS) ∧ (v3-easy g6 PASS with effect size) ∧ (cbt > gloss_sim on v3-hard)

| Condition | Result |
|---|---|
| v3-hard g3b PASS | ✓ PASS |
| v3-easy g6 PASS | ✗ FAIL (Δ-0.027, 0/10) |
| cbt > gloss_sim (v3-hard) | ✓ PASS |

**Gate verdict: NOT PASSED.** CBT-v1 (boundary-gated attention) remains GATED.

**Rationale**: v3-easy g6 failure means the gloss channel does not consistently improve
over a strong context-only baseline on seen lemmas within 5 training epochs. Before
investing in the more complex CBT-v1 architecture, either (a) demonstrate g6 on v3-easy
with more training / larger model, or (b) accept that the gloss channel's value is
specifically in the zero-shot regime (v3-hard g3b) and redesign the ecological test
accordingly — then revisit the v1 gate.

**What IS justified by these results:**
- Further investigation of the zero-shot regime (v3-hard) with stronger encoders
- Understanding why the MLP underperforms 1-NN on v3-easy (underfitting vs. task structure)
- A revised ecological test that specifically targets zero-shot sense disambiguation

---

## Relationship to Prior Claims

| Claim | Result |
|---|---|
| claim_v0.md (WP-ST-1) | CONFOUNDED — shuffled label invalid, level/capacity not isolated |
| claim_v2.md (WP-ST-2) | PASS — contract content isolated on synthetic data |
| **claim_v3.md (WP-ST-3)** | **PARTIAL** — g3b PASS on zero-shot NL; g6 FAIL on seen-lemma NL |

The chain: synthetic mechanism isolation (v2) → real NL zero-shot confirmation (v3-hard
g3b) → but MLP underperforms 1-NN on seen lemmas (v3-easy g6 null result).

---

*WP-ST-3 | v3-hard hash 25fc21e245581f64 | v3-easy hash fba8e3f6236ae5a8 | 10 seeds
EEF: ⊢ grounded on real-NL WSD (SemCor/WordNet). CBT-v1 GATED — g6 not met.*
