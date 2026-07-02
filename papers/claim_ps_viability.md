# Bounded Claim: PS (Possibility Score) Viability — WP-ST-18

**WP:** WP-ST-18 | **Project:** gem2-cbt | **Status:** DRAFT (CW read pending)
**Teacher:** deepseek-chat (temperature 0) | **Date:** 2026-07-01

---

## Verdict

**Outcome B (mechanical, from pre-registered thresholds):**
PS is **viable** as a level-detector (source-orthogonal, negative-control clean)
— but the **density+distance geometry is cosmetic**, because density and distance
are strongly collinear in this feature family (corr ≈ −0.92) and combining them
adds no margin over `density_only`.

**Adoption:** use `density_only` (Laplace-smoothed unigram class-conditional
log-likelihood, source-residualized) as the drop-filter. Do not use PS as
`density + distance`; the second term buys no information.

---

## Numbers (adopted, source-residualized pipeline)

Passage-disjoint 5-fold `GroupKFold` on 600 records = 200 passages × 3 levels;
feature hash `418e85eea4800d39`; pool freeze-hash `21603578fe641bfc`.

| Scorer | test macro-recall | fold_std (recall) | test macro-AUC | train macro-recall |
|---|---|---|---|---|
| chance floor | 0.3333 | — | 0.5000 | 0.3333 |
| 2feat_logreg (level-agnostic) | 0.7050 | 0.0239 | 0.8679 | 0.7121 |
| distance_only | 0.7300 | 0.0410 | 0.8848 | 0.7371 |
| density_only | **0.7617** | 0.0272 | 0.9193 | 0.7721 |
| PS (density + distance, 6-dim) | **0.7650** | 0.0260 | 0.9205 | 0.7733 |

- **PS over density_only:** +0.0033 test recall AND +0.0012 test AUC — both ≪ the pre-registered
  0.05 margin → geometry cosmetic.
- **PS over distance_only:** +0.035 test recall, +0.036 test AUC (both < 0.05).
- **All 4 scorers ≥ chance + 0.05** (0.383) → level is detectable.
- **PS ≈ density_only at every metric:** test recall (0.7650 vs 0.7617, Δ = +0.0033),
  test AUC (0.9205 vs 0.9193, Δ = +0.0012). The Outcome B verdict is robust to the
  recall-vs-AUC choice.
- **0.76 macro-recall is moderate, not high** — a `density_only` drop-filter misroutes
  roughly 1 in 4 prompts. Downstream C2NS synthesizer must be tolerant of that error
  rate, and abstain/replay paths should exist for the mis-routed cases. This is not a
  ceiling on the primitive; it is the honest number on this pool with these features.

---

## Guards — ALL PASS (adopted pipeline)

| Guard | Result | Value |
|---|---|---|
| Negative control (shuffled level labels → chance) | ✓ leak-free | density 0.352 / distance 0.340 / PS 0.353 / 2feat 0.340 (ceiling 0.45, chance 0.333) |
| D1 diversity (per-level top-opening 5-gram frac ≤ 0.30) | ✓ | task 0.985 / concept 0.870 / context 0.775 |
| D2 residual max per-token level-separation ≤ 0.30 | ✓ | 0.275 after 2 iterations, 23 tokens masked |
| D2 strip non-degenerate (survivable tokens remain) | ✓ | median 15 survivors, min 4, zero-empty framings 0 |
| fold_std > 0 (all scorers) | ✓ | min 0.024 across all scorers |
| Passage-disjoint train/test (all 3 framings of a passage on same side) | ✓ | 160 train + 40 test passages, ∅ intersection |
| Features from QUERY, not passage | ✓ | TF-IDF fit on TRAIN stripped queries only; passage never entered the vectorizer |
| Gate B — source macro-AUC ≤ 0.55 | ✓ | 0.523 (fold AUCs [0.528, 0.526, 0.516, 0.539, 0.507]) |

---

## Honest caveats (mandatory — do not remove)

### (a) Source-orthogonal viability is established only after deconfounding

Source-orthogonal PS viability is established only after post-hoc deconfounding
of the query features. The higher raw detection numbers (raw density_only test
recall ≈ 0.987, test AUC ≈ 1.000 — see `bakeoff_raw.json`) were **partly a
source artifact**: regressing each of the 6 (density, distance) features on the
source-classifier proba gave `R² = [0.891, 0.159, 0.683, 0.827, 0.032, 0.695]`
— **four of six feature dimensions had source-explained variance ≥ 0.68 on
TRAIN**.

The pool enforces **source ⊥ level for passages** (50 records per source × level
cell, 12 cells) — but the four sources (SQuAD, WSD, PAWS, MNLI) have
**systematically different query vocabulary** (SQuAD entities, PAWS convict-voyage
tokens, WSD lemma-marked sentences, MNLI genre markers), because DeepSeek
generated queries FROM the passages. So while source ⊥ level holds for
**passages** by construction, it does **NOT** hold for **query features**.

On raw features, Gate B failed at source macro-AUC = 0.576 > 0.55 pre-reg
ceiling — the confound-detector caught it. Residualization
(fit source-classifier on TRAIN 6-dim features; regress each feature on the
source proba; use the residuals) dropped source macro-AUC to 0.523. But
this is a **post-hoc statistical fix**, not a by-construction property. If the
pool were rebuilt with per-source query-vocabulary de-biasing at generation time,
residualization might not be needed — see "future refinements".

### (b) "Geometry cosmetic" = redundant, collinear features

Per-level `corr(density_L, distance_L)` on the adopted pipeline:
`task = −0.907, concept = −0.948, context = −0.926`. Density (mass:
class-conditional log-likelihood) and distance (geometry: 1 − cos to centroid)
carry **nearly the same information up to a sign**. PS = density + distance is
therefore a **redundant** combination — combining two collinear features
recovers ≈ one feature's worth of signal. This matches the U5 numbers exactly:
PS test_recall (0.765) ties density_only test_recall (0.762) at the 3-decimal
digit.

The **HPIC precedent** applies (WP-ST-5 / claim_hpic_cer.md): a proposed complex
scorer that reduces to a linear combination of its parts is cosmetic. Adopt the
simplest component, not the compound.

### (c) Bounds — this claim does not extend beyond

- **Pool:** 200 passages × 3 framings = 600 records, teacher = deepseek-chat
  (single teacher; temperature 0; pool freeze `21603578fe641bfc`).
- **Sources:** {SQuAD, WSD, PAWS, MNLI}; 50 records per source per level.
- **Features:** TF-IDF unigrams+bigrams on U2-stripped queries, class-conditional
  Laplace unigram LM for density, cosine to L2-normalized centroids for distance,
  post-hoc source-residualization. Feature hash `418e85eea4800d39`.
- **k = 5** folds passage-disjoint GroupKFold; **seed = 42**; **test frac = 0.20**.
- **Level partition:** {task, concept, context}. Task ⊢ / Concept ⊢ / Context ⊨
  from WP-16 U4 (verified priors; not re-litigated).
- **Verdict robust to metric choice:** Outcome B holds on both macro-recall
  (density_only 0.7617 vs PS 0.7650, Δ = +0.0033) and macro-AUC (density_only
  0.9193 vs PS 0.9205, Δ = +0.0012). Neither reaches the pre-registered 0.05
  margin.

**Not claimed:** cross-teacher, cross-pool, cross-language, decode-time behavior,
finetuned model behavior. This is a **prompt-level detection** experiment on
one DeepSeek-generated pool.

### (d) The discipline worked — do NOT overstate to "PS works cleanly"

The auto-gate at U6 **caught** the source confound (Gate B fail at 0.576) and
triggered auto-HALT per protocol; the negative-control at U7 **validated** the
residualization fix (shuffled labels → chance across all scorers). The
guard-set is doing exactly what it was designed to do — catching leaks the
first pass missed.

Do not read this claim as "PS works cleanly." Read it as: **PS is viable
after a documented post-hoc deconfound, and even after that fix, the
geometry-over-parts margin does not exist. Use density_only.**

---

## Optional future refinements (not required to accept Outcome B)

1. **De-bias queries at generation time.** Rebuild the pool with an
   additional constraint on the DeepSeek meta-prompt: instruct the teacher to
   avoid source-identifying content vocabulary (e.g., strip named entities /
   corpus-specific idioms before framing generation). Would remove the need for
   post-hoc residualization. **Prediction:** Outcome B would very likely
   reproduce — density and distance are collinear by their definitions, not by
   a source artifact.
2. **Test PS on a different feature family.** Local frozen encoder embeddings
   (e.g., a small offline sentence transformer) may yield less-collinear
   density and distance. WP-18 U8 ("embedding-feature retry, only if Outcome C")
   was aborted because the verdict was B, not C. If the community disputes the
   "cosmetic geometry" finding on TF-IDF features, U8 could be re-opened as a
   separate WP.

Neither is required to accept the Outcome B verdict.

---

## Files

- Pool: `data/ece_shared_pool_v2/items.jsonl` (600), `items_sorted.jsonl`,
  `frozen_dataset_hash.json` (`21603578fe641bfc`), `manifest.json`.
- Audit: `data/ece_shared_pool_v2/stripped.jsonl`, `audit.json`, `audit_gate.json`.
- Features: `features_r.jsonl` (adopted, hash `418e85eea4800d39`);
  `features.jsonl` (raw audit, hash `63c07a67f662b623`);
  `features_manifest.json`.
- Results: `bakeoff.json` (adopted), `bakeoff_raw.json` (audit),
  `source.json` (adopted, macro-AUC 0.523),
  `source_raw.json` (audit, macro-AUC 0.576),
  `neg_control.json` (leak-free),
  `verdict.json` (Outcome B mechanical),
  `u6_auto_halt.json` (superseded pre-residualization halt trigger),
  `u6_post_residualization.json` (retraction record + guard checklist),
  `corr.json` (per-level density/distance correlation).
- Scripts: `scripts/build_pool_v2.py`, `scripts/pool_v2_audit.py`,
  `scripts/ps_bakeoff_v2.py`.

## References

- WP-ST-16 (superseded) — Task ⊢ / Concept ⊢ / Context ⊨ verified;
  U6 featurization bug that motivated this WP.
- WP-ST-17 (retired) — processed_v2 saturation; topic-matched-negatives +
  source-detector-at-chance requirement folded into WP-18 protocol.
- WP-ST-5 / claim_hpic_cer.md — HPIC precedent: complex-scorer-reducible-to-parts
  = cosmetic. Same finding pattern here.
- WP-ST-15 / claim_payoff_squad.md — first red-team-surviving contract-conditioning
  payoff (18%→3% on source-silent questions) — the payoff PS is a filter for.
- (internal research-log reference redacted for public release)
