# WP-ST-5: HPIC-CER Routing Test — Results

**Seeds:** 0–9 (10/10 complete)  
**Domain:** 5-route ECE science micro-domain  
**Regimes:** CLEAR / MULTI / CONFLICT  
**Metrics:** oracle-threshold (no threshold tuning confound)  

**PROVED PREMISE:** Re(Z_j)=signed_strength_j; Im(Z_j)=evidence_spread_j.  
Complex z is an invertible map of 2 real features → HPIC adds no expressivity.  

---
## Per-Router Summary (mean±std, 10 seeds)

| Router | Recall@1 | Recall@2 | Abstain-F1 | Conflict-Abstain-Recall |
|---|---|---|---|---|
| keyword | 0.642±0.000 | 0.775±0.000 | 0.508±0.000 | 0.000±0.000 |
| tfidf | 0.637±0.000 | 0.797±0.000 | 0.518±0.000 | 0.100±0.000 |
| softmax | 0.677±0.000 | 0.828±0.000 | 0.527±0.000 | 0.600±0.000 |
| twofeature | 0.642±0.000 | 0.790±0.000 | 0.563±0.000 | 0.290±0.000 |
| hpic_cer | 0.642±0.000 | 0.790±0.000 | 0.563±0.000 | 0.290±0.000 |

---
## gate_route: conflict-aware router > baselines

| Comparison | Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|---|
| twofeature_vs_keyword_abstain_f1 | 10/10 | +0.0558 | 55766068.49 | PASS ✓ |
| twofeature_vs_keyword_recall@1 | 0/10 | +0.0000 | 0.00 | FAIL ✗ |
| twofeature_vs_tfidf_abstain_f1 | 10/10 | +0.0452 | 45245566.66 | PASS ✓ |
| twofeature_vs_tfidf_recall@1 | 10/10 | +0.0050 | 5000000.00 | FAIL ✗ |
| twofeature_vs_softmax_abstain_f1 | 10/10 | +0.0362 | 36206368.65 | PASS ✓ |
| twofeature_vs_softmax_recall@1 | 0/10 | -0.0350 | -35000000.00 | FAIL ✗ |
| hpic_cer_vs_keyword_abstain_f1 | 10/10 | +0.0558 | 55766068.49 | PASS ✓ |
| hpic_cer_vs_keyword_recall@1 | 0/10 | +0.0000 | 0.00 | FAIL ✗ |
| hpic_cer_vs_tfidf_abstain_f1 | 10/10 | +0.0452 | 45245566.66 | PASS ✓ |
| hpic_cer_vs_tfidf_recall@1 | 10/10 | +0.0050 | 5000000.00 | FAIL ✗ |
| hpic_cer_vs_softmax_abstain_f1 | 10/10 | +0.0362 | 36206368.65 | PASS ✓ |
| hpic_cer_vs_softmax_recall@1 | 0/10 | -0.0350 | -35000000.00 | FAIL ✗ |

**gate_route: PASS ✓**

---
## gate_complex: hpic_cer > twofeature

| Comparison | Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|---|
| hpic_cer_vs_twofeature_abstain_f1 | 0/10 | +0.0000 | 0.00 | FAIL ✗ |
| hpic_cer_vs_twofeature_recall@1 | 0/10 | +0.0000 | 0.00 | FAIL ✗ |
| hpic_cer_vs_twofeature_conflict_abstain_recall | 0/10 | +0.0000 | 0.00 | FAIL ✗ |

**gate_complex: FAIL ✗**

---
## Verdict

**gate_complex FAIL (confirmed, exact):** `hpic_cer` ≡ `twofeature` to the decimal on every
metric, every seed → complex HPIC adds ZERO. Third independent confirmation (classifier,
router, this). DROP the complex formalism.

> **RED-TEAM CORRECTION (this-session, raw-verified — supersedes "ADOPT twofeature router"):**
> The conflict-aware 2-feature router does NOT beat plain softmax. Raw numbers (identical all 10 seeds):
> - **Recall@1:** softmax **0.6775** > twofeature 0.6425 → softmax WINS routing.
> - **conflict_abstain_recall:** softmax **0.600** > twofeature **0.290** → softmax WINS the very thing the conflict-router was built for.
> - twofeature wins only aggregate `abstain_f1` (+0.036), outweighed by −0.31 on conflict recall.
> So "gate_route PASS / adopt twofeature" is WRONG. The conflict-aware advantage from WP-ST-4's synthetic abstain task did NOT generalize to routing; softmax's entropy already captures multi-route uncertainty better.
> **METHOD FLAW:** std = 0.00000 across all 10 seeds → the dataset is identical per seed → the seed sweep is VACUOUS; the reported Cohen's d ≈ 5e7 are divide-by-zero artifacts and "10/10 seeds" is one result ×10. Deltas are exact so conclusions hold, but discard the inflated stats.

**CORRECTED ADOPTION:** **CER = plain softmax** (best on routing AND conflict abstention here).
DROP complex HPIC (cosmetic) AND the special 2-feature conflict router (loses to softmax).
The conflict/spread feature remains useful only where it was actually shown to help
(WP-ST-4's 1-axis synthetic abstain vs 1-feature baselines) — not for routing.
