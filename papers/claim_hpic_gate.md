# CBT HPIC ⊥-Gate Claim — Conflict-Aware Abstention

**Experiment:** WP-ST-4 | **Date:** 2026-06-19 | **Status:** VERIFIED
**Config hash:** `42189d21dc9c0e91` | **Seeds:** 0–9 | **N:** 3000/seed | **K:** 5 sources
**ρ formula:** `|2p_k − 1|` (pre-registered, not learned)

---

## Verdict

**g_B: PASS — but the win is the SPREAD/conflict FEATURE, not the complex/angular form.**

> **RED-TEAM CORRECTION (this-session, code-verified — supersedes the original "adopt HPIC complex" conclusion):**
> WP-4's baselines (`linear`, `max_prob`) are BOTH 1-feature (magnitude of the signed score). They throw away the evidence-spread dimension. HPIC's `angular std` IS that spread feature (`Im(Z)=Σρ·2√(p(1−p))`, proven this session at ~3e-15). So HPIC beating the 1-feature baselines only shows "2 features (signed, spread) > 1 feature", NOT "complex helps".
> **Direct test (this session, on WP-4's frozen 3-regime task, 10 seeds):** a PLAIN 2-feature rule `certainty = |signed| − w·spread` achieves conflict-recall@80%-CLEAR-coverage = **1.000 ± 0.000 — IDENTICAL to hpic_interval (1.000 ± 0.000).** The complex/angular machinery adds nothing; it is an invertible reparam of `(signed_strength, evidence_spread)`.
> **Corrected adoption:** adopt a **conflict-aware abstain gate using two real features (signed_strength, evidence_spread) + a threshold.** The complex/phase form is COSMETIC and is dropped (consistent with the router-cosmetic proof). WP-ST-5's decisive control is exactly this 2-feature router — it will re-confirm.

**g_A: PASS vs max_prob only; vs linear below floor.** Consistent with the above (linear is 1-feature on the *signed* axis; once spread is added, the 2-feature rule dominates — but that is the spread feature, not complex numbers).

---

## Decisive Evidence

### g_B: HPIC > baselines on Conflict-Regime Abstain Recall

Operating point: CLEAR coverage ≈ 80% (the gate commits on 80% of unambiguous instances).

| Comparison | Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|---|
| hpic_point vs linear | 10/10 | +0.066 | 6.79 | PASS ✓ |
| hpic_point vs max_prob | 10/10 | +0.114 | 9.55 | PASS ✓ |
| **hpic_interval vs linear** | 10/10 | **+0.093** | **6.65** | **PASS ✓** |
| **hpic_interval vs max_prob** | 10/10 | **+0.140** | **8.85** | **PASS ✓** |

**hpic_interval achieves 100% conflict abstain recall (mean 1.000±0.000) across all
10 seeds** — it never commits on a conflicting-evidence instance at this operating point.

**linear achieves 90.7% conflict recall; max_prob achieves 86.0%** — they commit on
conflicting evidence ≈10–14% of the time, introducing avoidable classification errors.

### g_A: HPIC > max_prob on Risk–Coverage AUC

| Comparison | Seeds winning | Mean Δ (lower is better) | Cohen's d | Verdict |
|---|---|---|---|---|
| hpic_interval vs max_prob | 10/10 | +0.037 | 8.50 | PASS ✓ |
| hpic_point vs max_prob | 10/10 | +0.030 | 9.31 | PASS ✓ |
| hpic_interval vs linear | 10/10 | +0.022 | 7.31 | FAIL ✗ (below Δ floor) |
| hpic_point vs linear | 10/10 | +0.015 | 8.10 | FAIL ✗ (below Δ floor) |

The linear gate is a strong baseline that the angular criterion doesn't dominate in
overall AUC (within the Δ≥0.03 floor), but the CONFLICT regime recall is the more
diagnostic comparison (g_B) — and there both HPIC variants win decisively.

---

## Mechanism: Why Angular Dispersion Catches Conflict

For **CLEAR** evidence (4 sources aligned, 1 weak):
- `θ_k ≈ 37°` for all aligned sources → tight angular cluster far from 90°
- Angular std ≈ small; certainty = `|mean_θ − 90°| / (std_θ + ε)` → HIGH → commits ✓

For **INSUFFICIENT** evidence (all p ≈ 0.5):
- `θ_k ≈ 90°` for all → `|mean_θ − 90°| ≈ 0`; certainty → LOW → abstains ✓

For **CONFLICTING** evidence (e.g., 2 strong-A + 3 strong-B sources):
- A-sources: `θ ≈ 37°`; B-sources: `θ ≈ 143°`
- Angular std ≈ 44°; `|mean_θ − 90°| ≈ 9°` (near 90°); certainty → LOW → abstains ✓

For **asymmetric conflict** (e.g., 3 strong-A + 2 strong-B):
- Linear score: `Σρ(2p−1) = 3(0.49) − 2(0.49) = 0.49 > 0` → linear **commits to A**
- Angular std still ≈ 44° (same spread); certainty still LOW → hpic_interval **abstains** ✓

**This is the key differentiation**: linear score direction follows the majority and
commits on asymmetric conflict. The angular dispersion criterion is sensitive to the
PRESENCE of strong opposing evidence, not just the net magnitude — matching the
intended ⊥ semantics (the system should flag that BOTH directions are strongly asserted).

---

## Scope and Boundaries (EEF: ⊢ grounded claims only)

- ⊢ **Claim**: HPIC angular-dispersion abstain criterion outperforms scalar thresholds
  on conflict-regime abstain recall in a controlled synthetic evidence task (g_B PASS).

- ⊢ **Does NOT claim**: HPIC improves overall risk–coverage AUC vs the linear baseline
  beyond the effect-size floor (g_A partial). Linear remains a strong scalar competitor.

- ⊢ **Task scope**: 1-axis binary abstain-gate (accept A / accept B / abstain ⊥).
  Does NOT address M-way routing or multi-class classification. Router role is CLOSED
  (use softmax/logistic — proven equivalent to HPIC for that role).

- ⊢ **Synthetic only**: controlled evidence task with K=5 binary sources and pre-registered
  ρ_k = |2p_k − 1|. NOT tested on NLP, WSD, or real-world ambiguous signals yet.

- ⊢ **ρ constraint**: ρ must remain a deterministic pre-registered formula — NOT learned
  (else the proof that HPIC router ≡ logistic applies and the complex math collapses).

- ⊢ **CBT-v1 gate**: CBT-v1 (boundary-gated attention) remains GATED. The g6 condition
  from WP-ST-3 (cbt > text_only_nn on seen-lemma WSD) was NOT met. Adoption of the
  HPIC ⊥-gate does not unlock CBT-v1.

---

## Adoption Decision (CORRECTED)

**CONFLICT-AWARE 2-FEATURE ⊥-gate ADOPTED** for CBT's abstain criterion. The complex/
angular HPIC form is NOT adopted (cosmetic — ties the 2-feature rule, see RED-TEAM CORRECTION).

1. **Form**: two deterministic features per instance — `signed_strength = Σ ρ_k(2p_k−1)`
   and `evidence_spread = Σ ρ_k·2√(p_k(1−p_k))` — with `ρ_k=|2p_k−1|` (pre-registered);
   abstain when signed-strength is small OR spread is large (threshold). (The complex
   `Z=Σρ e^{iθ}` is an invertible reparam of these two; use whichever is simpler — the
   real-valued pair, no complex numbers needed.)
2. **Role**: accept/reject/⊥ decision at CBT's output — NOT routing/classification (closed → softmax).
3. **Threshold**: swept at deployment; operating point (e.g. 80% CLEAR coverage) is a deployment parameter.
4. **Why this matters (the real finding)**: a conflict/spread feature catches "both directions
   strongly asserted" that a scalar signed score (linear/max_prob) misses on asymmetric conflict.
   THAT is the contribution. The complex dressing is not.
5. **Next step**: validate the 2-feature conflict-aware abstain on real NLP signals (future WP);
   WP-ST-5 already pits it against complex HPIC-CER on a micro-domain.

---

## Relationship to Prior Claims

| Claim | Result |
|---|---|
| claim_v0.md (WP-ST-1) | CONFOUNDED — shuffled label invalid |
| claim_v2.md (WP-ST-2) | PASS — contract content isolated, synthetic data |
| claim_v3.md (WP-ST-3) | PARTIAL — g3b PASS zero-shot NL; g6 FAIL seen-lemma NL |
| **claim_hpic_gate.md (WP-ST-4)** | **PASS g_B (decisive) + g_A (vs max_prob)** |

Router question (M-way HPIC): CLOSED → use softmax (proven equivalent).
HPIC ⊥-gate (1-axis abstain): OPEN → PASS on synthetic controlled evidence.

---

*WP-ST-4 | config hash 42189d21dc9c0e91 | 10 seeds | K=5 | ρ=|2p−1|
EEF: ⊢ grounded on synthetic controlled evidence. CBT-v1 still GATED (WP-ST-3 g6 unmet).*
