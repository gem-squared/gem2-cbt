# Bounded Claim: HPIC as a Classifier — Spaceship Titanic Falsification

**Construct:** HPIC (*Hierarchical Phase-Interval Classifier*) — a complex-plane classifier.
**Origin:** predates the CBT work; this is HPIC's *first* falsification, in its original role as a
full classifier. (Its later falsification as a CER **router/⊥-gate** is in `claim_hpic_cer.md` /
`claim_hpic_gate.md`.)
**Verdict:** **FALSIFIED as a competitive classifier** on Spaceship Titanic — below logistic
regression at full coverage and at every abstention level. Two independent implementations agree.

---

## 1. What HPIC is

HPIC reframes classification as **direction on a complex "classifier plane" `Cf`**. A probability
becomes an angle; evidence regions contribute complex vectors; the decision is read off the real axis.

```
θ = arccos(2p − 1)        # p=1 → 0° (class A) ; p=0.5 → 90° (Unknown) ; p=0 → 180° (class B)
z = r · e^{iθ}            # per-evidence phasor
Decision = sign(Re(Z))    # Z = Σ contributions ; ≈0 → Unknown / abstain
```

`cos θ` is the projection onto the verdict axis of `Cf` — **not** the probability itself
(an early "confidence" mislabel, corrected in refinement).

**Sm — the evidence unit.** Not a single feature but a small **relation-region**: a bundle of 2–5
feature-conditions (default 3) drawn from training data. Each condition `c_j` has a shrunk rate
`p̃(c_j)` and angle `θ(c_j)`; the Sm's center is the circular mean blended with the joint angle.

**Aggregation — vector sum, not intersection.** An instance participates in several Sm's at once;
their phasors **add**, so repeated same-direction evidence accumulates:

```
Z(x) = Σ_{i ∈ I(x)} r · e^{iθ_i(x)}
```

**Interval form.** Each Sm contributes an *angular interval* Ω (width driven by uncertainty τ and
soft membership m). The summed interval's position relative to the 90° axis gives the verdict
(entirely <90° → A, >90° → B, crossing 90° → abstain). The point form is the center approximation.

So the name maps to the mechanics: **Phase** (angle θ), **Interval** (interval-valued Ω),
**Classifier**; "Hierarchical" refers to the conditions → Sm-bundle → aggregate composition.

## 2. The defining principle — LLM-in-the-loop at one step

HPIC's non-negotiable design rule: **every extrapolation passes through an LLM; statistics verify.**
Realized at exactly one place — **Sm proposal**:

```
LLM input     = condition statistics (feature, value, support n, shrunk rate p̃, direction, strength)
LLM job       = propose 2–5-condition Sm bundles BY MEANING (coherent subpopulations)
LLM forbidden = predicting labels, recomputing probabilities, seeing the target
downstream    = fully deterministic HPIC code
```

This is the object of the ablation in §4, not an add-on.

## 3. How it was built — construct → attack → repair → test → falsify

Three-party refinement: **David** (architect/arbiter, owned the idea + the LLM principle),
**ChatGPT** (formalization), **Claude** (red-team). The formula was closed only after ~nine
objection→repair cycles, including:

- intersection → **vector sum** (intersection loses evidence accumulation);
- single angle → **angular interval** (probabilistic regions give intervals, not points);
- a **σ-rotation rule rejected** — it could rotate past the 90° axis and flip the verdict sign;
- hard → **soft membership** (`m_i(x) ≥ λ`, λ set by validation);
- **shrinkage** `p̃ = (k + α·p₀)/(n + α)` — low support pulls the angle toward 90° (regularization
  in the geometry, not an external penalty), so support is **not** double-counted;
- **joint-blend** — marginal angular average loses interaction; blend in the joint-conditional angle;
- **τ** = angular *uncertainty width* (condition disagreement + support uncertainty + validation
  instability), not raw spread.

## 4. Results — falsified from data

5-fold out-of-fold on the real `train.csv`; two independent implementations agree:

```
Model                        Accuracy    Committed acc @30% coverage
GBT                          0.8109      0.9862
Logistic                     0.7977      0.9582
HPIC (impl A, 19 cols)       0.7445      0.8803
HPIC (impl B,  9 cols)       0.7217      —
HPIC mechanical bundles      0.6945
LLM-proposed Sm vs mechanical  +0.0273              # the §2 principle, isolated
λ nested-CV vs fixed 0.5       0.6820 vs 0.6945     # calibration did not raise the ceiling
```

**HPIC is falsified as a competitive classifier on this dataset** — below logistic at full coverage
and at every abstention level. The failure came *after* the formula was closed, implemented, and
compared — not from being undefined. The **LLM-proposal principle shows a small, real positive lift**
(+0.027) over mechanical bundles, but from a baseline far below the linear model. Threshold
calibration did not lift the ceiling.

## 5. Scope and limits

- **Does not claim** HPIC is a competitive classifier, or that the complex/interval formalism adds
  predictive value on this task.
- **Does claim** a fully-specified, implemented, and independently-reproduced construct, honestly
  falsified against strong baselines; and a small isolated lift from the LLM-in-the-loop Sm-proposal
  step over mechanical bundling.
- **Defined but unmeasured:** the interval/abstention machinery on its *intended* domain (imbalanced,
  dirty-label) — Spaceship Titanic is neither, so the interval form was not stress-tested where it
  was meant to help.
- Single dataset, single task. Not a general claim about complex-valued classifiers.

## 6. Relation to the CBT record

HPIC has now been **honestly falsified twice, in two roles**:

1. **As a classifier** (this paper) — below linear on Spaceship Titanic.
2. **As a CER router / ⊥-gate** (`claim_hpic_cer.md`, `claim_hpic_gate.md`, WP-4/5/8) — *cosmetic*:
   `Z = Σ ρ·e^{iθ}` reduced to an invertible reparameterization of two real features
   (`signed_strength`, `evidence_spread`) and lost to plain softmax.

In both roles the durable asset was **the procedure — construct → attack → repair → test →
falsify — not the mechanism.**

## 7. Role split (for the record)

```
ChatGPT: intuition → formula; Cf/Sm structure; θ mapping; soft-membership; interval summation
Claude:  attacked undefined parts; intersection-vs-sum; cosθ "confidence" misuse; marginal-average
         interaction loss; arccos noise amplification; support double-counting; σ-rotation sign-flip;
         the baseline-falsification requirement
David:   proposed the idea; set the LLM-extrapolation principle; arbitrated which definition kept the
         original intuition; ran the measurements
```

## Citation

> David Seo / GEM².AI (2026). *HPIC: a complex-plane classifier — construction and falsification.*
> CC-BY-4.0. Part of the gem2-CBT research log.

*Conclusions are bounded. The value of this record is the audited path: construct → attack → repair
→ test → falsify.*
