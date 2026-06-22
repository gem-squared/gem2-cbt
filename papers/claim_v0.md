# CBT-v0 Bounded Claim

_Frozen dataset hash: `2888afb565326361` | Seeds: 0–4 | 30 epochs per run_
_See `papers/results_robust.md` for full tables and per-seed deltas._

---

## What this is

CBT-v0 is a tiny character-level Transformer (~260K params) trained on controlled
synthetic data. It adds a boundary classification head to a standard LM and optionally
injects level and contract embeddings (the "semantic pixel"). All training is on
deliberately simple synthetic sentences — NOT on natural language, large corpora, or
real-world tasks.

---

## The claim (scoped)

> **CBT-v0 provides early evidence — on controlled synthetic data — that a model with
> level+contract embeddings outperforms a text-only boundary head at the concept level
> (Δconcept_acc PASS 4/5 seeds). This result is CONFOUNDED: cbt_v0 bundles
> level-embedding + contract-embedding + additional capacity together; the gain cannot
> be attributed to contract content specifically.**
>
> **The shuffle control used in WP-ST-1 was INVALID** (consistent train+test bijection =
> relabeling of a learnable embedding = does not disrupt the text↔contract association).
> The shuffle results are INCONCLUSIVE, not evidence of marginal signal.
>
> This is a confounded discriminability result on one synthetic dataset. It does NOT
> show that the model solves hallucination, proves semantic equivariance, proves contract
> content is load-bearing, or generalises beyond the training distribution.

Scope boundaries (explicit):
- "Early evidence" — 5-seed sweep on one synthetic dataset. Not a robust empirical result.
- "Controlled synthetic data" — sentences are short, labels are deterministic, surface
  cues exist at the task level (see Limitation 1 below).
- "Specifically at the concept level" — context/task contracts are single-valued per
  level in the current dataset; contract injection is therefore a no-op at those levels.
- "Not solving hallucination" — the task-level boundary is trivially separable by
  surface cues; task accuracy reflects the cue, not semantic understanding.
- "Not proving semantic equivariance" — the full group-theoretic property (v1 goal)
  is out of scope here.

---

## Evidence

From `papers/results_robust.md` (5 seeds, 30 epochs each, frozen hash `2888afb565326361`):

| Metric | cbt_textonly (mean±std) | cbt_v0 (mean±std) | Direction |
|---|---|---|---|
| Boundary Acc | 0.764 ±0.028 | 0.800 ±0.015 | ↑ good |
| Unsafe Accept Rate | 0.253 ±0.099 | 0.232 ±0.050 | ↓ good |
| Concept Acc | 0.535 ±0.054 | 0.592 ±0.026 | ↑ good |
| Context Acc | 0.788 ±0.051 | 0.837 ±0.049 | ↑ good |
| Task Acc | 1.000 ±0.000 | 1.000 ±0.000 | — (trivial cue) |

**Per-seed paired delta summary (cbt_v0 − cbt_textonly):**

| Delta | Result | Criterion |
|---|---|---|
| Δconcept_acc (positive = good) | **PASS 4/5 seeds** | ≥4/5 |
| Δunsafe_accept (negative = good) | FAIL 2/5 seeds | ≥4/5 |
| Δover_reject (negative = good) | FAIL 2/5 seeds | ≥4/5 |

The only delta that consistently moves in the good direction across seeds is concept accuracy.
Unsafe accept rate and over-reject rate have high seed-to-seed variance (cbt_textonly unsafe_accept
ranges 0.144–0.407 across seeds), which drowns the signal.

**Shuffle ablation (cbt_v0 vs cbt_v0_shuffled) — INVALID CONTROL, results INCONCLUSIVE:**

The shuffle applied the same concept-contract permutation to both train and test sets
(a consistent bijection). This is a relabeling of the contract embedding IDs — the model
can re-learn the same text↔contract association under new names. The shuffle did NOT
disrupt the correlation between contract content and labels. The numeric results below
are therefore uninterpretable as evidence for or against contract-content dependence.

| Delta | Numeric result | Interpretation |
|---|---|---|
| Δshuffle_concept_acc | FAIL 3/5 seeds | INCONCLUSIVE (invalid control) |
| Δshuffle_unsafe_accept | FAIL 2/5 seeds | INCONCLUSIVE (invalid control) |

---

## Structural finding (predicts the pattern above)

**Why concept gains but context/task do not:**

The current dataset has exactly **one contract per level per direction**:
- Context contract: `"role-preserve"` (all context examples, both labels)
- Task contract:    `"facts-only"` (all task examples, both labels)

Within-level shuffle of these single-valued contracts is a mathematical no-op — the
contract embedding the model sees is identical before and after shuffle. The contract
embedding for context and task levels is therefore **redundant with the level embedding**;
it contributes no additional discriminative signal.

At the concept level there are 15 distinct contracts (e.g. `"physics/kg/m^3"`,
`"biology/cells/mm^2"`, ...). The within-level shuffle genuinely perturbs 12/15 of them.
So concept-contract is the only level where the embedding carries information beyond
what the level ID already encodes — and it is the only level where cbt_v0 consistently
beats cbt_textonly.

**Implication:** the seed-0 pattern (concept improves, context flat) is structurally
predicted by this finding, not accidental. Any model trained on this dataset will show
the same pattern regardless of seed, as long as the data generator produces single-valued
context/task contracts.

---

## Limitations

**Limitation 1 — Task level has a trivial surface cue.**
Incompatible task examples always append an extra clause (`"and ..."` via the EXTRAS
generator). A length classifier and an n-gram classifier both achieve test accuracy 1.000
on task-level examples. Task accuracy in all reported tables reflects this surface cue,
not semantic boundary detection. Task results should be disregarded for the purpose of
evaluating boundary understanding.

**Limitation 2 — Context/task contracts are single-valued (shuffle is a no-op).**
As described in the structural finding above, the shuffle ablation cannot test
context/task contract signal. The ablation is a concept-contract test only. Extending
the ablation to context and task would require generating multiple contracts per level —
this is the optional U9 scope.

**Limitation 3 — Unsafe accept rate has high seed variance.**
cbt_textonly unsafe_accept_rate ranges 0.144–0.407 across seeds (std=0.099). This
variance makes it impossible to establish a reliable direction for Δunsafe_accept with
5 seeds. The variance is likely caused by boundary head convergence instability in small
models on small synthetic datasets.

**Limitation 4 — Shuffle control was invalid; contract-content effect is untested.**
The WP-ST-1 shuffle applied a consistent train+test bijection (same permutation on both
splits). This is equivalent to relabeling the contract embedding IDs. The model can
learn the relabeled mapping just as easily as the original — the text↔contract
association is preserved. The shuffle did NOT test whether contract *content* matters.
The results are INCONCLUSIVE, not evidence of marginal signal. A valid control requires
example-wise independent random assignment (breaking the text↔contract correlation
without a consistent relabeling). This is addressed in WP-ST-2.

**Limitation 5 — Synthetic data only.**
All results are on a controlled synthetic dataset generated from a small template pool.
No claim is made about generalization to natural language or real LLM outputs.

---

## Comparison axes

- `baseline_lm`: no boundary head; LM loss only. Reference for perplexity trade-off.
  Boundary acc column shows `—` (no prediction). Not a fair boundary comparison.
- `cbt_textonly`: adds boundary head; uses text + level embedding. No contract injection.
  The appropriate text-only baseline for measuring the contract injection effect.
- `cbt_v0`: adds contract (semantic pixel) injection on top of cbt_textonly.
  The primary config under evaluation.
- `cbt_v0_shuffled`: cbt_v0 architecture with concept contracts permuted within level.
  **INVALID CONTROL** — the permutation was applied consistently to train+test (relabeling).
  Results are INCONCLUSIVE; see Limitation 4.

Majority-class boundary accuracy (reference): **0.561** (on frozen test set).

---

## CBT-v1 gate

CBT-v1 (boundary-gated attention) is **explicitly gated** pending external review of
this bounded claim document. The preconditions for CBT-v1 work are:

1. This document accepted as the WP-ST-1 multi-seed claim output.
2. The concept-level discriminability result (4/5 seeds PASS on Δconcept_acc) treated
   as the single PASS finding — not as evidence for full contract-level boundary control.
3. Limitations 1–5 acknowledged as in-scope for any v1 extension.

If the reviewer accepts this framing, the recommended next step is either:
- (a) U9 (optional): enrich context/task contracts to make the shuffle ablation testable
  at all three levels before proceeding to v1.
- (b) Proceed directly to CBT-v1 with the understanding that current evidence supports
  concept-level discrimination only.

The choice between (a) and (b) requires explicit human sign-off.
