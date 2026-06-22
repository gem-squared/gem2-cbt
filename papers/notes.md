# CBT — Operational Definition (minimal root)

> Scope: the *minimal* root needed to make v0 labels and loss well-defined.
> NOT the full group-theoretic equivariance theory (that is deferred to v1 / paper framing).

## 1. From equivariance to boundary-conditioned invariance

General equivariance (CNN / SO(3) / SE(3)):

    f(ρ_in(g)·x) = ρ_out(g)·f(x),    g ∈ G

The semantic property CBT cares about is the **special case ρ_out(g) = I**
(boundary-conditioned *invariance*), not full equivariance.

Given a contract / mandate `P`, define a semantic projection `Proj_P` that maps an
expression `x` to its semantic pixel under `P`. Let `T` be a transformation of the
surface form (paraphrase, reorder, translate, ...).

- **Boundary-preserving** `T` (stays inside `P`):   `Proj_P(T·x) = Proj_P(x)`
- **Boundary-crossing** `T` (leaves / violates `P`): `Proj_P(T·x) ≠ Proj_P(x)`

This is the entire root, in checkable form. v0 does not prove it; v0 *empirically tests*
whether a tiny Transformer can approximate `Proj_P` well enough to separate the two cases.

## 2. Semantic Pixel

    p := (F : A → B | P)

- `A` = input glyph / span / source
- `B` = realized meaning / output
- `P` = contract: domain, units, allowed concept, task rule
- Connection is decided **at the boundary** (compatibility of B with P), not inside.

JSON form used by the data + model:

    { "pixel_id", "level" ∈ {concept,context,task}, "input", "output", "contract", "label" }

## 3. The learnable predicate (what the boundary head approximates)

    compat_P(x) = 1  if  output(x) is admissible under contract P
                = 0  otherwise

`compat_P` is the operational stand-in for "Proj_P keeps x inside P".
- `label = "compatible"`   ⇒ target 1
- `label = "incompatible"` ⇒ target 0

The boundary head learns `compat_P`. Above-chance accuracy on a held-out set is the
v0 success criterion for "boundary compatibility is learnable".

## 4. Label rules per level (so synthetic data is consistent)

**Concept** — same glyph, contract fixes the admissible concept.
  compatible:   output concept ∈ contract.allowed_concept (matches domain+unit)
  incompatible: output concept is a valid concept of the glyph but NOT admissible under P
  (e.g. ρ | {physics, kg/m^3}: mass_density = compatible, resistivity = incompatible)

**Context** — role/argument structure under P.
  compatible:   surface differs but agent/patient roles preserved
  incompatible: roles swapped or relation changed
  (e.g. "a person holding an umbrella" vs "an umbrella holding a person")

**Task** — output must satisfy the task contract.
  compatible:   output uses only admissible (provided) facts
  incompatible: output adds facts not licensed by the source (boundary crossing = hallucination)

## 5. What v0 is / is NOT

v0 IS:  tiny Transformer + token + position + level + contract embeddings
        + LM head + boundary head, loss = L_LM + λ·L_boundary.
        Attention is UNCHANGED.

v0 is NOT: boundary-gated attention (v1), complex-valued nets, full TPMN compiler,
           LLM fine-tuning, large models.

## 6. Success criteria (today)

1. baseline tiny Transformer trains; LM loss decreases.
2. CBT-v0 boundary head reaches clearly above-chance held-out accuracy.
3. comparison table: {baseline, CBT-v0} × {LM loss, boundary acc, violation rate}.

---

## 7. Multi-seed evidence + bounded claim (post WP-ST-1)

The single-seed framing above has been superseded by a 5-seed sweep (seeds 0–4,
30 epochs each, frozen dataset hash `2888afb565326361`). The bounded claim and
limitations are documented in `papers/claim_v0.md`.

Key findings (WP-ST-1, corrected interpretation per WP-ST-2 U1):
- **Δconcept_acc PASS 4/5 seeds** (cbt_v0 − cbt_textonly). This result is **CONFOUNDED**:
  cbt_v0 bundles level-embedding + contract-embedding + capacity; gain cannot be
  attributed to contract content specifically.
- Δunsafe_accept and Δover_reject: FAIL (high seed variance in cbt_textonly baseline).
- Shuffle ablation: **INCONCLUSIVE** — the control was a consistent train+test bijection
  (relabeling of contract IDs), NOT a true disruption of the text↔contract association.
  "FAIL 3/5 marginal signal" was the wrong conclusion. See `papers/claim_v0.md` Limitation 4.
- Structural reason: context/task contracts are single-valued → contract embedding
  redundant with level embedding at those levels. Concept is the only testable level.
- Task level has trivial surface cue (length + n-gram = 1.000) — task acc is not a
  measure of semantic boundary understanding.

**CBT-v1 is gated.** WP-ST-2 addresses confounding (ablation ladder) + invalid shuffle
(example-wise random contract) + dataset leakage (family-split) before any v1 work.
