# Claim — WP-ST-10 CER/ECE first scoped microcell

**Date:** 2026-06-19  
**Scope:** First learned-extractor build of the CBT cell, scoped to memory-independent counterfactual SI bindings (the WP-6A regime where oracle payoff was real).

---

## Bounded scope (what these claims apply to — and only to)

- **Subject model:** `deepseek-chat` (DeepSeek), `temperature=0`
- **Items:** 37 clear-counterfactual SI-binding items
  - 25 in-domain (reused verbatim from WP-6A: 12 novel + 13 adversarial)
  - 12 NEW held-out (6 novel + 6 adversarial), built from disjoint SI units never used adversarially in WP-6A
- **Reps:** 3 per (item, condition) cell at `temp=0`
- **Conditions (4, pre-registered + frozen at U1):**
  - `B_FAIR` — fair SI-expert prompt, **no facts** (floor)
  - `PLAINFACTS` — same facts as ordinary prose, **no scaffold** (RAG-like, anti-RAG-confound control)
  - `C_PACK_LEARNED` — same facts inside the WP-6A contract scaffold, produced by a LEARNED ECE (DeepSeek llm_prompt CE)
  - `C_KNOW_ORACLE` — same facts inside the same scaffold, produced by WP-6A's hand-written oracle (ceiling)
- **Pre-registered gate floors:** Δ ≥ 0.05 and |Cohen d| ≥ 0.5 (or absolute_unanimity)

Out of scope: CBT-v1 boundary-gated attention (still GATED); ambiguous regime (WP-6A's saturated-fail zone); other models, other scaffolds, other domains.

---

## Pre-registered gates — verdicts

### `gate_learned_payoff` → **PASS** (all 3 splits)

| Split | Δ (PACK − B_FAIR) | Cohen d | Effect | Verdict |
|---|---:|---:|:---:|:---:|
| overall | −0.991 | −18.08 | very_large | **PASS** |
| in_domain | −0.987 | −14.80 | very_large | **PASS** |
| held_out | −1.000 | — | absolute_unanimity | **PASS** |

**Reading:** A LEARNED automatic extractor (DeepSeek llm_prompt CE, 1 LLM call per item) RETAINS the WP-6A oracle payoff. Counterfactual violation drops from ≈100% (no facts) to 0% (with the learned contract pack), on BOTH in-domain AND held-out splits.

**Bounded interpretation:** the WP-6A oracle-payoff finding survives moving from a hand-written contract pack to a learned automatic extractor — at this scope, for this model, with this scaffold.

---

### `gate_structure_vs_facts` → **FAIL [SATURATED TIE]** (all 3 splits)

| Split | V(C_PACK_LEARNED) | V(PLAINFACTS) | Δ | Effect | Floor-pass | Verdict |
|---|---:|---:|---:|:---:|:---:|:---:|
| overall | 0.0% | 0.0% | +0.000 | absolute_tie | ✗ | **FAIL (SATURATED)** |
| in_domain | 0.0% | 0.0% | +0.000 | absolute_tie | ✗ | **FAIL (SATURATED)** |
| held_out | 0.0% | 0.0% | +0.000 | absolute_tie | ✗ | **FAIL (SATURATED)** |

The pre-registered floor (Δ ≥ 0.05, PACK below PLAINFACTS) was not met. The verdict is **FAIL** by the rules locked at U1. The reframing below explains why FAIL here ≠ "structure has no value."

---

## Saturation Note (load-bearing — read before reading the verdict)

The gate_structure_vs_facts FAIL is a **SATURATED TIE**, not a demonstrated failure of contract structure.

**What "saturated" means here:**

- `PLAINFACTS` already drives violation to **0.0%** — the floor of measurable performance.
- `C_PACK_LEARNED` also drives violation to 0.0% — cannot go lower than zero.
- A tie at the floor is what this scope STRUCTURALLY ALLOWS. There is no headroom for any condition (structure or otherwise) to outperform PLAINFACTS, because PLAINFACTS already wins everything.

**The honest reading:** on the scoped binding-injection task with **single binding + short context + no distractor**, plain in-context facts SATURATE the violation metric. The structured contract pack offers **no demonstrated marginal value** over plain facts **here**.

### What the FAIL does NOT mean (refused overclaims)

- ⊬ It does **NOT** mean "contract structure is cosmetic in general." (L→G violation — refused.)
- ⊬ It does **NOT** mean "scoped CBT = RAG." (Drafted as a candidate claim at sign-off and explicitly rejected by David.)
- ⊬ It does **NOT** mean the contract-structure hypothesis is falsified. The test simply has no headroom on this scope; falsification requires a non-saturating scope.
- ⊬ It does **NOT** mean we should pivot architecture away from structured contracts now.

### What we CAN honestly say (bounded ⊢ claim)

> ⊢ **On the scoped single-binding SI-counterfactual task with deepseek-chat at temp=0, plain in-context facts saturate the violation metric at 0%; the structured contract pack therefore offers no demonstrated marginal value over plain facts at this scope.**

That is the entire scope of the gate_structure_vs_facts finding. Nothing more.

---

## Diagnostic — `C_PACK_LEARNED` vs `C_KNOW_ORACLE`

- 0 gap at verdict level (both 0.0% V).
- Inputs to the subject LLM are byte-identical on all 37 items (perfect U3 extraction → learned facts ≡ oracle facts; both serialized with `ensure_ascii=False` matching).
- A 5.4% byte-level near-determinism slip from DeepSeek at `temp=0` was observed on response TEXT but did not change a single verdict.

**Consequence:** the extraction-quality-floor question WP-6A flagged ("at what extractor noise level does payoff disappear?") is **untestable at this scope** — extraction was perfect (fidelity 1.000). Future scopes must inject extraction noise (or use a noisier extractor) to probe it.

---

## Rep-variance honesty audit

- Cells with within-cell variance: 1 out of 444 (item 224, B_FAIR — see Caveats).
- Deterministic cells: 442/444 (99.5%).
- This is structurally honest at temp=0 (identical inputs SHOULD give identical outputs); it is NOT WP-5-style vacuity.
- Effect-size labeling uses `absolute_unanimity_delta_<x>` and `absolute_tie` when paired-delta std=0, explicitly avoiding the WP-5 Cohen-d=∞ overclaim.

---

## Caveats (honest, not load-bearing on the verdicts)

1. **B_FAIR rate is 99.1%, not 100%** — a checker-keyword-overlap artifact on item 224 rep 2. The LLM answered "ampere per metre" (SI prior); the checker's `correct_answer_keywords = ["ampere", "a"]` matched substring "ampere" → counted NOT-violating. True B_FAIR rate is effectively 1.000. The gate_learned_payoff verdict is unaffected (very_large effect either way).

2. **DeepSeek `temp=0` near-determinism slip** — 6/111 byte-identical (PACK, ORACLE) pairs produced non-identical response TEXT. All 6 yielded matching VERDICTS. Known platform behavior, not a pipeline bug.

3. **The router was not stress-tested** — the scoped domain has only 2 binding-types with very distinct prose forms; CV and held-out routing accuracy were both 1.000. Recorded honestly in `data/cer_ece/router_report.json` as "wired, not stress-tested" — true router stress lives in scopes with more binding-types.

4. **Extraction was perfect (fidelity 1.000)** — the LEARNED ECE is essentially copy-prose-to-JSON on this scope. The "extractor-quality floor" curve cannot be drawn from one fidelity point. Recorded honestly as a single data point, not a curve.

---

## Architecture decision

**The scoped CBT extractor stack is NOT scaled as-is to more domains.**

The right NEXT experiment is a HARDER, NON-SATURATING injection scope where PLAINFACTS does not already win at the floor — only there can the structure-vs-facts comparison surface a real signal.

### Next WP — WP-ST-11 (proposed): harder non-saturating injection scope

Falsifier-spirit successor to WP-10. Designed so PLAINFACTS does NOT saturate at 0% violation. Stressors (any or all, pre-registered before any run):

1. **Multi-binding** — multiple in-context bindings to track simultaneously (e.g., 3–5 concurrent quantity → unit redefinitions). PLAINFACTS prose becomes long; structure may help the model anchor each binding to its quantity.
2. **Distractors** — irrelevant binding facts interleaved with the load-bearing one; the structured pack may help the model attend to the marked binding instead of an interleaved decoy.
3. **Long context** — embed the binding hundreds to thousands of tokens away from the question, with intervening irrelevant text. The pack may survive the distance better than prose.
4. **Conflicting constraints** — two binding statements that partially conflict; the contract pack can express a precedence order or a `forbidden` field that prose cannot make first-class.

Falsifier:
- If `C_PACK_LEARNED` beats `PLAINFACTS` on any of those by Δ ≥ floor → `gate_structure_vs_facts` PASS on a non-saturating scope → structured contracts have demonstrated marginal value somewhere in the tested space.
- If `C_PACK_LEARNED ≈ PLAINFACTS` across the harder stressors too → structure is cosmetic across the explored injection space → the next pivot (decode-time / verifier-gated) becomes the strong default.

### Future pivot — decode-time / verifier-gated contracts

Still a valid candidate direction. **Ordered AFTER WP-11**, not before. Rationale:

- If structure pays off on harder injection scopes → keep structured contracts; scope-selection is the open question, not architecture.
- If structure ties even on harder injection scopes → facts-injection mode of CBT is RAG-equivalent across the tested space; the structural payoff (if any) lives elsewhere — most plausibly at decode-time (CONSTRAIN generation rather than SUPPLY facts).

Premature pivot away from structured-contract architecture on a single saturated scope would be a S→T overclaim. WP-11 first.

---

## What we will NOT do

- ⊢ NOT scale the current pack format to more domains and call it "CBT" — that would be selling RAG-with-extra-JSON.
- ⊢ NOT claim general cosmetic failure of contract structure — the evidence does not support that.
- ⊢ NOT pivot architecture (to decode-time gates) BEFORE the harder-scope structure test — that would be a structural conclusion drawn from a single saturated scope.

---

## CBT-v1 status

CBT-v1 boundary-gated attention REMAINS GATED. WP-10 did not test it — out of scope per the work plan. The CBT-v1 question is independent of the WP-10 / WP-11 extractor-stack thread.

---

## Sign-off

- **Planner (Alchy + Kritik):** bounded claim accepted with the SATURATED TIE reframing per David's redirect.
- **Engineer (Gineer):** all 7 units executed, all artifacts persisted (`scripts/cer_ece_cell.py`, `data/cer_ece/*`, `papers/results_cer_ece.md`, this claim).
- **Human (David):** explicit redirect on the claim wording applied verbatim: keep gate_learned_payoff PASS; reframe gate_structure_vs_facts FAIL as SATURATED TIE; no universal cosmetic claim; plan WP-11 (harder non-saturating scope) before any decode-time pivot.

## References

- `data/cer_ece/items.jsonl` — frozen items (hash `c52feb2b07876905`).
- `data/cer_ece/raw_runs.jsonl` — 444 cells (37 items × 4 cond × 3 reps).
- `data/cer_ece/eval_results.json` — all per-condition rates, paired deltas, gate verdicts.
- `data/cer_ece/extraction_fidelity.json` — ECE fidelity vs oracle.
- `data/cer_ece/binder_spotcheck.json` — info-constancy + scaffold-cleanliness audit.
- `data/cer_ece/router_report.json` — CER router (cv 1.000, held-out 1.000, honesty note recorded).
- `papers/results_cer_ece.md` — full results table.
- `.gem-squared/work-plan/WP-ST-10.md` — work plan + per-unit results.
- WP-ST-6A — counterfactual items + checker + oracle pack (PRESERVED, untouched).
- WP-ST-7 — `concept_ce.py` llm_prompt CE (the LEARNED extractor pattern reused).
- WP-ST-5 / WP-ST-8 — softmax-over-raw-features router pattern (reused).
