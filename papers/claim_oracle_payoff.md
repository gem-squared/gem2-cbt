# Claim — Oracle-Contract Payoff (WP-ST-6)

**Status:** BOUNDED CLAIM, pending David red-team sign-off.
**Subject model:** `deepseek-chat` (DeepSeek API, OpenAI-compatible `/chat/completions`, `temperature=0`).
**Domain:** SI symbol/unit micro-domain — 10 quantities × {what_unit, is_correct_unit, is_wrong_unit, what_quantity} CLEAR (40 items) + 10 AMBIGUOUS (qualifier-required) = 50 items, frozen hash `ee16b7b03512a143`.
**Conditions:** A naked / B strong-prompt / C oracle-contract / Cp wrong-contract. **N_reps = 3 per cell, gate floor Δ ≥ 0.05 absolute.** 600 records, 0 errors.

---

## 1. Raw numbers (deepseek-chat, temp=0, N=3)

| Cond | n | V_clear | cell-σ_clear | V_ambig | Abstain_ambig | CW_clear | CW_ambig |
|---|---|---|---|---|---|---|---|
| A — naked | 150 | 0.025 | 0.156 | 0.000 | 1.000 | 0.025 | 0.000 |
| **B — strong prompt** | 150 | **0.000** | 0.000 | **0.900** | 0.100 | 0.000 | 0.900 |
| **C — oracle contract** | 150 | **0.000** | 0.000 | **0.000** | 1.000 | 0.000 | 0.000 |
| Cp — wrong contract | 150 | 0.875 | 0.331 | 0.500 | 0.500 | 0.875 | 0.500 |

⊢ All numbers reproducible from `data/oracle_payoff/eval_results.json` (deterministic checker on `raw_runs.jsonl`).

## 2. Literal gate verdicts (per WP CONTRACT.B, CLEAR-only)

| Gate | Formula | Δ | Verdict |
|---|---|---|---|
| `gate_payoff` | V_B − V_C ≥ 0.05 | **+0.000** | **FAIL (literal)** |
| `gate_uses_contract` | V_Cp − V_C ≥ 0.05 | **+0.875** | **PASS** |
| `headroom_BA` | V_A − V_B | +0.025 | small |

⊢ The literal `gate_payoff` FAIL is honored as a number.

## 3. ⚠ MEMORIZATION CONFOUND — `gate_payoff` is UNINTERPRETABLE here

`V_clear(B) = 0.000` exactly. The strong prompt saturates a hard floor on the CLEAR factual subset. Oracle contract C has **no headroom to add** below zero. Therefore the literal `gate_payoff` FAIL **cannot distinguish**:

- **H_dead:** "oracle contract is useless / decorative" — the architecture-rejecting reading.
- **H_memorized:** "deepseek-chat has SI units memorized, so strong-prompt alone hits the truth floor; nothing is left for an oracle contract to fix on this subset" — a domain artifact, NOT evidence against contracts.

⊨ The cross-WP signal from WP-7 U5 (`llm_prompt` accuracy = 1.000 on the same inventory under DeepSeek) is consistent with H_memorized.

**Therefore:** **U6 explicitly DECLINES to use the literal `gate_payoff` FAIL as architecture-decision evidence.** Doing so would commit `S→T` (treating a domain artifact as a permanent trait of contracts) and would inflate `Δe→∫de` (one memorized micro-domain → all-domain verdict).

To recover an interpretable `gate_payoff`, follow-up WP-ST-6.1 is required: a **counterfactual-contract subset** where the contract carries non-memorizable information (novel symbol↔quantity bindings, fictional units, or context-overrides of the SI prior) such that **only contract-conditioned C can be correct** — the LLM-prior cannot solve it. This isolates contract content as the active variable.

## 4. The interpretable signals (NOT memorization-confounded)

### 4.1 `gate_uses_contract` — STRONG PASS

V_Cp − V_C = **+0.875** (CLEAR). The wrong contract drives confident-wrong factual answers from 0% to 87.5%. ⊢ Contract content is unambiguously **active in the LLM's decision** — Cp flipped "what's the SI unit of force?" → "joule, J" and "Is newton the SI unit of force?" → "No". This is not a behavior an indifferent model would show. **The contract is read and obeyed.**

### 4.2 AMBIG-regime: C dominates B on abstention (Δ = +0.900)

This is the most architecturally informative finding and it is **not memorization-confounded** (abstention is a behavioral pattern, not a factual recall task):

- V_ambig(B) = **0.900** (strong prompt collapses qualification → terse Yes/No)
- V_ambig(C) = **0.000** (oracle contract restores qualification)
- Δ_C-over-B = **+0.900** absolute (≫ 0.05 floor)
- Abstain_ambig: B = 0.100 vs C = 1.000

Mechanism: B's general guardrail includes "for Yes/No questions, respond with ONLY the word Yes or No" — a sensible rule for CLEAR items that **catastrophically misfires** on AMBIGUOUS items. C's per-item contract explicitly says "Acknowledge the ambiguity explicitly. Do NOT give a single confident answer without qualification." The contract **resolves the rule conflict** that a static prompt cannot.

⊨ A strong static prompt cannot encode "be terse when sharp, qualify when fuzzy" simultaneously — it has to pick one. A per-item contract can.

## 5. Architecture decision

⊢ **REFRAMED VERDICT (per memorization-confound discipline):**

**NOT a `gate_payoff` PASS.** **NOT a `gate_payoff` FAIL → reject build.** The literal `gate_payoff` is UNINTERPRETABLE on this memorized subset.

**Conditional justification:**

| Architecture question | Evidence | Verdict |
|---|---|---|
| Is contract content actively used by the LLM? | `gate_uses_contract` Δ=+0.875 | ⊢ **YES — confirmed** |
| Does an oracle contract beat a strong prompt on factual CLEAR items? | `gate_payoff` Δ=0.000 BUT V_B=0 floor | ⊥ **UNDETERMINED (memorization confound)** |
| Does an oracle contract beat a strong prompt on AMBIGUOUS items? | Δ_C-over-B = +0.900 on V_ambig | ⊢ **YES — strong** |

**Recommendation:** CBT contract-conditioning architecture is **justified for the abstention/qualification use case** in this micro-domain on this model. The CLEAR-factual payoff question is **DEFERRED** to WP-ST-6.1 (counterfactual-contract subset).

**Specifically:**

- ⊢ A static strong prompt has **a structural ceiling on multi-regime tasks** (one set of rules cannot serve both "be terse when sharp" and "qualify when fuzzy"). Per-item contracts pass that ceiling. The CBT pipeline's contract-conditioning step has **measurable behavioral value** on at least the abstention dimension.
- ⊬ Whether building the full 12-component CBT extractor stack (CER routers / ECE / Binder / Validators / boundary heads) is justified depends on **whether automated extraction of contracts at inference time matches the quality of hand-written oracle contracts**. WP-6 used oracle (hand-written) contracts; extractor-quality is a separate empirical question (WP-ST-6.1 or later).
- ⊥ Whether contract conditioning matters on **non-memorized factual subsets** is the unresolved question. Until WP-ST-6.1 closes it, the architecture decision cannot be a full PASS or full FAIL.

**Action items (pending David sign-off):**
1. **CBT-v1 stays gated.** This WP does not lift the gate.
2. **WP-ST-6.1 proposed:** add counterfactual-contract subset (novel symbol↔quantity bindings the LLM cannot have memorized) and rerun the four conditions; gate_payoff on that subset is the missing piece.
3. **CBT abstention-conditioning component justified for planning** — the AMBIG result is robust enough to start designing the contract-pack format and binding interface, even before WP-ST-6.1 closes the factual payoff question.

## 6. Bounded scope (don't overclaim)

- ⊢ **Subject:** `deepseek-chat`. Other models may show different `gate_payoff` headroom (e.g., a weaker base model with V_B > 0 would yield an interpretable gate_payoff result). Optional follow-up: cross-model check.
- ⊢ **Domain:** SI symbol/unit factual + qualification (50 items). Generalization to other domains is **unsupported** by these data alone (avoid `L→G`).
- ⊢ **N=3 reps at temp=0:** byte-perfect across reps for B and C; A and Cp showed some rep-flip (cell-σ_clear 0.156 and 0.331 respectively). Variance is real on the boundary-flipping conditions; per-cell verdicts are stable for the saturated cells.
- ⊬ **Cp's 0.875 V_clear is a ceiling, not a floor.** Wrong contract drove violations up but not to 1.000 — DeepSeek occasionally resisted obviously-wrong contract content. This is interesting but not load-bearing for the gate.
- ⊥ **Unanswered:** whether the abstention-behavior gain (AMBIG Δ=+0.900) persists when contracts are AUTOMATICALLY extracted (not oracle-written). The whole CBT extraction stack hinges on this.

## 7. Process

⊢ **Backend:** DeepSeek `/chat/completions` over `urllib.request`, credentials from repo-root `.env` (gitignored). Determinism via explicit `temperature=0`. 600 calls / 0 errors / ~10.5 min wall.

⊢ **Discipline:** frozen-hash assert on every run, smoke-before-full, N=3 reps, co-headline metrics (V × CW × Abstain × cell-σ), gate floor Δ ≥ 0.05 absolute, raw-number verdict re-confirmation, S→T / L→G / Δe→∫de avoidance.

⊢ **Reproducibility:** `data/oracle_payoff/{items.jsonl, frozen_items_hash.json, raw_runs.jsonl, eval_results.json}` + `scripts/oracle_payoff.py` + `.env` env vars.

---

*Bounded to `deepseek-chat` × SI symbol/unit 50-item micro-domain. CBT-v1 stays gated pending WP-ST-6.1 counterfactual-contract test. ⊢ Sign-off requested from David.*
