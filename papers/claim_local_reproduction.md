# Claim — WP-ST-11 Local-backend reproduction of WP-10 on Qwen2.5-32B-Instruct-Q8

**Date:** 2026-06-19  
**Scope:** Reproduce WP-10's CER/ECE scoped microcell findings on a second subject (Qwen2.5-32B-Instruct, Q8 GGUF via Ollama), changing **ONE variable only — the subject LLM**. Items, conditions, checker, contract scaffold, and pre-registered gate floors all stay frozen at WP-10's values.

---

## Bounded scope (what these claims apply to — and only to)

- **Subject A:** `deepseek-chat` (DeepSeek API, near-deterministic at temp=0 — observed 5.4% byte-level slip on WP-10)
- **Subject B:** `qwen2.5-32b-instruct-q8_0` (Ollama local, greedy: `temperature=0, top_p=1.0, top_k=1, seed=42, num_ctx=8192`; observed 0.00% slip)
- **Items:** WP-10's frozen 37 clear-counterfactual SI-binding items (hash `c52feb2b07876905`)
  - 25 in-domain (reused verbatim from WP-6A: 12 novel + 13 adversarial)
  - 12 held-out (6 novel + 6 adversarial, disjoint adversarial SI units)
- **Reps:** 3 per (item, condition) cell
- **Conditions (4, frozen at WP-10 U1):**
  - `B_FAIR` — fair SI-expert prompt, no facts (floor)
  - `PLAINFACTS` — same facts as prose, no scaffold (RAG-like, anti-RAG-confound control)
  - `C_PACK_LEARNED` — same facts inside the WP-6A scaffold, produced by a LEARNED ECE per subject
  - `C_KNOW_ORACLE` — same facts inside the same scaffold, produced by WP-6A's hand-written oracle (ceiling)
- **Pre-registered gate floors (UNCHANGED across subjects, anti-cherry-pick):**
  Δ ≥ 0.05, |Cohen d| ≥ 0.5 (or `absolute_unanimity`)
- **New gate added in WP-11:** `gate_local_reproduction` with three sub-checks (see below).

Out of scope: CBT-v1 boundary-gated attention (still GATED); ambiguous regime; other models, scaffolds, or domains; harder-prompt-scope (→ WP-ST-12); white-box / decode-time (→ Phase B).

---

## Per-condition violation rate (overall, n=37; two-subject, NO silent merge)

| Condition | V (deepseek-chat) | V (qwen2.5-32b-instruct-q8_0) | Δ (qwen − deepseek) |
|---|---:|---:|---:|
| `B_FAIR` | 99.1% | 97.3% | −1.8 pp |
| `PLAINFACTS` | 0.0% | 0.0% | +0.0 pp |
| `C_PACK_LEARNED` | 0.0% | 0.0% | +0.0 pp |
| `C_KNOW_ORACLE` | 0.0% | 0.0% | +0.0 pp |

Rows are subject-labeled per David's anti-silent-merge directive. The same table broken down per-regime (in-domain / held-out) is in `data/cer_ece_qwen/cross_model_compare.md`.

---

## Gate verdicts

### `gate_local_reproduction` (new at WP-11) → **PASS**

Composed of three sub-checks:

**Sub-check (a) — extraction yield + adversarial leak** (Qwen track only)

| Metric | Value | Floor | Verdict |
|---|---:|---:|:---:|
| ECE yield | 100.0% | ≥ 90.0% | ✓ |
| Adversarial leak_rate | 0.0% (0/19) | ≤ 10.0% | ✓ |
| **Sub-check (a) verdict** | | | **PASS** |

Loose adversarial-leak floor of 10% was set BEFORE running on Qwen (DeepSeek had observed 2%; floor was set looser for Qwen-32B as anti-knife-edge protection). Qwen came in at 0.0%, matching DeepSeek exactly.

**Sub-check (b) — `gate_learned_payoff` reproduces on Qwen**

| Subject | Δ (PACK − B_FAIR) | Cohen d | Effect | Verdict |
|---|---:|---:|:---:|:---:|
| `deepseek-chat` | −0.991 | −18.08 | very_large | **PASS** |
| `qwen2.5-32b-instruct-q8_0` | −0.973 | −5.92 | very_large | **PASS** |
| **Sub-check (b) verdict** | | | | **PASS** (Qwen verdict matches DeepSeek) |

Per-split on Qwen: overall (Δ=−0.973, d=−5.92), in_domain (Δ=−0.978, d=−7.50), held_out (Δ=−0.963, d=−3.66). All three splits PASS the pre-registered floors.

**Sub-check (c) — `gate_structure_vs_facts` diagnostic (NOT a gate component)**

| Subject | V(PLAINFACTS) | V(C_PACK_LEARNED) | structure_headroom | Verdict | Note |
|---|---:|---:|---:|:---:|---|
| `deepseek-chat` | 0.0% | 0.0% | +0.000 | FAIL (saturated tie) | WP-10 finding |
| `qwen2.5-32b-instruct-q8_0` | 0.0% | 0.0% | +0.000 | FAIL (saturated tie) | new at WP-11 |

The (c) sub-check was a DIAGNOSTIC question (does the weaker model break PLAINFACTS saturation and expose structure-vs-facts headroom?). It does NOT gate `gate_local_reproduction` overall. **Answer: saturation is NOT broken** — V(PLAINFACTS) = 0.0% on Qwen too, across overall + in_domain + held_out. structure_headroom = +0.000 on every regime.

**`gate_local_reproduction` overall = PASS** = (a) PASS ∧ (b) PASS.

---

## What we can honestly say (bounded ⊢ claim)

> ⊢ **WP-10's `gate_learned_payoff` reproduces on Qwen2.5-32B-Instruct-Q8 local: a learned automatic extractor drives counterfactual violation from ~100% (no facts) to 0% (with the contract pack), on both in-domain AND held-out items. The single-model caveat substantially addressed across two tested model families (deepseek-chat and qwen2.5-32b-instruct-q8_0), not resolved universally.**

> ⊢ **WP-10's `gate_structure_vs_facts` SATURATED TIE reading also holds on Qwen: V(PLAINFACTS) = V(C_PACK_LEARNED) = 0.0% on both tested subjects (deepseek-chat and qwen2.5-32b-instruct-q8_0). At this single-binding short-context scope, plain in-context facts saturate the violation metric on both subjects; the structured contract pack offers no demonstrated marginal value here. This still does NOT show that contract structure is cosmetic in general — the test has no headroom at this scope.**

> ⊢ **Side finding: local greedy decoding (Ollama, `temperature=0, top_k=1, seed=42`) was 100% byte-deterministic across 111 byte-identical-input pairs on Qwen. The 5.4% near-determinism slip WP-10 observed was a DeepSeek-API property, not a property of greedy decoding generally.**

That is the entire scope of what WP-11 establishes. Nothing more.

---

## What we will NOT do (refused overclaims; explicit `⊬`)

- ⊬ Claim "WP-10 finding holds universally across all LLMs." Two model families is not a universe.
- ⊬ Claim "Contract structure is cosmetic in general." The saturated tie shows no demonstrated marginal value at this scope, not absence of value everywhere.
- ⊬ Claim "Scoped CBT = RAG." Same refusal as at WP-10 sign-off — saturation is a scope property, not an architecture verdict.
- ⊬ Treat a future WP-12 harder-prompt-scope PASS as proof of CBT architectural novelty. Per the **red-team note** below.
- ⊬ Pivot architecture (decode-time / verifier-gated) on this reproduction alone — that pivot is still ordered AFTER WP-12.

---

## Red-team note (baked in per David's WP-11 directive — S→T guard)

> **A future WP-12 harder-prompt-scope PASS (multi-binding / distractors / long-context / conflicting constraints) would be evidence of formatting / RAG value at the prompt-injection layer, NOT proof of CBT architectural novelty. The architectural-novelty question lives at white-box / decode-time (Phase B), not at prompt-scope.**
>
> Prompt-scope wins demonstrate "structure helps the LLM use facts that were already going to be provided in the system message." That is RAG with extra formatting. Architectural novelty would require a behavior that prompt-supply mode cannot achieve — most plausibly at decode-time (constrain generation, verifier-gated sampling) rather than at facts-supply.
>
> Reading any prompt-scope PASS as architectural-novelty is a S→T overclaim (state-as-trait) — taking a contextual finding ("structure helped at this prompt-scope") and asserting it as a permanent architectural property of CBT. This claim doc refuses that move pre-emptively so it cannot be made by later readers.

---

## Honest caveats

1. **B_FAIR std difference**: Qwen B_FAIR has std 16.4% across items (vs DeepSeek 5.5%). Qwen apparently gets a small number of cf items right without facts (likely lucky SI-prior coincidences on specific bindings). Mean is still 97.3% (vs 99.1%) — within the model-variance band, gate verdict unaffected.

2. **Determinism inheritance**: WP-11's 0.00% byte-determinism slip applies to greedy decoding via Ollama at the listed options on this machine. DeepSeek's 5.4% slip applies to its API at temp=0 on the WP-10 run. The future-proof move is to keep greedy + local as the default subject for any work that depends on byte-determinism (e.g., the C_PACK == C_KNOW sanity equality).

3. **Loose adversarial-leak floor (10% vs DeepSeek's observed 2%)**: set looser BEFORE running on Qwen because Qwen-32B was expected to be weaker; Qwen came in at 0.0% so the loose floor never gated the result. Disclosure noted; no post-hoc tightening.

4. **B_FAIR rate caveat from WP-10 carries over**: the checker keyword-overlap edge case on item 224 (where "ampere per metre" matches substring "ampere") still affects DeepSeek's B_FAIR rate (99.1% vs true 100%). Qwen item 224 was also checked under the same logic; its overall 97.3% B_FAIR includes its own checker-keyword artifacts plus genuine lucky-prior wins. Gate verdicts unaffected.

5. **Router not stress-tested** (inherited from WP-10): scoped domain has 2 binding-types; routing is trivial. Honesty note in `data/cer_ece/router_report.json` still applies — wired, not stress-tested.

---

## Diagnostic — C_PACK_LEARNED vs C_KNOW_ORACLE (extraction-loss cost)

| Subject | V(C_PACK_LEARNED) | V(C_KNOW_ORACLE) | Δ | Note |
|---|---:|---:|---:|---|
| `deepseek-chat` | 0.0% | 0.0% | +0.000 | extraction fidelity 1.000 → byte-identical inputs at temp=0 |
| `qwen2.5-32b-instruct-q8_0` | 0.0% | 0.0% | +0.000 | extraction fidelity 1.000 → byte-identical inputs; 111/111 byte-identical responses confirmed |

Extraction-quality floor question remains untestable at this scope on both subjects (both extractors hit fidelity 1.000). Future scopes with noisier inputs are needed to draw the payoff-vs-fidelity curve.

---

## Architecture decision

**Adopt the WP-10 finding as ⊢ across two model families (deepseek-chat, qwen2.5-32b-instruct-q8_0) at this scope.** Do NOT scale the current pack format to more domains; do NOT pivot to decode-time yet. The next legitimate step is:

### Next WP — WP-ST-12 (proposed): harder non-saturating prompt-scope on Qwen-local

Falsifier-spirit successor. Designed so PLAINFACTS does NOT saturate at 0% violation on the verified-deterministic subject (Qwen-local greedy). Stressors (any or all; pre-register before any run):

1. **Multi-binding** — 3–5 concurrent quantity → unit redefinitions in one context
2. **Distractors** — irrelevant binding facts interleaved with the load-bearing one
3. **Long context** — binding embedded hundreds-to-thousands of tokens from the question
4. **Conflicting constraints** — two partial-conflict bindings with an implied precedence order

Falsifier:
- `C_PACK_LEARNED` beats `PLAINFACTS` on any axis by Δ ≥ pre-registered floor → `gate_structure_vs_facts` PASS on a non-saturating scope → structured contracts have demonstrated marginal value somewhere in the prompt-injection space (still RAG-flavored per the red-team note; NOT architectural novelty).
- All axes also saturate or tie → structure is cosmetic across the explored prompt-scope axes; the decode-time / verifier-gated pivot becomes the strong default for Phase B.

### Future pivot — decode-time / verifier-gated contracts (Phase B)

Still valid. Still ordered AFTER WP-12. Rationale unchanged from WP-10 sign-off: avoid pivoting architecture on a single-scope saturated tie. Two saturated subjects strengthen the "scope is the wrong axis to test architecture on" reading; they do NOT yet license skipping WP-12.

---

## CBT-v1 status

CBT-v1 boundary-gated attention REMAINS GATED. WP-11 did not test it (out of scope). The CBT-v1 question is independent of the WP-10 / WP-11 / WP-12 extractor-stack thread.

---

## Sign-off

- **Planner (Alchy + Kritik):** bounded claim accepted with the David sign-off redirect on the two anti-universal-claim sentences (line 1: "substantially addressed across two tested model families ... not resolved universally"; line 2: "on both tested subjects (deepseek-chat and qwen2.5-32b-instruct-q8_0)").
- **Engineer (Gineer):** all 7 units executed (U7 the bounded claim itself); all artifacts persisted (`cbt/llm_backend.py`, `scripts/cer_ece_cell.py`, `data/cer_ece_qwen/*`, this claim, `papers/results_local_reproduction.md`).
- **Reviewers (CW + CD):** independently flagged the same anti-overclaim sentences; CW also extended the fix to sentence 2; both edits applied. The synthesis was the strongest version of both.
- **Human (David):** explicit handoff directive followed: ONE variable changed (model only); non-destructive backend selector; greedy temp=0 determinism verified (0% slip on Qwen vs 5.4% on DeepSeek); separate-track data discipline (no silent merge); harder-scope WP-12 stays the next step; decode-time/Phase B stays ordered AFTER WP-12; S→T guard baked in.

## References

- `data/cer_ece_qwen/items.jsonl` (= WP-10's frozen items; hash `c52feb2b07876905`).
- `data/cer_ece_qwen/raw_runs.jsonl` — 444 Qwen cells (37 × 4 × 3).
- `data/cer_ece_qwen/eval_results.json` — Qwen per-condition rates, paired Δ, gate verdicts.
- `data/cer_ece_qwen/extracted_facts.jsonl` + `extraction_fidelity.json` — yield 1.000, adversarial leak 0.000.
- `data/cer_ece_qwen/cross_model_compare.{md, json}` — side-by-side compare with backend labels (anti-silent-merge).
- `data/cer_ece_qwen/determinism_report.json` — 4 probes × 5 reps, slip 0.000.
- `data/cer_ece_qwen/manifest_*.json` — backend + model + quant + ollama_version + git_commit per unit.
- `data/cer_ece/eval_results.json` — DeepSeek baseline (WP-10 archived).
- `papers/claim_cer_ece.md` — WP-10 claim (with SATURATED TIE reframe; what this WP reproduces).
- `cbt/llm_backend.py` — backend selector (deepseek default, local opt-in, never logs api_key).
- `scripts/cer_ece_cell.py` — full backend-aware pipeline (WP-6A/WP-7 imports UNCHANGED).
- `.gem-squared/archive/WP-ST-10.md` — parent WP, replication target.
- `.gem-squared/archive/WP-ST-6A.md` — items + checker + oracle pack source (UNTOUCHED).
- `.gem-squared/archive/WP-ST-7.md` — concept_ce.py source for the LEARNED ECE pattern (UNTOUCHED).
