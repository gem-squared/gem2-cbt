# WP-ST-15: Contract-conditioned Payoff on WP-14 Seed — Results

**Subject:** local Qwen (qwen2.5:32b-instruct-q8_0), temp=0, Ollama backend
**Corpus:** WP-14 SQuAD-v2 token-grounding seed (frozen hash `bdc404d760819e19`)
**Sample:** 400 items (200 answerable + 200 unanswerable), 3 conditions, 1 rep, 1200 records total, 0 errors
**Verifier:** `scripts/contract_schema.py::verify` (CW+CD converged; token-grounding ¬B — anti-fabrication only; **coverage_FN=100%** — no role-swap / negation / semantic-error detection)
**Pre-registered gate (per WP CONTRACT.B):** Δ ≥ 0.05 absolute — LOCKED in code before aggregation
**Headline regime:** `unanswerable` — the only regime with interpretable headroom (see U6 correction below)

---

## Violation rate by condition × regime (verifier)

| Condition   | V(answerable) | n_viol / n | V(unanswerable) | n_viol / n |
|---|---|---|---|---|
| PLAIN_NAKED | 0.240 | 48/200 | 0.575 | 115/200 |
| PLAIN_FAIR  | 0.440 | 88/200 | 0.190 | 38/200 |
| **CONTRACT**| **0.200** | 40/200 | **0.035** | 7/200 |

---

## `gate_contract_payoff` per regime — U6 corrected framing

### Unanswerable (HEADLINE) → **PASS** — real, red-team-surviving payoff

| Metric | Value |
|---|---|
| Δ(CONTRACT − PLAIN_FAIR) | **−0.155** |
| Δ(CONTRACT − PLAIN_NAKED) | −0.540 |
| Δ(PLAIN_FAIR − PLAIN_NAKED) | −0.385 (fair prompt captures most of the abstain gain) |
| paired Cohen's d (C vs F) | −0.41 (moderate effect) |

Contract-conditioning drives Qwen fabrication from **19.0% (fair prompt) → 3.5% (contract)** — a 5.4× compression, well above the 0.05 pre-registered floor. Both fair and contract crush naked (57.5% → 19% and 57.5% → 3.5%). The contract's marginal edge over a fair prompt is what the WP was designed to test — and it survives.

### Answerable → **UNINTERPRETABLE** (NOT a pass) — verifier-artifact regime

⊬ On answerable rows, the verifier's token-grounding ¬B is triggered by Qwen paraphrases that introduce non-source content tokens (typical culprits: `downfall`, `fulfilled`, `contributed`, `teaches`, `briefly`, `original`, and other reasonable-but-non-source connectives). These "violations" are **paraphrase false-positives** of the verifier, not real hallucinations. The regime cannot answer whether the contract genuinely reduces fabrication on answerable content; a stricter verifier (semantic grounding, not token grounding) would be required.

Numeric pattern on answerable: PLAIN_FAIR = 44.0% > PLAIN_NAKED = 24.0% > CONTRACT = 20.0%. PLAIN_FAIR being WORSE than PLAIN_NAKED is diagnostic of the verifier interacting with prompt verbosity: the "answer using only info in the passage" framing invites longer paraphrased responses, which introduce more non-source tokens. CONTRACT's constraint on content-tokens pulls Qwen back toward extractive spans. **This is not a payoff claim — it is a description of a verifier-artifact regime.**

**U6 correction (2026-07-01):** the U5 "both regimes PASS" language was wrong. Only the unanswerable regime is interpretable. Answerable is `UNINTERPRETABLE` — the verifier's coverage limits (token-grounding, coverage_FN=100% on semantic errors) mean an answerable "violation" cannot be attributed to Qwen fabrication vs verifier strictness.

---

## Vocabulary-independent real-hallucination re-score (U6)

**Concern to defuse:** the verifier's `must_abstain` check accepts a specific `abstain_markers` list ("not stated", "does not", etc.). Reasonable question: does CONTRACT win only because it mentions those markers to Qwen, who parrots them? Without the markers, would "real" hallucination rate track differently?

**Test:** rescore unanswerable rows with a **broad, vocabulary-independent refusal detector** (13 regex patterns for negation / uncertainty families that don't overlap with the verifier's marker list). A response is a "real hallucination" iff it contains NO refusal signal from this broader family. Reproducible via `python3 scripts/payoff_squad.py --rescore`.

| Condition | Verifier V (unans) | Real-hall V | Δ (broad − verifier) |
|---|---|---|---|
| PLAIN_NAKED | 0.575 | **0.570** | −0.5 pp |
| PLAIN_FAIR  | 0.190 | **0.175** | −1.5 pp |
| CONTRACT    | 0.035 | **0.035** | 0.0 pp  |

Real-hallucination tracks the verifier's `must_abstain` rate **within ~1.5 pp on all three conditions**. The payoff pattern is preserved:

- Without contract: **57%** real hallucinations
- Fair prompt (broad instruction, no specific markers): **17.5%** (fair prompt captures 39.5 pp of the reduction — most of it)
- **Contract (structured ⟨A,F,B,P,¬B⟩ pack): 3.5%** (contract adds another 14 pp on top, 5× further compression)

**Verdict:** the CONTRACT win over PLAIN_FAIR on unanswerable is **real hallucination reduction, not marker-vocabulary artifact**. The verifier's specific-marker check and a broader refusal-family detector agree on the ranking and on the magnitude.

---

## Bounding — what this result IS and IS NOT

### What it is

- ⊢ First contract-conditioning payoff that survives a red-team pass: pre-registered floor met, baseline-fairness confirmed (fair prompt captures most of the gain, contract adds a real 14 pp on top), vocabulary independence checked.
- ⊢ Real hallucination reduction on the unanswerable regime — Qwen fabricates specific answers 57% of the time without any instruction, 18% with a fair abstain-permitting instruction, and 3% when given the structured contract pack.
- ⊢ Deterministic verifier — no LLM in the eval loop. Reproducible end-to-end from `LLM_BACKEND=local python3 scripts/payoff_squad.py --run` and `--aggregate` and `--rescore`.

### What it is NOT (⊬ / ⊥ boundaries)

- ⊬ Not a payoff claim on answerable content. That regime is uninterpretable under this verifier.
- ⊬ Not a proof that contracts help with semantic errors. The verifier's token-grounding ¬B is anti-fabrication-only — `coverage_FN=100%` on role-swap / negation / temporal / synonym errors (documented at WP-14 U2). Anything that reuses source tokens in wrong roles will pass verification whether it's fabricated or not.
- ⊬ Not model-general. Single subject: `qwen2.5:32b-instruct-q8_0` at temp=0 via Ollama. Different families / sizes / decoding temps unknown.
- ⊬ Not trained-extractor. This is **prompt-level conditioning** — Qwen reads the ⟨A,F,B,P,¬B⟩ pack as system-user text and follows it. A learned extractor (PreCon LLM) that must PRODUCE contracts from raw NL is a separate WP.
- ⊬ Effect size is MODERATE, not strong (|d| ≈ 0.41 on the headline contrast). Population-mean shift is stable and above floor; per-item paired variance is non-trivial.
- ⊥ Payoff outside the abstain contract shape (facts-only content-grounding, complex reasoning, multi-hop, ambiguity resolution) is not established here.

### What it does

- **De-risks the pipeline.** WP-14 seeded the shape; WP-15 shows the shape produces real value when the model is prompted with it. The abstain contract is now a "known-good" configuration to build learned extraction against.
- **Sets the direction for WP-16**: build teacher-extracted contract training data so PreCon can produce these contracts from raw NL, not just consume templated ones.

---

## Explicit note on WP-15's role in the roadmap

WP-15 was a **value-check DETOUR**, NOT PreCon training data. Its outputs at `data/contract_squad_payoff/raw_runs.jsonl` are audit artifacts (local-only, gitignored) — they answer "does the WP-14 contract shape produce measurable payoff at prompt-time?" and nothing more. PreCon training data lives in a separate WP-16 workflow (teacher-extracted contracts, self-verified, corpus-diverse).

---

*Subject: local Qwen (qwen2.5:32b-instruct-q8_0) | 400 items × 3 conds × 1 rep | seed=42 | gate Δ≥0.05 (per WP CONTRACT) | corrected 2026-07-01 U6 | See papers/claim_payoff_squad.md for the bounded claim.*
