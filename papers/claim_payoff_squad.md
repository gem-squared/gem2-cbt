# Claim: Contract-Conditioned Payoff on WP-14 Seed (WP-ST-15)

**Subject:** local Qwen (`qwen2.5:32b-instruct-q8_0`), temp=0, Ollama
**Corpus:** WP-14 SQuAD-v2 token-grounding seed (frozen hash `bdc404d760819e19`)
**Sample:** 400 items (200 answerable + 200 unanswerable) × 3 conditions × 1 rep = 1200 records, 0 errors
**Verifier:** `scripts/contract_schema.py::verify` (token-grounding ¬B; `coverage_FN=100%` documented ceiling — anti-fabrication only)
**Pre-registered gate (per WP CONTRACT.B):** `Δ(V(CONTRACT) − V(PLAIN_FAIR)) ≤ −0.05`

---

## The narrow, real claim

⊢ **On unanswerable SQuAD-v2 items under prompt-level contract conditioning, structured `⟨A, F, B, P, ¬B⟩` conditioning drives Qwen-32B fabrication from 19.0% (fair abstain-permitting prompt) → 3.5% (contract), a 5.4× compression, over 200 paired items** — pre-registered gate PASS at Δ=−0.155.

⊢ **The reduction is real hallucination reduction, not marker-vocabulary artifact.** A vocabulary-independent broad-refusal detector (13 regex patterns disjoint from the verifier's specific marker list) tracks the verifier's `must_abstain` rate within ~1.5 pp on all three conditions (NAKED 0.575→0.570; FAIR 0.190→0.175; CONTRACT 0.035→0.035). Payoff pattern preserved.

⊢ **The fair prompt captures most of the abstain gain (57% → 17.5%, Δ = −40 pp).** CONTRACT's contribution is the marginal 14 pp on top (17.5% → 3.5%). This is the honest signature of a real payoff over a fair baseline — not a strawman-baseline win.

⊢ **This is the first contract-conditioning payoff in the CBT program that survives a red-team pass.** WP-6A's oracle-injection payoff was retracted for confounds; WP-10/11's saturated-tie diagnostic left the pipeline unproven. WP-15 is narrow but clean.

## Bounding — the ⊬ / ⊥ boundaries

⊬ **Not answerable.** The answerable regime is UNINTERPRETABLE under this verifier. Qwen paraphrases introduce non-source content tokens (typical culprits: `downfall`, `fulfilled`, `contributed`, `briefly`, `original`) which trigger the token-grounding ¬B. These are paraphrase false-positives of the verifier, not real hallucinations. Any answerable "payoff" number is a verifier interaction with prompt verbosity, NOT a claim about Qwen's fabrication behavior on answerable content.

⊬ **Not semantic-error coverage.** The verifier's `¬B` is anti-fabrication-only. `coverage_FN=100%` on role-swap / negation / temporal / relation / synonym-morphology errors was documented at WP-14 U2 and re-surfaced at WP-14 U6/U7. Any output that reuses source tokens in the wrong semantic roles will pass verification regardless of correctness. **The payoff is in the abstain contract shape only.**

⊬ **Not trained-extractor.** This is **prompt-level conditioning** — Qwen reads the rendered ⟨A, F, B, P, ¬B⟩ pack in the user turn and follows it. A learned extractor that PRODUCES contracts from raw NL (PreCon-LLM) is a separate WP-16 workflow. Any claim about "trained CBT extractor payoff" is out of scope here.

⊬ **Not model-general.** Single subject: `qwen2.5:32b-instruct-q8_0` at temp=0. Different families, sizes, or decoding regimes are not tested. WP-6A's WP-11 replication showed the pattern held between DeepSeek and Qwen, but neither of those was on a corpus-scale seed with real natural language.

⊬ **Effect size is MODERATE.** Paired Cohen's d ≈ −0.41 (moderate, not strong). Population-mean shift is stable and above the pre-registered floor; per-item paired variance is non-trivial (some items flip between conditions).

⊥ **Payoff on other contract shapes** (facts-only content grounding, complex reasoning, multi-hop, ambiguity resolution) — not established. WP-15 tested only the two shapes WP-14 built (answerable content-grounding + unanswerable abstain), and only unanswerable was interpretable.

⊥ **Payoff on non-SQuAD domains** (long-context, medical, legal, code, dialogue) — unknown. SQuAD is short-context Wikipedia extractive QA; the transfer characteristics are open.

## What this de-risks

⊢ The WP-14 abstain-contract SHAPE is a "known-good" configuration for the downstream pipeline: at prompt time, on 200 held-out unanswerable questions, contract-conditioned Qwen holds to the abstain contract at 96.5% under a deterministic check. This makes the abstain contract a defensible target for PreCon extraction (WP-16).

⊢ The verifier is deterministic and cheap (regex over the response — no LLM in eval). Any downstream training loop that consumes contract-conditioned outputs can use `verify()` as an inexpensive reward / filter with predictable characteristics.

## Architecture decision — narrow "GO" with explicit next step

⊢ **PROCEED to WP-16 (teacher-extracted contract training data)** — the abstain-contract payoff is real, defensible against a red-team, and reproducible via `scripts/payoff_squad.py --run --aggregate --rescore`. WP-16 builds the corpus that would let PreCon-LLM learn to EMIT contracts (rather than consume templated ones), specifically because WP-14's mechanically-templated `build_contract()` cannot teach general extraction — only teacher-extracted contracts can.

⊢ **CBT-v1 boundary-gated attention remains GATED** — this WP speaks to prompt-time contract conditioning, not to the CBT-v1 training-time attention mechanism. WP-ST-2/3 g4/g5/g6 FAILs still bind.

⊢ **Explicit scope note: WP-15 was a VALUE-CHECK DETOUR, NOT PreCon training data.** The `data/contract_squad_payoff/raw_runs.jsonl` outputs are audit artifacts — David's eyeball on Qwen's raw generations against 400 pre-registered prompts. They are NOT the pairs a training run would consume. PreCon training data lives in WP-16 (teacher-extracted, self-verified).

## Reproducibility

```bash
# 1. WP-14 seed frozen at bdc404d760819e19 (assert_frozen_hash gate)
python3 -c "from cbt.fingerprint import assert_frozen_hash; \
            print(assert_frozen_hash('data/contract_squad'))"

# 2. Full run (400 items × 3 conds × 1 rep; ~85 min on Qwen-32B-Q8)
LLM_BACKEND=local python3 scripts/payoff_squad.py --run

# 3. Aggregate + pre-registered gates (Δ≥0.05 per regime)
python3 scripts/payoff_squad.py --aggregate

# 4. Vocabulary-independent real-hallucination re-score
python3 scripts/payoff_squad.py --rescore
```

Anti-tuning: prompts locked at `PREREGISTRATION_UTC = "2026-07-01T02:50:00Z"` before the first Qwen call; SHARED_FORMAT identical across all three conditions; verifier semantics mirrored in CONTRACT prose so retention is not adapter-mismatch.

---

*WP-ST-15 | qwen2.5:32b-instruct-q8_0 | 400 items × 3 conds × 1 rep | seed=42 | narrow-but-real | scope: prompt-level conditioning + abstain contract shape only | corrected under U6 red-team framing 2026-07-01 | See papers/results_payoff_squad.md for the full numbers.*
