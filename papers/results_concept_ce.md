# WP-ST-7: Concept Contract-Extractor — Results

**Main domain:** 50 items (40 CLEAR + 10 AMBIGUOUS)
**Counter-probe (U5B):** 49 items (24 novel + 25 adversarial)
**Methods:** rule_lookup (ceiling), embedding_nn (sentence-transformers/all-MiniLM-L6-v2), llm_prompt (deepseek-chat), majority (floor)
**Pre-registered floors:** margin=0.1, cw≤0.1, ar≥0.6, τ=0.5

---
## Main-domain per-method summary

| Method | runs | extraction_accuracy | confident_wrong_rate | abstain_precision | abstain_recall |
|---|---|---|---|---|---|
| rule_lookup | 1 | 1.000 (det) | 0.000 (det) | 1.000 (det) | 1.000 (det) |
| embedding_nn ⚠std=0 | 3 | 0.100±0.000 | 0.080±0.000 | 0.143±0.000 | 0.600±0.000 |
| llm_prompt ⚠std=0 | 3 | 1.000±0.000 | 0.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| majority | 1 | 0.100 (det) | 0.000 (det) | 0.000 (det) | 0.000 (det) |

*⚠std=0 = stochastic seeds produced zero variance — protocol vacuity (WP-5 lesson).*

---
## Counter-probe (U5B): memorization-vs-extraction de-confound

Novel-binding items use made-up vocabulary defined in-prompt only.
Adversarial-redefinition items use REAL SI surfaces re-bound to a different quantity by an in-prompt contract — correct extraction = follow the contract, override SI prior.

| Method | runs | RAW acc | NORM acc | RAW cw | NORM cw | abstain_recall | mem_revert_rate |
|---|---|---|---|---|---|---|---|
| rule_lookup_probe (det) | 1 | 1.000±0.000 | 1.000±0.000 | 0.000±0.000 | 0.000±0.000 | 1.000±0.000 | 0.000±0.000 |
| llm_prompt_probe | 3 | 0.800±0.000 | 0.975±0.000 | 0.163±0.000 | 0.020±0.000 | 1.000±0.000 | 0.050±0.000 |

*NORM = sense-string normalizer collapses display-form synonyms (e.g. 'electric_inductance' == 'inductance'). RAW vs NORM gap reflects the LLM's tendency to emit display-form names, NOT an extraction failure.*

**mem_revert_rate** = fraction of adversarial-CLEAR items where LLM returned the ORIGINAL SI prior despite the contract redefinition. Low = LLM follows contract over memorized prior (the CBT need).

---
## gate_ce verdict (strict: main + probe both required for llm_prompt)

Main ceiling = 1.000; Main floor = 0.100; Probe ceiling = 1.000

| Method | MAIN: acc/cw/ar/beats_floor | PROBE: acc/cw/ar/memrev | OVERALL |
|---|---|---|---|
| **embedding_nn** | ✗0.100 / ✓0.080 / ✓0.600 / ✗ | n/a | **FAIL** |
| **llm_prompt** | ✓1.000 / ✓0.000 / ✓1.000 / ✓ | ✓0.975 / ✓0.020 / ✓1.000 / ✓0.050 | **PASS** |

**gate_ce overall: PASS ✓**

Winning non-lookup CE(s): **llm_prompt**

