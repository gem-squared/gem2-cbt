# WP-ST-3 Ecological Validity — Results v3
**Encoder:** all-MiniLM-L6-v2 (384-dim, frozen)  
**Splits:** v3-hard (canonical hash `25fc21e245581f64`) · v3-easy (canonical hash `fba8e3f6236ae5a8`)  
**Seeds:** 0–9 · **Epochs:** 5 · **Architecture:** MLP 1536→512→256→1  

---
## v3_hard

**MFS baseline (majority):** 0.504  

### Accuracy (mean ± std, 10 seeds)

| Config | Mean Acc | ±Std | AUROC mean |
|---|---|---|---|
| heuristic_baseline | 0.504 | ±0.000 | N/A |
| text_only_nn | 0.145 | ±0.000 | N/A |
| gloss_similarity_baseline | 0.245 | ±0.000 | N/A |
| text_contract | 0.319 | ±0.020 | 0.704 |
| cbt | 0.332 | ±0.031 | 0.706 |
| easy_random_contract | 0.141 | ±0.002 | 0.500 |
| hard_same_lemma_random | 0.206 | ±0.003 | 0.505 |
| contract_only | 0.117 | ±0.019 | 0.400 |
| target_word_only | 0.225 | ±0.058 | 0.555 |

### Gate Verdicts

| Gate | Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|---|
**NOTE: g6 (cbt > text_only) is NOT reported for v3-hard — text_only_nn is zero-shot-degenerate on unseen lemmas (by-construction failure, not ecological signal).**

| Gate | Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|---|
| g3a: cbt > easy_random | 10/10 | +0.190 | 6.07 | PASS ✓ |
| **g3b (DECISIVE): cbt > hard_same_lemma** | 10/10 | +0.125 | 4.08 | PASS ✓ |
| parsimony: cbt > gloss_sim | 10/10 | +0.086 | 2.79 | PASS ✓ |
| parsimony: cbt > target_word_only | 10/10 | +0.106 | 1.56 | PASS ✓ |

---
## v3_easy

**MFS baseline (majority):** 0.533  

### Accuracy (mean ± std, 10 seeds)

| Config | Mean Acc | ±Std | AUROC mean |
|---|---|---|---|
| heuristic_baseline | 0.533 | ±0.000 | N/A |
| text_only_nn | 0.603 | ±0.000 | N/A |
| gloss_similarity_baseline | 0.321 | ±0.000 | N/A |
| text_contract | 0.576 | ±0.003 | 0.808 |
| cbt | 0.576 | ±0.002 | 0.808 |
| easy_random_contract | 0.182 | ±0.003 | 0.501 |
| hard_same_lemma_random | 0.292 | ±0.003 | 0.528 |
| contract_only | 0.508 | ±0.005 | 0.753 |
| target_word_only | 0.578 | ±0.003 | 0.807 |

### Gate Verdicts

| Gate | Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|---|
| g6: cbt > text_only_nn (weak frozen baseline) | 0/10 | -0.027 | -11.01 | FAIL ✗ |
| g3a: cbt > easy_random | 10/10 | +0.393 | 77.34 | PASS ✓ |
| g3b: cbt > hard_same_lemma | 10/10 | +0.284 | 65.72 | PASS ✓ |
| parsimony: cbt > gloss_sim | 10/10 | +0.255 | 105.13 | PASS ✓ |

---
## Summary

**v3-hard g3b (DECISIVE):** cbt > hard_same_lemma → PASS ✓ (Δ+0.125)  
**v3-hard parsimony (cbt > gloss_sim):** PASS ✓ (Δ+0.086)  
**v3-easy g6 (cbt > text_only_nn):** FAIL ✗ (Δ-0.027)  
**v3-easy parsimony (cbt > target_word_only):** FAIL ✗  

**text_only_nn on v3-hard: NOT reported as g6 — zero-shot degenerate by lemma-split construction.**  

*See papers/claim_v3.md for bounded claim + CBT-v1 gate decision.*
