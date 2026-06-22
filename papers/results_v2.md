# CBT v2 Ablation Results — Full 10-Seed Sweep

**Frozen v2 hash:** `4c44d14d16757d4d`  |  **Seeds:** 0–9  |  **Epochs:** 30

**Dataset:** 7637 train / 1363 test (9000 total)  |  **Majority ref:** 0.502


## Table 1: Overall Metrics (mean ± std, 10 seeds)

| Config | acc | F1 | AUROC | unsafe_accept | over_reject |
|---|---|---|---|---|---|
| text_only | 0.715±0.007 | 0.737±0.032 | 0.834±0.006 | 0.386±0.115 | 0.184±0.121 |
| text_level | 0.714±0.007 | 0.747±0.024 | 0.835±0.006 | 0.427±0.093 | 0.144±0.093 |
| text_contract | 0.888±0.012 | 0.893±0.011 | 0.972±0.003 | 0.162±0.032 | 0.062±0.028 |
| cbt_v0 | 0.889±0.016 | 0.896±0.015 | 0.973±0.005 | 0.178±0.030 | 0.043±0.024 |
| random_contract | 0.713±0.006 | 0.737±0.032 | 0.835±0.006 | 0.397±0.124 | 0.177±0.122 |
| contract_only | 0.623±0.006 | 0.628±0.017 | 0.694±0.004 | 0.395±0.058 | 0.359±0.051 |
| heuristic_baseline | 0.502±0.000 | 0.000±0.000 | — | — | — |

## Table 2: Per-Level Accuracy (mean ± std)

| Config | concept | context | task |
|---|---|---|---|
| text_only | 0.506±0.011 | 0.958±0.009 | 0.705±0.017 |
| text_level | 0.511±0.017 | 0.964±0.011 | 0.692±0.011 |
| text_contract | 0.998±0.003 | 0.955±0.012 | 0.713±0.030 |
| cbt_v0 | 0.999±0.002 | 0.957±0.013 | 0.715±0.043 |
| random_contract | 0.503±0.011 | 0.963±0.010 | 0.699±0.014 |
| contract_only | 0.469±0.010 | 0.713±0.008 | 0.698±0.009 |
| heuristic_baseline | 0.512±0.000 | 0.500±0.000 | 0.517±0.000 |

## Table 3: Per-Level F1 (mean ± std)

| Config | concept | context | task |
|---|---|---|---|
| text_only | 0.562±0.104 | 0.958±0.009 | 0.734±0.026 |
| text_level | 0.574±0.136 | 0.964±0.011 | 0.740±0.006 |
| text_contract | 0.998±0.003 | 0.954±0.013 | 0.746±0.028 |
| cbt_v0 | 0.999±0.002 | 0.956±0.014 | 0.758±0.034 |
| random_contract | 0.532±0.176 | 0.963±0.010 | 0.738±0.014 |
| contract_only | 0.444±0.024 | 0.686±0.025 | 0.740±0.013 |
| heuristic_baseline | 0.677±0.000 | 0.667±0.000 | 0.000±0.000 |

## Gate Verdicts

### g1: heuristic_baseline ≤ 0.65 per level → PASS ✓
  - concept: 0.512 → PASS
  - context: 0.500 → PASS
  - task: 0.517 → PASS

### g2: cbt_v0 > text_level (contract isolated from level/capacity) → PASS ✓
  - concept_acc: 10/10 seeds win, mean Δ=+0.488 → PASS
  - overall_f1: 10/10 seeds win, mean Δ=+0.149 → PASS
  - auroc: 10/10 seeds win, mean Δ=+0.138 → PASS

### g3: cbt_v0 > random_contract (contract content matters) → PASS ✓
  - concept_acc: 10/10 seeds win, mean Δ=+0.495 → PASS
  - overall_f1: 10/10 seeds win, mean Δ=+0.159 → PASS
  - auroc: 10/10 seeds win, mean Δ=+0.139 → PASS

### g4: cbt_v0 > text_only per-level acc → FAIL ✗
  - concept: 10/10 seeds win, mean Δ=+0.493 → PASS
  - context: 5/10 seeds win, mean Δ=-0.002 → FAIL
  - task: 5/10 seeds win, mean Δ=+0.010 → FAIL

### g5: unsafe_accept_rate reduction + AUROC increase (cbt_v0 vs text_only) → FAIL ✗
  - unsafe_accept/concept: 10/10 seeds win, mean Δ=-0.668 → PASS
  - unsafe_accept/context: 1/10 seeds win, mean Δ=+0.007 → FAIL
  - unsafe_accept/task: 3/10 seeds win, mean Δ=+0.048 → FAIL
  - auroc: 10/10 seeds win, mean Δ=+0.139 → PASS

## Summary

- g1 (heuristic baseline): PASS ✓
- g2 (contract vs level): PASS ✓
- g3 (contract content): PASS ✓
- g4 (cbt_v0 vs text_only): FAIL ✗
- g5 (unsafe_accept + AUROC): FAIL ✗

**Decisive gate g2 ∧ g3: PASS — contract content ISOLATED as load-bearing**

**Frozen v2 hash:** `4c44d14d16757d4d` | **Seeds:** 10 | **Epochs:** 30

## Interpretation Notes

- `text_contract` concept_acc ≈ `cbt_v0` concept_acc → contract embedding alone drives concept-level discrimination; level embedding adds little for concept
- `contract_only` (text zeroed) concept_acc << `cbt_v0` → contract ID alone insufficient; text is still needed (contract embedding ≠ raw contract ID)
- Context/task level: cbt_v0 ≈ text_contract ≈ text_level — all near-ceiling, semantic boundaries are surface-expressible (inherent to English)
- `random_contract` concept_acc ≈ `text_only` concept_acc → wrong contract degrades to text-only level (contract content matters, not just the embedding slot)

