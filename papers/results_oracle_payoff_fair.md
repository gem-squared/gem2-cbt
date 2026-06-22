# WP-ST-6A: FAIR Oracle-Contract Payoff — Results

**Model:** deepseek-chat  **Temp:** 0  **Items:** 52 (15 clear-memorized + 25 clear-counterfactual + 12 ambiguous-fair)  
**Reps per cell:** 3  **Total records:** 936  **Frozen items hash:** see `data/oracle_payoff_fair/frozen_items_hash.json`  

**Pre-registered gate floors (locked before aggregation):** Δ ≥ 0.05 absolute, |d| ≥ 0.5 paired Cohen's d.  
**Primary regime for gates:** `clear-counterfactual` (memory-independent — only regime where the knowledge contract carries information the SI prior cannot supply).

---
## Violation rate by condition × regime

Cell value = mean per-item rep-mean violation_rate. cell-σ = std across items of the per-item rep-mean.

| Condition | V(clear-memorized) | V(clear-counterfactual) | V(ambiguous-fair) | σ(clear-memorized) | σ(clear-counterfactual) | σ(ambiguous-fair) |
|---|---|---|---|---|---|---|
| A | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B_FAIR | 0.000 | 1.000 | 0.083 | 0.000 | 0.000 | 0.289 |
| B_GAG | 0.000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.522 |
| C_KNOW | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C_INST | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Cp | 0.600 | 1.000 | 0.583 | 0.507 | 0.000 | 0.515 |

---
## Counterfactual subtype split

| Condition | V(cf-novel) | V(cf-adversarial) | σ(cf-novel) | σ(cf-adversarial) |
|---|---|---|---|---|
| A | 1.000 | 1.000 | 0.000 | 0.000 |
| B_FAIR | 1.000 | 1.000 | 0.000 | 0.000 |
| B_GAG | 1.000 | 1.000 | 0.000 | 0.000 |
| C_KNOW | 0.000 | 0.000 | 0.000 | 0.000 |
| C_INST | 0.000 | 0.000 | 0.000 | 0.000 |
| Cp | 1.000 | 1.000 | 0.000 | 0.000 |

---
## Paired contrasts (item-paired)

Δ = V(A) − V(B). Negative Δ means A is BETTER (fewer violations).
d = paired Cohen's d; '—' = deterministic (std of per-item differences = 0).

### clear-memorized

| Contrast | Δ | d | n |
|---|---|---|---|
| C_KNOW_vs_B_FAIR | +0.000 | — | 15 |
| C_KNOW_vs_C_INST | +0.000 | — | 15 |
| C_KNOW_vs_Cp | -0.600 | -1.18 | 15 |
| C_INST_vs_B_FAIR | +0.000 | — | 15 |
| B_GAG_vs_B_FAIR | +0.000 | — | 15 |
| B_FAIR_vs_A | +0.000 | — | 15 |

### clear-counterfactual

| Contrast | Δ | d | n |
|---|---|---|---|
| C_KNOW_vs_B_FAIR | -1.000 | — | 25 |
| C_KNOW_vs_C_INST | +0.000 | — | 25 |
| C_KNOW_vs_Cp | -1.000 | — | 25 |
| C_INST_vs_B_FAIR | -1.000 | — | 25 |
| B_GAG_vs_B_FAIR | +0.000 | — | 25 |
| B_FAIR_vs_A | +0.000 | — | 25 |

### ambiguous-fair

| Contrast | Δ | d | n |
|---|---|---|---|
| C_KNOW_vs_B_FAIR | -0.083 | -0.29 | 12 |
| C_KNOW_vs_C_INST | +0.000 | — | 12 |
| C_KNOW_vs_Cp | -0.583 | -1.13 | 12 |
| C_INST_vs_B_FAIR | -0.083 | -0.29 | 12 |
| B_GAG_vs_B_FAIR | +0.417 | +0.81 | 12 |
| B_FAIR_vs_A | +0.083 | +0.29 | 12 |

---
## SI-prior leak diagnostic (cf-adversarial)

Leak = response mentions the canonical SI unit for the IN-CONTEXT QUANTITY (i.e. the model fell back to memorized prior instead of following the in-context binding).

| Condition | n records | n leaks | leak_rate |
|---|---|---|---|
| A | 39 | 36 | 0.923 |
| B_FAIR | 39 | 36 | 0.923 |
| B_GAG | 39 | 36 | 0.923 |
| C_KNOW | 39 | 0 | 0.000 |
| C_INST | 39 | 0 | 0.000 |
| Cp | 39 | 0 | 0.000 |

---
## Decisive fair gates

### gate_payoff_fair: **PASS**

> V(C_KNOW) − V(B_FAIR) = -1.000 on clear-counterfactual; need ≤ -0.05

### gate_not_tautology: **PASS**

> |Δ(C_KNOW − C_INST)|=0.000 (need < 0.05); Δ(C_KNOW − B_FAIR)=-1.000 (need ≤ -0.05); Δ(C_INST − B_FAIR)=-1.000 (need ≤ -0.05)

### gate_uses_contract: **PASS**

> V(C_KNOW) − V(Cp) = -1.000 on clear-counterfactual; need ≤ -0.05

---
## Diagnostics

- **Gag effect on ambiguous** (V(B_GAG) − V(B_FAIR) on ambiguous-fair): Δ=+0.417, d=+0.81 — quantifies how much of WP-6's 0.900 ambiguous-violation was the gag, vs a real baseline weakness.
- **Counterfactual headroom** (V(B_FAIR) − V(C_KNOW) on clear-counterfactual): Δ=+1.000 — payoff space where memory cannot help. Positive Δ = real headroom; near zero = no payoff.

---
*WP-ST-6A | deepseek-chat | 52 items × 6 conditions × 3 reps | floors locked Δ≥0.05, |d|≥0.5 | See papers/claim_oracle_payoff_fair.md for the bounded claim.*
