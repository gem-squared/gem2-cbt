# WP-ST-6: Oracle-Contract Payoff — Results

**Model:** deepseek-chat  **Items:** 150 (120 CLEAR + 30 AMBIGUOUS)  
**Frozen hash:** see frozen_items_hash.json  

---
## Co-Headline Metrics by Condition (CLEAR + AMBIGUOUS)

Headline ≠ raw accuracy. We pair **violation_rate** with **confident-wrong** (violation without abstention) and **abstention** (correct-on-ambiguous).

| Condition | n | err | V_clear | cell-σ_clear | CW_clear | V_ambig | Abstain_ambig |
|---|---|---|---|---|---|---|---|
| A — naked LLM | 150 | 0 | 0.025 | 0.156 | 0.025 | 0.000 | 1.000 |
| B — strong prompt | 150 | 0 | 0.000 | 0.000 | 0.000 | 0.900 | 0.100 |
| C — oracle contract | 150 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| C' — wrong contract | 150 | 0 | 0.875 | 0.331 | 0.875 | 0.500 | 0.500 |

*cell-σ = std-dev of per-item rep-mean violation rate (rep stability diagnostic; 0 → perfectly stable across reps).*

---
## Gate Verdicts (CLEAR violation_rate, floor Δ≥0.05 absolute)

| Gate | Δ (absolute) | Verdict |
|---|---|---|
| gate_payoff: C < B (oracle beats strong prompt) | +0.000 | FAIL ✗ |
| gate_uses_contract: C < C' (content matters) | +0.875 | PASS ✓ |
| headroom (B vs A): strong prompt improvement | +0.025 | small |

---
## Architecture Decision

**gate_uses_contract PASS, gate_payoff FAIL.** Content is used (wrong contract increases violations) but the oracle doesn't beat the strong prompt by the floor (0.05). Marginal payoff: strong prompt already captures most benefit. CBT justified for content sensitivity but payoff over a strong prompt is limited on this domain.

*WP-ST-6 | deepseek-chat | 150 items | floor Δ=0.05 | See papers/claim_oracle_payoff.md*
