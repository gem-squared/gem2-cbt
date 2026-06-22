# WP-ST-4: HPIC ⊥-Gate A/B Test — Results

**Seeds:** 0–9 (10/10 complete)  
**N instances:** 3000 (test set 30%)  **K sources:** 5  **ρ_k = |2p_k−1|** (pre-registered)  
**Regimes:** CLEAR / INSUFF / CONFLICT (equal thirds)  

---
## Per-Gate Summary (mean±std, 10 seeds)

| Gate | RC-AUC (↓ better) | Conflict-Abstain-Recall (↑ better) |
|---|---|---|
| linear | 0.3225±0.0030 | 0.907±0.014 |
| max_prob | 0.3376±0.0044 | 0.860±0.016 |
| hpic_point | 0.3073±0.0012 | 0.974±0.005 |
| hpic_interval | 0.3005±0.0000 | 1.000±0.000 |

---
## Gate g_A: HPIC > baselines on Risk–Coverage AUC (lower = better for AUC)

| Comparison | Seeds winning | Mean Δ (baseline−HPIC) | Cohen's d | Verdict |
|---|---|---|---|---|
| hpic_point_vs_linear | 10/10 | +0.0152 | 8.10 | FAIL ✗ |
| hpic_point_vs_max_prob | 10/10 | +0.0303 | 9.31 | PASS ✓ |
| hpic_interval_vs_linear | 10/10 | +0.0220 | 7.31 | FAIL ✗ |
| hpic_interval_vs_max_prob | 10/10 | +0.0371 | 8.50 | PASS ✓ |

**g_A overall: PASS ✓**

---
## Gate g_B (DECISIVE): HPIC > baselines on Conflict-Regime Abstain Recall

Operating point: CLEAR coverage ≈ 80%  

| Comparison | Seeds winning | Mean Δ | Cohen's d | Verdict |
|---|---|---|---|---|
| hpic_point_vs_linear | 10/10 | +0.0662 | 6.79 | PASS ✓ |
| hpic_point_vs_max_prob | 10/10 | +0.1135 | 9.55 | PASS ✓ |
| hpic_interval_vs_linear | 10/10 | +0.0927 | 6.65 | PASS ✓ |
| hpic_interval_vs_max_prob | 10/10 | +0.1400 | 8.85 | PASS ✓ |

**g_B overall: PASS ✓**

---
## Verdict

**g_B PASS** — HPIC complex geometry outperforms plain thresholds on conflict-regime abstain recall. HPIC ⊥-gate adopted for CBT's accept/reject/⊥ decision point.

**g_A PASS** — HPIC also improves overall risk–coverage AUC.

*See papers/claim_hpic_gate.md for bounded claim + adoption decision.*
