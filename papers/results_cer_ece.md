# Results — WP-ST-10 CER/ECE first scoped microcell

**Model:** `deepseek-chat` (temperature=0)  
**Items:** 37 clear-counterfactual  (in-domain 25 + held-out 12)  
**Conditions:** B_FAIR / PLAINFACTS / C_PACK_LEARNED / C_KNOW_ORACLE

## Per-condition violation rate (mean ± std across items)

### Split: `all`
| Condition | n items | mean V | std across items |
|---|---:|---:|---:|
| `B_FAIR` | 37 | 99.1% | 5.5% |
| `PLAINFACTS` | 37 | 0.0% | 0.0% |
| `C_PACK_LEARNED` | 37 | 0.0% | 0.0% |
| `C_KNOW_ORACLE` | 37 | 0.0% | 0.0% |

### Split: `in_domain`
| Condition | n items | mean V | std across items |
|---|---:|---:|---:|
| `B_FAIR` | 25 | 98.7% | 6.7% |
| `PLAINFACTS` | 25 | 0.0% | 0.0% |
| `C_PACK_LEARNED` | 25 | 0.0% | 0.0% |
| `C_KNOW_ORACLE` | 25 | 0.0% | 0.0% |

### Split: `held_out`
| Condition | n items | mean V | std across items |
|---|---:|---:|---:|
| `B_FAIR` | 12 | 100.0% | 0.0% |
| `PLAINFACTS` | 12 | 0.0% | 0.0% |
| `C_PACK_LEARNED` | 12 | 0.0% | 0.0% |
| `C_KNOW_ORACLE` | 12 | 0.0% | 0.0% |

## DECISIVE gates (pre-registered floors locked at U1)

Floors: gate_learned_payoff = Δ ≥ 0.05 (PACK lower than B_FAIR), gate_structure_vs_facts = Δ ≥ 0.05 (PACK lower than PLAINFACTS), effect size |d| ≥ 0.5 OR absolute_unanimity.

### gate_learned_payoff (V(C_PACK_LEARNED) < V(B_FAIR))
| Split | n | Δ mean | std | Cohen d | floor pass | effect pass | verdict |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| `all` | 37 | -0.991 | +0.055 | -18.08 | ✓ | ✓ | **PASS** |
| `in_domain` | 25 | -0.987 | +0.067 | -14.80 | ✓ | ✓ | **PASS** |
| `held_out` | 12 | -1.000 | +0.000 | — | ✓ | ✓ | **PASS** |

### gate_structure_vs_facts (V(C_PACK_LEARNED) < V(PLAINFACTS))
| Split | n | Δ mean | std | Cohen d | floor pass | effect pass | verdict |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| `all` | 37 | +0.000 | +0.000 | — | ✗ | ✗ | **FAIL** |
| `in_domain` | 25 | +0.000 | +0.000 | — | ✗ | ✗ | **FAIL** |
| `held_out` | 12 | +0.000 | +0.000 | — | ✗ | ✗ | **FAIL** |

## Diagnostic — C_PACK_LEARNED vs C_KNOW_ORACLE

- n = 37, Δ mean = +0.000, effect = `absolute_tie`
- Interpretation: Expected ≈0 (byte-identical inputs at temp=0). Non-zero gap signals extraction loss or stochasticity.

## Rep-variance honesty audit

- cells with rep variance: 1 / 148
- deterministic cells:     147 / 148  (99.3%)

Deterministic ≠ vacuous: at temp=0, identical (system,user) inputs SHOULD yield identical responses, so per-cell std=0 is a structural property, not a bug. Effect strength uses `absolute_unanimity` labeling when paired-delta std=0 — avoids the WP-5 Cohen-d=∞ vacuity.

## Extraction quality

- ECE yield: 100.0%
- in_domain per-item exact-match: 1.0
- held_out per-item exact-match:  1.0
- Single data point — extraction is essentially perfect at this scope. The payoff-vs-fidelity curve cannot be drawn from one fidelity value. Future scopes with noisier inputs will sweep this axis.
