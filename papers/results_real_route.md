# WP-ST-8: Real-NL routing — Results

**N items:** 1992  **Domain:** WSD sense-selection (lemma_overlap held out for regime split)  
**Encoder:** sentence-transformers/all-MiniLM-L6-v2  
**Pre-registered floors:** Δrecall@1 ≥ 0.02, Cohen's d ≥ 0.3, Δabstain-F1 ≥ 0.03

---
## Per-router summary (mean ± std across seeds)

| Router | runs | recall@1 | clear-rec@1 | ambig-rec@1 | abstain_f1 |
|---|---|---|---|---|---|
| softmax_raw | 5 | 0.7036±0.0089 | 0.7007±0.0185 | 0.7045±0.0124 | 0.8552±0.0062 |
| softmax_sszes | 5 | 0.3532±0.0096 | 0.2359±0.0194 | 0.3928±0.0083 | 0.8575±0.0050 |
| twofeature (det) | 1 | 0.3348±0.0000 | 0.2266±0.0000 | 0.3723±0.0000 | 0.8522±0.0000 |
| hpic_complex (det) | 1 | 0.3348±0.0000 | 0.2266±0.0000 | 0.3723±0.0000 | 0.8556±0.0000 |
| mfs (det) | 1 | 0.7013±0.0000 | 0.6992±0.0000 | 0.7020±0.0000 | nan±nan |
| keyword (det) | 1 | 0.2932±0.0000 | 0.1719±0.0000 | 0.3351±0.0000 | nan±nan |
| tfidf (det) | 1 | 0.2575±0.0000 | 0.1973±0.0000 | 0.2784±0.0000 | nan±nan |

*⚠std=0 = stochastic router seeds produced zero variance → vacuous protocol (WP-5 lesson).*
*abstain_f1=nan = abstain signal has zero variance (constant signal) → oracle-F1 is a sort-stability artifact, not real abstain ability. mfs/keyword/tfidf have no abstain mechanism in this implementation.*

---
## U5B Probe — per-slice rec@1 (does hpic_complex/twofeature win on ANY slice?)

Targeted check: WP-4 said the spread/conflict feature survived ONLY in the conflict regime. Before rejecting the spread feature on real NL, verify it does not win on the conflict slice.

| slice | n | softmax_raw | softmax_sszes | twofeature | hpic_complex | mfs | keyword | tfidf |
|---|---|---|---|---|---|---|---|---|
| all | 1992 | 0.7028 | 0.3609 | 0.3348 | **0.3348** | 0.7013 | 0.2932 | 0.2575 |
| clear | 512 | 0.7012 | 0.2422 | 0.2266 | **0.2266** | 0.6992 | 0.1719 | 0.1973 |
| ambiguous | 1480 | 0.7034 | 0.4020 | 0.3723 | **0.3723** | 0.7020 | 0.3351 | 0.2784 |
| low_margin | 902 | 0.6929 | 0.4080 | 0.3825 | **0.3825** | 0.6929 | 0.4180 | 0.3137 |
| high_margin | 331 | 0.6949 | 0.2296 | 0.2175 | **0.2175** | 0.6918 | 0.1722 | 0.1782 |
| K_2_3 | 498 | 0.8635 | 0.6305 | 0.5663 | **0.5663** | 0.8614 | 0.4940 | 0.4317 |
| K_4_6 | 498 | 0.7249 | 0.3373 | 0.3092 | **0.3092** | 0.7269 | 0.2932 | 0.2610 |
| K_7_10 | 498 | 0.5622 | 0.2028 | 0.1988 | **0.1988** | 0.5582 | 0.1747 | 0.1627 |
| K_11+ | 498 | 0.6606 | 0.2731 | 0.2651 | **0.2651** | 0.6586 | 0.2108 | 0.1747 |
| high_K | 820 | 0.6085 | 0.2402 | 0.2329 | **0.2329** | 0.6049 | 0.1927 | 0.1610 |

**Δrec@1 (hpic_complex − softmax_raw) per slice:**
- all: Δ = -0.3680  (softmax wins)
- clear: Δ = -0.4746  (softmax wins)
- ambiguous: Δ = -0.3311  (softmax wins)
- low_margin: Δ = -0.3104  (softmax wins)
- high_margin: Δ = -0.4773  (softmax wins)
- K_2_3: Δ = -0.2972  (softmax wins)
- K_4_6: Δ = -0.4157  (softmax wins)
- K_7_10: Δ = -0.3635  (softmax wins)
- K_11+: Δ = -0.3956  (softmax wins)
- high_K: Δ = -0.3756  (softmax wins)

**No slice — including the conflict-regime slices (ambiguous, low_margin, high_K) where WP-4 said the spread feature lives — shows hpic_complex/twofeature beating softmax_raw.** The spread/conflict feature's survival in WP-4 does NOT replicate on real NL routing.

### Abstain-signal sanity (probe 2)

| router | min | max | std | n_unique | degenerate? |
|---|---|---|---|---|---|
| softmax_raw | 0.2181 | 0.9948 | 0.1318 | 1990 | no |
| softmax_sszes | 0.0091 | 1 | 0.2737 | 1987 | no |
| twofeature | 0.7500 | 1.92e+09 | 1.672e+08 | 1607 | no |
| hpic_complex | 0.2000 | 1.04 | 0.2108 | 1601 | no |
| mfs | 0.0000 | 0 | 0 | 1 | **YES (constant)** |
| keyword | 0.0000 | 0 | 0 | 1 | **YES (constant)** |
| tfidf | 0.0000 | 0 | 0 | 1 | **YES (constant)** |

*mfs / keyword / tfidf have no abstain mechanism (constant signal) → their previously-reported abstain_f1=1.000 was a sort-stability artifact, NOW marked as nan in per-router results.*

*twofeature's max=1.92e9 reflects es/(|ss|+ε) blowing up when ss≈0; the oracle threshold sweep is monotone-invariant so this does not break the metric, but it is the heaviest-tailed signal in the comparison.*

---
## Decomposition (paired deltas)

| A | B | metric | mean Δ (A−B) | Cohen's d | seeds A wins | label |
|---|---|---|---|---|---|---|
| hpic_complex | softmax_raw | recall@1 | -0.3688 | -41.40 | 0/5 | HEADLINE — complex vs strong baseline |
| hpic_complex | softmax_sszes | recall@1 | -0.0184 | -1.91 | 0/5 | PROOF TEST — complex vs softmax over (ss,es) |
| hpic_complex | twofeature | recall@1 | +0.0000 | n/a | 0/1 | complex-only delta (invertibility ⇒ Δ≈0) |
| softmax_sszes | softmax_raw | recall@1 | -0.3504 | -31.34 | 0/5 | does the spread feature help? |
| hpic_complex | softmax_raw | abstain_f1_oracle | +0.0004 | 0.06 | 3/5 | HEADLINE abstain — complex vs softmax |
| hpic_complex | softmax_sszes | abstain_f1_oracle | -0.0020 | -0.39 | 2/5 | PROOF TEST abstain |
| hpic_complex | twofeature | abstain_f1_oracle | +0.0034 | n/a | 1/1 | complex-only abstain delta |

---
## gate_real_route verdict

- **HEADLINE recall@1 (hpic_complex vs softmax_raw):** FAIL
   - Δ=-0.3688; d=-41.40; wins 0/5
- **HEADLINE abstain-F1 (hpic_complex vs softmax_raw):** FAIL
   - Δ=0.0004; d=0.06; wins 3/5

**gate_real_route: FAIL — cosmetic CONFIRMED on real NL**

**Proof test** (hpic_complex vs softmax_sszes, recall@1): Δ=-0.0184, d=-1.91. Invertibility predicts Δ≈0.

**Complex-only delta** (hpic_complex vs twofeature, recall@1): Δ=+0.0000. Proof predicts Δ≈0 (identical math).

