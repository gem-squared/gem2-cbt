# Claim: FAIR Oracle-Contract Payoff (WP-ST-6A)

**Model:** deepseek-chat | **Temp:** 0 | **Items:** 52 (15 clear-memorized + 25 clear-counterfactual + 12 ambiguous-fair) | **Reps:** 3 | **Total records:** 936
**Pre-registered floors:** Δ ≥ 0.05 absolute, |d| ≥ 0.5 paired (locked in code before aggregation)
**Primary regime for gates:** `clear-counterfactual` (memory-independent — the only regime where the knowledge contract carries information the SI prior cannot supply)

---

## Question

With FAIR controls, does conditioning a strong LLM on a **knowledge-only** task/context/concept contract reduce boundary violations vs a **fair** strong prompt — **non-tautologically** (without the abstain-imperative carrying the win) and **where memory cannot help** (counterfactual content the SI prior does not contain)?

## Verdict — three gates PASS

| Gate | Δ on clear-counterfactual | Verdict |
|---|---|---|
| `gate_payoff_fair`     V(C_KNOW) < V(B_FAIR) by floor | **−1.000** (floor −0.05) | **PASS ✓** |
| `gate_not_tautology`   V(C_KNOW) ≈ V(C_INST) AND both < V(B_FAIR) | \|Δ(K−I)\|=0.000 ∧ Δ(K−B)=Δ(I−B)=−1.000 | **PASS ✓** |
| `gate_uses_contract`   V(C_KNOW) < V(Cp) by floor | **−1.000** (floor −0.05) | **PASS ✓** |

## Headline numbers

| Cond | V(clear-memorized) | V(clear-counterfactual) | V(ambiguous-fair) |
|---|---|---|---|
| A naked | 0.000 | 1.000 | 0.000 |
| B_FAIR | 0.000 | 1.000 | 0.083 |
| B_GAG | 0.000 | 1.000 | 0.500 |
| **C_KNOW** | **0.000** | **0.000** | **0.000** |
| C_INST | 0.000 | 0.000 | 0.000 |
| Cp | 0.600 | 1.000 | 0.583 |

## Diagnostic findings

- **SI-prior leak on cf-adversarial:** A/B_FAIR/B_GAG = **92.3% leak** (36/39); C_KNOW/C_INST/Cp = **0% leak**. The model defaults to memorized SI prior 92% of the time on adversarial bindings when no contract is present, and follows any contract (right or wrong) when one is.
- **Gag effect on ambiguous:** Δ(B_GAG − B_FAIR) on ambiguous-fair = **+0.417, d=+0.81** → **~42 pp of WP-6's 0.900 ambiguous-violation was a gag artifact**, not a real baseline weakness. The fair baseline (gag removed) qualifies 91.7% of the time on ambiguous items.
- **Counterfactual headroom:** V(B_FAIR) − V(C_KNOW) on cf = **+1.000** (maximum possible) — the entire space is payoff; without the contract the model gets none right, with the contract it gets all right.
- **C_KNOW ≡ C_INST exactly** (V=0.000 in every regime; 0% leak in both) → the abstain-imperative the WP-6 ambiguous contract carried is **not** what drove the win. The payoff is contract-content, not contract-instruction.

## Bounded claim

⊢ **Conditioning the inference LLM on a knowledge-only contract reduces boundary violations vs a fair strong prompt by Δ=−1.000 on memory-independent content** (52 items × deepseek-chat × temp=0; primary regime = clear-counterfactual; n_items=25; pre-registered floor crossed).
⊢ **The payoff is not the abstain-imperative.** C_KNOW (facts-only contract, no behavioral instruction) ties C_INST (facts + imperative) exactly at V=0.000 across all regimes; both crush B_FAIR by Δ=−1.000 on cf.
⊢ **The model uses contract content, not contract structure.** Cp (wrong-content contract, identical scaffold) drives V to 1.000 on cf and 0.600 on memorized; right-content contract drives V to 0.000. Δ(C_KNOW − Cp) = −1.000.
⊢ **WP-6's ambiguous payoff was substantially a gag artifact, not a real baseline weakness.** B_FAIR (gag removed) qualifies 91.7% of the time; the WP-6 0.900 ambiguous-violation gap shrinks to 0.083 once the rule-3 Yes/No gag is lifted.
⊨ **Mechanism = contract injection passes structured facts the model parrots.** On cf-adversarial, the SI prior is *active* (92% leak without contract); ANY contract suppresses the leak to 0% — including the wrong contract. The model attends to the system-prompt contract over its prior whenever a contract is present. This is **basic input-following**, not complex reasoning.
⊬ **No payoff demonstrated on clear-memorized content.** B_FAIR=0.000 and C_KNOW=0.000 both saturate → headroom is zero by construction (WP-6's saturation caveat carries forward). UNINTERPRETABLE, not PAYOFF=0. Basis: same domain (curated SI quantities) DeepSeek has memorized; no test could distinguish payoff from saturation here without a different micro-domain.
⊬ **Negligible payoff on ambiguous-fair.** Δ(C_KNOW − B_FAIR) = −0.083, **below the 0.05 floor**. Once the gag is removed, the fair baseline already captures qualification on ambiguous SI questions. Basis: 12-item sample (small); could shift with a denser ambiguous-item population, but the current finding is "no significant gap."

## Architecture decision

⊢ **PROCEED with the CBT extractor stack** — building the contract-extraction machinery (CER routers / Concept-CE / Context-CE / Task-CE / Binder / Verifier) is **JUSTIFIED** on memory-independent content. The payoff is real, non-tautological, and content-driven on the only regime where the test had headroom.
⊢ **SCOPE the build to memory-independent territory.** The payoff is on content **outside** the model's pretrained prior — novel bindings, adversarial redefinitions, in-context-defined entities. For memorized factual content, no payoff is demonstrable on this evidence; engineering effort should not target that regime.
⊢ **Extractor design target: produce parrot-ready contracts.** The mechanism is "model reads contract → applies binding"; the extractor must emit structured fact packs the model can READ + APPLY directly. Subtle reasoning loads on the contract are not required and not tested.
⊨ **CBT-v1 boundary-gated attention stays GATED.** WP-ST-6A speaks to the LLM-conditioning leg of the architecture (does the CONTRACT help at inference time?). It says nothing about the boundary-attention training mechanism that defines CBT-v1; that hypothesis remains separately gated by WP-ST-2/3 (where g4/g5 FAIL and the v3-hard g6 FAILed). The "build the extractor stack" verdict here is a contract-conditioning verdict, not a CBT-v1 verdict.

## Caveats — what this claim does NOT establish

- ⊥ Generalization beyond deepseek-chat. Different families (GPT-4, Claude, open-weights) may have different parrot-vs-reason ratios on counterfactual content. Replication needed before a cross-model claim.
- ⊥ Generalization beyond the SI-unit micro-domain. The payoff was demonstrated on a single checkable domain (52 hand-curated items). The extractor stack's value depends on whether contract-injection generalizes to messier real-world domains where the "right contract" is not as cleanly definable.
- ⊥ Generalization to non-oracle contracts. C_KNOW is hand-written, perfect, frozen. A learned extractor will produce IMPERFECT contracts; the payoff curve as contract quality degrades is the WP that should come next. (Open question: at what contract-quality floor does the payoff disappear?)
- ⊥ Generalization beyond the parrot mechanism. Cf-novel and cf-adversarial test "follow the in-context binding"; they do NOT test "use the contract to disambiguate genuinely hard ambiguous cases" or "use the contract to reason." The ambiguous-fair regime (where reasoning would matter) showed no significant payoff.

## Historical reference — WP-ST-6 footprint preserved

WP-ST-6 (the parent WP) is **preserved untouched** as the public record of how three confounds fool a payoff test:

1. **Gagged baseline** — WP-6's B forbade qualification on ambiguous items via "Yes/No only" rule 3 → B's 0.900 ambig-violation was a gag artifact. WP-6A confirmed ~42 pp of the gap was the gag.
2. **Tautological oracle** — WP-6's ambiguous oracle contract explicitly injected "acknowledge ambiguity / do not commit" → C's 0.000 ambig-violation was instruction-following. WP-6A confirmed by holding facts constant and stripping the imperative: C_KNOW ≡ C_INST exactly.
3. **Memorized clear regime** — WP-6's 40 CLEAR items were memorized SI facts → B saturated to 0 with no headroom for C to demonstrate payoff. WP-6A confirmed by adding novel + adversarial bindings; on those, B saturates at 1.000 and C wins by Δ=−1.000.

WP-6's `gate_uses_contract` survivor (Cp=0.875 wrong-contract drives violations confidently wrong) was the one clean finding; WP-6A re-confirmed it across all three regimes (Cp=1.000 on cf, 0.583 on ambig, 0.600 on memorized).

The discovery that confounds are this easy to introduce in a payoff test is itself public value — preserved as `scripts/oracle_payoff.py` + `papers/{claim,results}_oracle_payoff.md` + `.gem-squared/work-plan/WP-ST-6.md`.

## Next

→ **Plan the first CBT microcell** (next /plan-work) — design a contract-extractor that produces a parrot-ready structured fact pack for a single quantity-binding task. Scope: memory-independent content. Target: drive cf-adversarial violation rate from B_FAIR's 1.000 toward C_KNOW's 0.000 using a LEARNED contract (not oracle-written). Establish the contract-quality → payoff curve.

---

*WP-ST-6A | deepseek-chat | 52 items × 6 conditions × 3 reps | floors locked Δ≥0.05, |d|≥0.5 | three fair gates PASS | architecture decision: PROCEED extractor stack, bounded to memory-independent content | CBT-v1 stays GATED separately*
