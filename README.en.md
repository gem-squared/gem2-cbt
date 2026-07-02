<!-- Language: English | Korean (default): README.md -->

# gem2-CBT — Contract-Bounded Transformer Research Log

> **Status: closed research record.**
> This repository is a completed public record of my attempt to build toward
> **Contract-Bounded Transformer (CBT)** systems — including the wrong turns,
> failed proxies, retractions, and the surviving mechanism. It concludes that **CBT as a new
> Transformer is unsupported by its own evidence**; the one result that survived red-teaming
> is a **bounded, contract-conditioned abstain mechanism**. This is not a finished model and
> not a proven new Transformer.

**License:** [CC-BY-4.0](LICENSE) | (c) 2026 David Seo / GEM².AI | Korean (default): [README.md](README.md)

---

## Why This Exists

CBT started from a simple thesis:

```text
Before an LLM answers, it should know the contract that bounds the answer.
```

For this project, a real contract is not a tag or label. A contract is closer to:

```text
F: A -> B | P
```

where `P` is the condition that bounds how the transformation from input `A` to output `B`
is allowed to happen.

The long-term CBT target was a MoE-like contract architecture:

```text
Input corpus
  -> CER: Contract Extractor Router
  -> Task / Context / Concept CE or ECE modules
  -> Contract Pack
  -> Contract-conditioned inference
  -> Verifier / abstain / repair
```

The rebuild began at the first gate, **Task-CER**, and then tested the real contract object
directly (WP-13–18). Earlier experiments did not always test that object correctly; this
record keeps both the mistakes and the corrections.

---

## The Human-AI Story

This repository records not only the experimental results, but also how a human and AI
collaborators thought together and corrected failures.

The first evaluation phase was intentionally driven by AI collaborators (Claude Cowork,
Codex, and Claude Code). They were fast, rigorous, and useful, but they often evaluated the
wrong object. Several early tests treated labels, facts, WSD senses, or prompt scaffolds as
if they were CBT contracts. Those tests produced real experiments and useful negative
results, but they also pushed the project toward overly strong conclusions such as "CBT is
just structured RAG" before the real contract object had been tested.

One recurring observation from this journey is that AI systems are fragile in creative work.
When they are asked to plan and proceed autonomously, they can drift away from a newly
defined concept and fall back into familiar methods, even when the new concept has been
stated clearly.

During the project, my main role was to watch for that drift and keep forcing the AI
collaborators back to the newly established principles:

- a contract is not a label;
- Task / Context / Concept must be extracted as bounded contract pixels;
- CER is the first architectural gate;
- if CE cannot be built, CBT should be discarded;
- if a test targets a proxy instead of the real object, the result must be bounded or rejected.

That is why I keep the failed attempts in the public record. They are not marketing. They
are evidence of how easy it is to make a precise experiment answer the wrong question. The
result is the same whether the mistake comes from a human or from an AI system.

This is a single case study, not a universal claim about AI systems or human-AI research.

---

## What Survived

### 1. A contract-conditioned abstain mechanism reduces hallucination (bounded)

The strongest surviving result. On source-silent (unanswerable) questions, conditioning the
model on a contract with a deterministic **abstain** boundary reduced asserted-hallucination
substantially versus a *fair* prompt that also permits abstention — validated
vocabulary-independently:

```text
regime: unanswerable / source-silent questions
hallucination (fair prompt)      ~ 0.18
hallucination (contract-abstain) ~ 0.03
```

Bounded reading:

```text
Supplying a deterministic abstain boundary makes the model refuse when the source is silent,
beyond what a fair prompt achieves. Bounded to one model family, prompt-level, this regime.
```

This is a narrow reliability mechanism. It does **not** prove a new Transformer.

### 2. Contract content is behaviorally active

WP-6A showed that, on memory-independent counterfactual content, a knowledge-only contract
can reduce boundary violations versus a fair strong prompt:

```text
B_FAIR violation: 1.000
C_KNOW violation: 0.000
```

The win was not the earlier "abstain" imperative; WP-6A removed that tautology and still
observed the payoff in the counterfactual regime.

### 3. A learned/prompted extractor retained the payoff, across two models

WP-10 replaced the hand-written oracle pack with a learned/prompted extractor in the same
single-binding counterfactual scope; the extracted pack retained the payoff. WP-11 reproduced
it on a second subject model (`deepseek-chat`, `qwen2.5-32b-instruct-q8_0`). The single-model
caveat is substantially addressed across two tested families, not resolved universally.

### 4. The deterministic contract check works only where violations change tokens

A key structural finding from the per-level datasets (WP-16): a deterministic `¬B` check is
genuinely deterministic for levels whose violation *changes the surface tokens* —

```text
Task    (fabrication / abstain)   : deterministically checkable
Concept (wrong word-sense)        : deterministically checkable
Context (role-swap, same tokens)  : NOT deterministically checkable (graded only)
```

A role-swap ("the bag held the water" vs "the water held the bag") reuses the exact tokens,
so token-grounding cannot catch it. The contract-with-a-deterministic-boundary primitive is
strong for Task/Concept and only graded for Context.

---

## What Was Wrong or Retracted

These are kept because they matter.

- Early controls confused label shuffling with real negative controls.
- WSD was useful as a concept-disambiguation probe, but it was not a full Concept Contract.
- Several experiments treated tags or facts as contracts. That was not sufficient.
- A first oracle-payoff test was confounded by a gagged baseline, memorized domain, and
  behavior-injecting contract.
- **Complex HPIC** (*Hierarchical Phase-Interval Classifier*) was an attempt to reframe
  classification as direction on a complex plane. Each piece of evidence became an angle
  `θ=arccos(2p−1)`, and small relation-regions (`Sm`) contributed phasors summed as
  `Z = Σ ρ·exp(iθ)`; the sign of `Re(Z)` gave the decision, while proximity to the 90-degree
  uncertainty axis produced abstain/Unknown. It was first falsified as a Spaceship Titanic
  classifier, then re-tested inside CBT as a router / `⊥`-gate. In that CBT role, HPIC reduced
  to an invertible reparameterization of two real features, `Re(Z)=signed_strength` and
  `Im(Z)=evidence_spread`, and lost to plain softmax over raw features (three independent fronts).
- **The possibility-score density+distance geometry was cosmetic** (WP-18): density and
  distance were strongly collinear (corr ≈ −0.92), and combining them added no margin over a
  single feature. Adopt the simplest scorer, not the compound. This is the *fourth* time a
  "fancy geometry" reduced to a simple rule.
- A synthetic level-labeled substrate was **structurally saturated** for level detection
  (WP-17): its three levels were three different topics, so any scorer "won" by detecting
  topic, not level — invalid for the intended test, and retired.
- "CBT is structured RAG" was too strong as a general statement. In the tested simple scope,
  plain facts saturated. That is not the same as proving structure has no value everywhere.

---

## Conclusion

The CBT journey is closed as a record. Its own experiments settled the central question:

- **CBT as a new Transformer / new neural architecture: unsupported by this project's
  evidence.** Every novel differentiable/geometric mechanism reduced to a standard or graded
  one — the complex HPIC router (cosmetic, three fronts), the possibility-score density+distance
  geometry (cosmetic, reduced to a single feature), CBT-v1 boundary-gated attention (gated).
  This is a scoped result: *the mechanisms we tried* failed — not a proof that no such
  architecture could exist.
- **What survived is a bounded mechanism, not an architecture:** supplying a contract
  in-context — specifically a deterministic *abstain* boundary — reduces hallucination when
  the source is silent (bounded to one model, prompt-level); and the deterministic contract
  check is genuinely deterministic for some semantic levels (Task, Concept) and only graded
  for others (Context).

```text
The new-architecture thesis did not survive. A narrow, deterministic-boundary reliability
mechanism did. The durable asset is the audited falsification path itself.
```

---

## Work Package Ledger

| WP | Question | Current reading |
|----|----------|-----------------|
| 1 | Harden the first CBT signal | Confounded; control issue found |
| 2 | Is contract content behaviorally active? | Yes, bounded |
| 3 | Real-language WSD/ecological probe | Useful probe; not full Concept Contract |
| 4 | Complex boundary gate | Tied by plain 2-feature rule |
| 5 | Complex CER router | Failed vs softmax |
| 6 | First oracle payoff | Confounded; preserved as footprint |
| 6A | Fair oracle payoff | PASS on memory-independent counterfactual content |
| 7 | Concept extraction | LLM/prompt extractor passed a bounded synthetic probe; non-LLM extractor failed |
| 8 | Real-NL HPIC router falsifier | HPIC-complex rejected for routing |
| 9 | Public release preparation | Curated release surface, not raw main |
| 10 | Learned extractor microcell | Payoff retained; structure vs facts saturated |
| 11 | Local Qwen reproduction | Payoff reproduced across two tested subjects; saturation reproduced |
| 13 | C/C/T as the partition axis | Coherent on synthetic data but by-construction; inconclusive |
| 14 | Real contract seed from QA data | Token-grounding anti-fabrication extraction seed built |
| 15 | Contract-conditioned abstain payoff | **Real, bounded** — hallucination cut on source-silent questions (one model, prompt-level) |
| 16 | Per-level contract datasets | Task / Concept deterministically checkable; **Context graded only** |
| 17 | Possibility-score bake-off (synthetic) | Substrate structurally saturated; invalid — retired |
| 18 | Possibility-score geometry | density+distance **cosmetic** (= single feature); use the simplest scorer |
| — | CBT thesis | **CLOSED**: new-Transformer unsupported; surviving result = bounded contract-abstain |

---

## Reproducibility

Reproducibility of these experiments is not guaranteed.
Some facts and new logic discovered during this work are being used as foundations for other
solutions. This is also why the experiment remains worth recording even though it failed at
its original goal.

This public repository includes only reviewed release files:

```text
README.md
README.en.md
LICENSE
cbt/
scripts/
papers/
configs/
small audited data artifacts and hashes
```

The private working branch is **not** published as-is. The private history contains internal
planning machinery and is not exposed. This public repository includes only audited files.

---

## Scope and Limits

This repository does not claim:

- a finished verified Transformer;
- a general hallucination solution;
- that prompt-level contracts are architecturally novel;
- that HPIC-complex is useful as a router;
- that the possibility-score geometry is an essential signal;
- that the CE modules solve real open-domain task extraction.

It does claim:

- a documented falsification-first process;
- a bounded contract-conditioned abstain payoff on source-silent questions;
- a reproducible record of failed proxies (four "fancy geometry → simple rule" reductions);
- an honest closing verdict: the new-architecture thesis did not survive its own tests.

---

## Citation

> David Seo / GEM².AI (2026). *gem2-CBT: Human-led falsification toward
> Contract-Bounded Transformer systems.* CC-BY-4.0.

---

*Conclusions are bounded. The value of this repository is the audited path: what was proposed,
what was tested, what failed, what survived, and how the target object was corrected — closed
honestly rather than left open.*
