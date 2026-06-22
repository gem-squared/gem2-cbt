<!-- Language: English | Korean: README.ko.md -->

# gem2-CBT — Contract-Bounded Transformer Research Log

> **Status: active research, Phase 1 record.**
> This repository is not a finished model and not a proven new Transformer. It is a public
> record of a human-led, AI-accelerated attempt to build toward **Contract-Bounded
> Transformer (CBT)** systems, including the wrong turns, failed proxies, retractions,
> surviving mechanisms, and the current rebuild around real contract extraction.

**License:** [CC-BY-4.0](LICENSE) | (c) 2026 David Seo / GEM².AI | Korean: [README.ko.md](README.ko.md)

---

## Why This Exists

CBT starts from a simple thesis:

```text
Before an LLM answers, it should know the contract that bounds the answer.
```

For this project, a real contract is not a tag or label. A contract is closer to:

```text
F: A -> B | P
```

where `P` is the condition that bounds how the transformation from input `A` to output `B`
is allowed to happen.

The long-term CBT target is a MoE-like contract architecture:

```text
Input corpus
  -> CER: Contract Extractor Router
  -> Task / Context / Concept CE or ECE modules
  -> Contract Pack
  -> Contract-conditioned inference
  -> Verifier / abstain / repair
```

The current rebuild begins at the first gate: **Task-CER**. The system must decide whether
a corpus contains task-worthy structure, estimate a Task-Possibility Score (TPS), split
large corpora into task-bearing chunks, recursively decompose multi-task prompts, and
activate one or more contract extractors.

That is the object we are now trying to test. Earlier experiments did not always test this
object correctly.

---

## The Human-AI Story

This repository also records a collaboration pattern.

The first evaluation pass was led heavily by AI collaborators. They were fast, rigorous,
and useful, but they often evaluated the wrong object. Several early tests treated labels,
facts, WSD senses, or prompt scaffolds as if they were CBT contracts. That produced real
experiments and useful negative results, but it also pushed the project toward overly
strong conclusions such as "CBT is just structured RAG" before the real contract object
had been tested.

The lesson is not "AI is useless at creative work." The sharper lesson is:

```text
Strong critique + weak problem framing = confident rejection of the wrong object.
Strong critique + human framing = productive falsification.
```

David's role has been to keep forcing the project back to first principles:

- a contract is not a label;
- Task / Context / Concept must be extracted as bounded contract pixels;
- CER is the first architectural gate;
- if CE cannot be built, CBT should be discarded;
- if a test targets a proxy instead of the real object, the result must be bounded or rejected.

This is why the failed attempts are kept. They are not marketing. They are evidence of how
easy it is to make a precise experiment answer the wrong question.

This is a single case study, not a universal claim about AI systems or human-AI research.

---

## What Survived So Far

### 1. Contract content is behaviorally active

WP-6A showed that, on memory-independent counterfactual content, a knowledge-only contract
can reduce boundary violations versus a fair strong prompt:

```text
B_FAIR violation: 1.000
C_KNOW violation: 0.000
Delta: -1.000
```

The win was not the earlier "abstain" imperative. WP-6A removed that tautology and still
observed the payoff in the counterfactual regime.

Bounded reading:

```text
When the model lacks or resists the intended binding, supplying the binding in-context works.
```

This supports building an extractor stack. It does **not** prove a new Transformer.

### 2. A learned/prompted extractor retained the payoff

WP-10 replaced the hand-written oracle pack with a learned/prompted extractor in the same
single-binding counterfactual scope. The extracted pack retained the payoff:

```text
C_PACK_LEARNED violation: 0.000
C_KNOW_ORACLE violation: 0.000
```

WP-11 reproduced the same finding on a second subject model:

```text
deepseek-chat
qwen2.5-32b-instruct-q8_0
```

Bounded reading:

```text
The single-model caveat is substantially addressed across two tested model families,
not resolved universally.
```

### 3. Plain facts saturated the simple scope

The same WP-10/WP-11 tests showed:

```text
PLAINFACTS violation: 0.000
C_PACK_LEARNED violation: 0.000
```

This is a **saturated tie**, not a universal failure of structure.

Bounded reading:

```text
In a simple single-binding, short-context scope, plain in-context facts already solve the task.
The structured contract pack has no demonstrated marginal value there.
```

This does not prove contract structure is cosmetic in general. It proves the simple prompt
scope had no headroom.

### 4. Complex HPIC as router is rejected

The complex phasor form was tested as classifier/router/gate machinery. It failed as a
general routing mechanism.

The key result:

```text
Z = Sigma rho * exp(i theta)
```

is an invertible reparameterization of two real features for the decisions tested. On real
NL routing, the compressed `(signed_strength, evidence_spread)` representation discarded
the dominant signal and lost badly to softmax over raw features.

Bounded decision:

```text
Use plain softmax/raw-feature routing for CER baselines.
Do not claim HPIC-complex as a routing advantage.
```

### 5. CBT-v1 boundary-gated attention remains gated

The early Transformer variant did not pass its ecological gate. It is not adopted.

---

## What Was Wrong or Retracted

These are kept because they matter.

- Early controls confused label shuffling with real negative controls.
- WSD was useful as a concept-disambiguation probe, but it was not a full Concept Contract.
- Several experiments treated tags or facts as contracts. That was not sufficient.
- A first oracle-payoff test was confounded by a gagged baseline, memorized domain, and
  behavior-injecting contract.
- HPIC-complex looked attractive formally, but did not add routing value under test.
- "CBT is structured RAG" was too strong as a general statement. In the tested simple
  scope, plain facts saturated. That is not the same as proving structure has no value
  everywhere.

---

## Current Direction

The project is now back at the first architectural gate:

```text
Task-CER
```

The immediate object is not full CBT. It is a runnable, testable Task-CER scaffold:

```text
Input corpus
  -> chunk if needed
  -> TCLLM estimates TPS per chunk/span
  -> low TPS: route to ordinary TextLLM
  -> high TPS: extract Task = (Actor, Input, Operation, Output, Constraint)
  -> TTCLLM attempts sub-task decomposition
  -> if no smaller task is found: return current task as a leaf
  -> merge/deduplicate repeated leaves
  -> activate one or more Task-CE modules
```

The next public-grade test must evaluate:

- TPS calibration;
- task/no-task false activation;
- task boundary detection;
- recursive decomposition;
- trivial-task stopping;
- multi-task extraction;
- over-fragmentation and merge behavior;
- whether extracted Task frames match human intent.

If Task / Context / Concept CE cannot be made to work, the CBT architecture should be
discarded or narrowed to a much smaller method.

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
| Next | Task-CER rebuild | In progress: real Task contract extraction, not labels-as-contracts |

---

## Reproducibility

Some experiments are CPU-only and require no API key. LLM-dependent experiments use an
OpenAI-compatible backend configured locally. Secrets must remain outside git.

Public release should use a curated surface:

```text
README.md
README.ko.md
LICENSE
cbt/
scripts/
papers/
configs/
small audited data artifacts and hashes
```

Do **not** publish the private working branch as-is. The private history contains internal
planning machinery and should not be exposed. Use a fresh public/orphan branch or a new
public repository with only audited files.

---

## Scope and Limits

This repository does not claim:

- a finished verified Transformer;
- a general hallucination solution;
- that prompt-level contracts are architecturally novel;
- that HPIC-complex is useful as a router;
- that current CE modules solve real open-domain task extraction.

It does claim:

- a documented falsification-first process;
- a bounded contract-content payoff in tested counterfactual settings;
- a reproducible record of failed proxies;
- a current rebuild toward real Task / Context / Concept contract extraction.

---

## Citation

> David Seo / GEM².AI (2026). *gem2-CBT: Human-led falsification toward
> Contract-Bounded Transformer systems.* CC-BY-4.0.

---

*Conclusions are provisional. The value of this repository is the audited path: what was proposed,
what was tested, what failed, what survived, and how the target object was corrected.*
