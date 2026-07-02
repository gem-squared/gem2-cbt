<!-- 언어: 한국어 (default) | English: README.en.md -->

# gem2-CBT — Contract-Bounded Transformer 연구 기록

> **상태: 종료된 연구 기록 (closed).**
> 이 저장소는 내가 주도한 **Contract-Bounded Transformer (CBT)** 시스템 구축 시도의
> **완결된 공개 기록**입니다.
> 잘못된 갈래, 실패한 대리 지표,
> 철회, 그리고 살아남은 메커니즘을 모두 포함합니다. 결론적으로 **CBT를 새로운
> Transformer로 보는 명제는 이 프로젝트의 증거로는 지지되지 않으며**, red-team을
> 통과해 살아남은 유일한 결과는 **범위가 한정된 contract-conditioned abstain
> 메커니즘**입니다. 이것은 완성된 모델도, 검증된 새 Transformer도 아닙니다.

**라이선스:** [CC-BY-4.0](LICENSE) | © 2026 David Seo / GEM².AI | English: [README.en.md](README.en.md)

---

## 왜 만들었는가

CBT는 간단한 명제에서 출발했습니다:

```text
LLM이 답하기 전에, 그 답을 한정짓는 contract를 먼저 알아야 한다.
```

이 프로젝트에서 진짜 contract는 태그나 라벨이 아닙니다. contract는 다음에 가깝습니다:

```text
F: A -> B | P
```

여기서 `P`는 입력 `A`에서 출력 `B`로의 변환이 어떻게 일어날 수 있는지를
한정짓는 조건입니다.

장기적인 CBT 목표는 MoE와 유사한 contract 아키텍처였습니다:

```text
입력 코퍼스
  -> CER: Contract Extractor Router
  -> Task / Context / Concept CE 또는 ECE 모듈
  -> Contract Pack
  -> contract-conditioned 추론
  -> Verifier / abstain / repair
```

재구축은 첫 번째 게이트인 **Task-CER**에서 시작했고, 이후 진짜 contract 대상을
직접 시험했습니다 (WP-13–18). 초기 실험들이 언제나 그 대상을 정확히 겨냥한 것은
아니었습니다. 이 기록은 그 실수와 교정을 함께 보존합니다.

---

## 인간-AI 협업 이야기

이 저장소는 실험 결과뿐 아니라, 인간과 AI가 어떻게 함께 사고하고 실패를
교정했는지도 기록합니다.

첫 평가 단계는 AI 협력자들(Claude Cowork, Codex, Claude Code)이 주도하도록
설계했습니다. 그들은 빠르고, 엄격하고, 유용했지만, 종종 잘못된 대상을
평가했습니다. 초기 몇몇 실험은 라벨, 사실, WSD sense, 프롬프트 비계를 CBT
contract처럼 다루었습니다. 덕분에 실제 실험과 유용한 부정적 결과는 얻었지만,
**진짜 contract 대상**이 시험되기도 전에 "CBT는 그냥 structured RAG일 뿐"과 같은
지나치게 강한 결론으로 흘러가기도 했습니다.

매번 드는 생각이지만, "AI는 창의적 작업에 매우 취약하다". 계획만 세우고
autonomous하게 진행하도록 하면, 명확하게 정의된 새로운 개념마저 무시하고 기존에
익숙한 방법으로 drifting되는 것을 반복적으로 관찰하게 되었습니다.

이 여정을 계속하는 동안 나의 주된 역할은 AI가 drifting되는지를 감시하고, 새롭게
수립된 원칙으로 AI 협력자들을 계속 되돌려놓는 것이었습니다:

- contract는 라벨이 아니다;
- Task / Context / Concept는 경계가 있는 contract pixel로 추출되어야 한다;
- CER이 첫 번째 아키텍처 게이트다;
- CE를 만들 수 없다면 CBT는 폐기되어야 한다;
- 어떤 시험이 진짜 대상이 아니라 대리 지표를 겨냥했다면, 결과는 그 범위
  안에서만 인정하거나 기각되어야 한다.

그래서 실패한 시도들을 지우지 않았습니다. 이것은 마케팅이 아닙니다. 정밀한
실험도 질문을 잘못 잡으면 얼마나 쉽게 틀린 대상을 반증할 수 있는지 보여주는
기록입니다. 그 잘못이 사람으로부터 왔던, AI로부터 왔던 결과는 동일합니다.

이것은 단일 사례 연구이며, AI 시스템이나 인간-AI 연구에 대한 보편적
주장이 아닙니다.

---

## 살아남은 것

### 1. Contract-conditioned abstain 메커니즘이 hallucination을 줄인다 (범위 한정)

가장 강하게 살아남은 결과입니다. 근거 문서가 답을 제공하지 않는 질문에서,
결정론적 **abstain** 경계를 담은 contract로 모델을 조건화하면, abstain을 허용하는
*fair* 프롬프트 대비 단정형 hallucination이 크게 줄었습니다. 이 효과는
vocabulary에 의존하지 않는 재검증도 통과했습니다:

```text
regime: 답이 없는 / 근거 문서가 침묵하는 질문
hallucination (fair prompt)      ~ 0.18
hallucination (contract-abstain) ~ 0.03
```

범위가 한정된 해석:

```text
결정론적 abstain 경계를 공급하면, 근거 문서가 침묵할 때 모델이 거부하도록 만든다 —
fair 프롬프트가 달성하는 것 이상으로. 단일 모델 family, 프롬프트 수준, 이 regime에 한정.
```

이것은 좁은 신뢰성 메커니즘입니다. 새 Transformer를 **증명한 것은 아닙니다.**

### 2. Contract 내용은 행동에 영향을 준다 (behaviorally active)

WP-6A는, **메모리 의존이 없는 반사실(counterfactual) 콘텐츠** 위에서,
지식만 담은 contract가 fair한 strong prompt 대비 경계 위반을 줄일 수 있음을
보였습니다:

```text
B_FAIR violation:  1.000
C_KNOW violation:  0.000
```

이 결과는 이전의 "abstain" 명령이 일으킨 효과가 아니었습니다. WP-6A는 그
tautology를 제거하고도 반사실 영역에서 payoff를 관찰했습니다.

### 3. 학습/프롬프트 기반 extractor가 payoff를 유지했다 (두 모델)

WP-10은 손으로 쓴 oracle pack을, 동일한 단일-binding 반사실 범위에서,
**학습/프롬프트 기반 extractor**로 대체했고 payoff는 유지됐습니다. WP-11은 두 번째
시험 대상 모델(`deepseek-chat`, `qwen2.5-32b-instruct-q8_0`)에서 재현했습니다. 단일-모델
caveat은 시험한 두 모델 family에 걸쳐 실질적으로 다뤄졌으나, 보편적으로 해소된
것은 아닙니다. (같은 시험에서 단순 범위의 plain facts는 천장을 쳤습니다 —
구조의 보편적 실패가 아니라, 그 단순 범위에 헤드룸이 없었다는 뜻.)

### 4. 결정론적 contract check는 위반이 토큰을 바꾸는 레벨에서만 작동한다

레벨별 데이터셋(WP-16)에서 나온 핵심 구조적 발견입니다. 결정론적 `¬B` check는
위반이 *표면 토큰을 바꾸는* 레벨에서만 진짜로 결정론적입니다:

```text
Task    (fabrication / abstain)   : 결정론적으로 검증 가능
Concept (잘못된 word-sense)        : 결정론적으로 검증 가능
Context (역할 뒤바뀜, 같은 토큰)    : 결정론적으로 검증 불가 (graded만 가능)
```

역할 뒤바뀜("사람이 우산을 들었다" vs "우산이 사람을 들었다")은 정확히 같은 토큰을
재사용하므로 token-grounding으로 잡을 수 없습니다. 결정론적 경계를 가진 contract
원시는 Task/Concept에는 강하고 Context에는 graded일 뿐입니다.

---

## 잘못되었거나 철회된 것

중요하기 때문에 보존합니다.

- 초기 control 설계가 라벨 셔플링을 진짜 negative control과 혼동했다.
- WSD는 concept-disambiguation probe로 유용했지만, 완전한 Concept Contract는
  아니었다.
- 몇몇 실험들은 태그나 사실을 contract처럼 다루었다. 충분하지 않았다.
- 첫 oracle-payoff 시험은 gagged baseline, 메모리에 박힌 도메인, 행동을
  주입하는 contract 때문에 confounded 되었다.
- **Complex HPIC**(*Hierarchical Phase-Interval Classifier*, 위계적 위상-구간 분류기)는
  분류를 복소수 평면 위의 방향으로 해석하려던 장치였다. 각 evidence를 각도
  `θ=arccos(2p−1)`로 놓고, 작은 relation-region(`Sm`)들이 만든 phasor를
  `Z = Σ ρ·exp(iθ)`로 합산해 `Re(Z)`의 부호로 판정하고 90도 축 근처에서는
  abstain/Unknown을 내도록 설계했다. 처음에는 Spaceship Titanic 분류기로 falsified
  되었고, CBT 안에서는 router / `⊥`-gate로 다시 시험되었다. 그 결과 HPIC는
  `Re(Z)=signed_strength`, `Im(Z)=evidence_spread`라는 두 실수 feature의 가역적
  재매개화로 축소되었고, raw feature 위 softmax에 졌다 (세 개의 독립 전선에서).
- **Possibility-score의 density+distance 기하는 cosmetic이었다** (WP-18): density와
  distance가 강하게 collinear(corr ≈ −0.92)였고, 둘을 합쳐도 단일 feature 대비
  마진이 없었다. 복합 점수가 아니라 가장 단순한 scorer를 채택하라. 이것은 "화려한
  기하 → 단순 규칙" 축소가 일어난 **네 번째** 사례다.
- 합성 레벨-라벨 substrate는 레벨 검출에 대해 **구조적으로 saturated**였다
  (WP-17): 세 레벨이 세 개의 서로 다른 topic이어서, 어떤 scorer든 레벨이 아니라
  topic을 검출해 "이겼다" — 의도한 시험에는 무효이며 폐기됨.
- "CBT는 structured RAG"라는 일반 진술은 너무 강했다. 시험된 단순 범위에서
  plain facts가 천장을 쳤다는 것이지, 구조가 어디서도 가치가 없다는 증명은
  아니었다.

---

## 결론

CBT 여정은 기록으로서 종료됩니다. 이 프로젝트의 실험은 핵심 질문을 다음처럼
정리했습니다:

- **CBT를 새 Transformer / 새 신경망 아키텍처로 보는 명제: 이 프로젝트의 증거로는
  지지되지 않음.** 모든 새로운 미분가능/기하 메커니즘이 표준이나 graded로
  축소되었다 — complex HPIC router(cosmetic, 세 전선), possibility-score density+distance
  기하(cosmetic, 단일 feature로 축소), CBT-v1 boundary-gated attention(gated). 이는
  범위-한정 결과다: *우리가 시험한 메커니즘들*이 실패한 것이지, 그런 아키텍처가
  존재할 수 없음을 증명한 것은 아니다.
- **살아남은 것은 아키텍처가 아니라 범위-한정된 메커니즘이다:** contract를
  in-context로 공급하고, 그 안에 결정론적 *abstain* 경계를 넣으면 근거 문서가
  침묵할 때 hallucination이 줄어든다(단일 모델, 프롬프트 수준에 한정). 또한
  결정론적 contract check는 일부 레벨(Task, Concept)에는 진짜 결정론적이고 다른
  레벨(Context)에는 graded일 뿐이다.

```text
새 아키텍처 명제는 살아남지 못했다. 좁은 결정론적 경계 기반 신뢰성 메커니즘은
살아남았다. 남은 자산은 감사된 반증 경로 그 자체다.
```

---

## 작업 패키지 원장 (Work Package Ledger)

| WP | 질문 | 현재 해석 |
|----|------|-----------|
| 1 | 첫 CBT 신호 hardening | Confounded; control 문제 발견 |
| 2 | Contract 내용이 행동에 영향을 주는가? | Yes, 범위 한정 |
| 3 | 실제 언어 WSD/ecological probe | 유용한 probe; 완전한 Concept Contract는 아님 |
| 4 | Complex boundary gate | plain 2-feature 규칙과 동률 |
| 5 | Complex CER router | softmax에 실패 |
| 6 | 첫 oracle payoff | Confounded; footprint로 보존 |
| 6A | Fair oracle payoff | 메모리-독립 반사실 콘텐츠에서 PASS |
| 7 | Concept 추출 | LLM/프롬프트 extractor가 범위 한정 합성 probe 통과; 비-LLM extractor 실패 |
| 8 | 실제 NL HPIC router falsifier | HPIC-complex가 routing에서 기각됨 |
| 9 | 공개 릴리스 준비 | raw main이 아니라 curated release surface |
| 10 | Learned extractor microcell | payoff 유지됨; structure vs facts saturated |
| 11 | 로컬 Qwen 재현 | 두 시험 대상 모델에 걸쳐 payoff 재현됨; saturation 재현됨 |
| 13 | C/C/T 분할 축 | 합성 데이터에서 coherent하나 by-construction; 결론 불가 |
| 14 | QA 데이터에서 진짜 contract seed | token-grounding 반-조작 추출 seed 구축 |
| 15 | Contract-conditioned abstain payoff | **진짜, 범위 한정** — 근거 문서가 침묵하는 질문에서 hallucination 감소 (단일 모델, 프롬프트 수준) |
| 16 | 레벨별 contract 데이터셋 | Task / Concept 결정론적 검증 가능; **Context는 graded만** |
| 17 | Possibility-score bake-off (합성) | substrate가 구조적으로 saturated; 무효 — 폐기 |
| 18 | Possibility-score 기하 | density+distance **cosmetic** (= 단일 feature); 가장 단순한 scorer 사용 |
| — | CBT 명제 | **종료(CLOSED)**: 새-Transformer 지지 안 됨; 살아남은 결과 = 범위-한정 contract-abstain |

---

## 재현 가능성

본 실험에 대한 재현 가능성은 보장하지 않습니다. 
본 실험에서 얻은 일부 사실과 새로운 로직들은 다른 솔루션의 기반으로 사용되기 때문입니다. 
이 실험이 원래의 목적에는 실패했지만, 기록할 만한 가치가 있는 이유가 여기에 있습니다. 
이 공개 저장소에는 검토를 마친 파일만 포함했습니다:

```text
README.md           (한국어, 기본)
README.en.md        (English)
LICENSE
cbt/
scripts/
papers/
configs/
small audited data artifacts and hashes
```

비공개 작업 브랜치는 그대로 공개하지 **않습니다.** 비공개 history에는 내부
계획 machinery가 들어 있어 노출되지 않으며, 이 공개 저장소는 감사된 파일만
담긴 surface입니다.

---

## 범위와 한계

이 저장소는 다음을 **주장하지 않습니다:**

- 완성되고 검증된 Transformer;
- 일반적인 hallucination 해법;
- 프롬프트 수준 contract가 아키텍처적으로 새롭다는 것;
- HPIC-complex가 router로 유용하다는 것;
- possibility-score 기하가 핵심 신호라는 것;
- CE 모듈이 실제 open-domain task 추출을 해결한다는 것.

이 저장소는 다음을 **주장합니다:**

- 문서화된 falsification-first 프로세스;
- 근거 문서가 침묵하는 질문에서 범위 한정된 contract-conditioned abstain payoff;
- 실패한 대리 지표들의 재현 가능한 기록 ("화려한 기하 → 단순 규칙" 축소 4회);
- 정직한 종결 판정: 새-아키텍처 명제는 자신의 시험을 통과하지 못했다.

---

## 인용

> David Seo / GEM².AI (2026). *gem2-CBT: Human-led falsification toward
> Contract-Bounded Transformer systems.* CC-BY-4.0.

---

*결론은 범위가 한정되어 있습니다. 이 저장소의 가치는 감사된 경로입니다: 무엇이
제안되었고, 무엇이 시험되었고, 무엇이 실패했고, 무엇이 살아남았고, 대상이
어떻게 교정되었는지 — 열린 채로 두지 않고 정직하게 종결했습니다.*
