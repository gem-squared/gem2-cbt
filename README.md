<!-- 언어: 한국어 (default) | English: README.en.md -->

# gem2-CBT — Contract-Bounded Transformer 연구 기록

> **상태: 진행 중인 연구, Phase 1 기록.**
> 이 저장소는 완성된 모델이 아니며, 검증된 새로운 Transformer도 아닙니다.
> 인간이 주도하고 AI가 가속한 **Contract-Bounded Transformer (CBT)** 시스템
> 구축 시도의 공개 기록입니다. 잘못된 갈래, 실패한 대리 지표, 철회,
> 살아남은 메커니즘, 그리고 진짜 contract 추출을 중심에 둔 현재의
> 재구축 작업까지 모두 포함합니다.

**라이선스:** [CC-BY-4.0](LICENSE) | © 2026 David Seo / GEM².AI | English: [README.en.md](README.en.md)

---

## 왜 존재하는가

CBT는 간단한 명제에서 출발합니다:

```text
LLM이 답하기 전에, 그 답을 한정짓는 contract를 먼저 알아야 한다.
```

이 프로젝트에서 진짜 contract는 태그나 라벨이 아닙니다. contract는 다음에 가깝습니다:

```text
F: A -> B | P
```

여기서 `P`는 입력 `A`에서 출력 `B`로의 변환이 어떻게 일어날 수 있는지를
한정짓는 조건입니다.

장기적인 CBT 목표는 MoE 유사한 contract 아키텍처입니다:

```text
입력 코퍼스
  -> CER: Contract Extractor Router
  -> Task / Context / Concept CE 또는 ECE 모듈
  -> Contract Pack
  -> contract-conditioned 추론
  -> Verifier / abstain / repair
```

현재의 재구축은 첫 번째 게이트인 **Task-CER**에서 시작합니다. 시스템은
코퍼스에 task-가치 있는 구조가 있는지 판단하고, Task-Possibility Score (TPS)를
추정하고, 큰 코퍼스를 task-단위 청크로 분할하고, 다중-task 프롬프트를
재귀적으로 분해하며, 하나 또는 여러 contract extractor를 활성화해야 합니다.

이것이 지금 우리가 시험하려는 대상입니다. 초기 실험들이 항상 이 대상을
정확히 시험한 것은 아니었습니다.

---

## 인간-AI 협업 이야기

이 저장소는 협업 패턴 또한 기록합니다.

첫 평가 단계는 AI 협력자가 주도했습니다. 빠르고, 엄격하고, 유용했지만,
잘못된 대상을 평가하는 경우가 잦았습니다. 초기 몇몇 실험은 라벨, 사실,
WSD sense, 프롬프트 비계를 마치 CBT contract인 양 다루었습니다. 그 결과
실제 실험과 유용한 부정적 결과는 얻었지만, **진짜 contract 대상**이
시험되기도 전에 "CBT는 그냥 structured RAG일 뿐"과 같은 지나치게 강한
결론으로 프로젝트가 휩쓸렸습니다.

교훈은 "AI는 창의적 작업에 쓸모없다"가 아닙니다. 더 날카로운 교훈은:

```text
강한 비판 + 약한 문제 정의 = 잘못된 대상에 대한 자신 있는 기각
강한 비판 + 인간의 정의      = 생산적인 반증
```

David의 역할은 프로젝트를 첫 원칙으로 계속 되돌려놓는 것이었습니다:

- contract는 라벨이 아니다;
- Task / Context / Concept는 경계가 있는 contract pixel로 추출되어야 한다;
- CER이 첫 번째 아키텍처 게이트다;
- CE를 만들 수 없다면 CBT는 폐기되어야 한다;
- 어떤 시험이 진짜 대상이 아니라 대리 지표를 겨냥했다면, 결과는 그 범위
  안에서만 인정하거나 기각되어야 한다.

이것이 실패한 시도들을 보존하는 이유입니다. 마케팅이 아닙니다. 정밀한
실험이 얼마나 쉽게 잘못된 질문에 답할 수 있는지에 대한 증거입니다.

이것은 단일 사례 연구이며, AI 시스템이나 인간-AI 연구에 대한 보편적
주장이 아닙니다.

---

## 지금까지 살아남은 것

### 1. Contract 내용은 행동에 영향을 준다 (behaviorally active)

WP-6A는, **메모리 의존이 없는 반사실(counterfactual) 콘텐츠** 위에서,
지식만 담은 contract가 fair한 strong prompt 대비 경계 위반을 줄일 수 있음을
보였습니다:

```text
B_FAIR violation:  1.000
C_KNOW violation:  0.000
Delta:            -1.000
```

이 결과는 이전의 "abstain" 명령이 일으킨 효과가 아니었습니다. WP-6A는 그
tautology를 제거하고도 반사실 영역에서 payoff를 관찰했습니다.

범위가 한정된 해석:

```text
모델이 의도된 binding을 모르거나 거부할 때, in-context로 binding을 공급하면 작동한다.
```

이는 extractor stack을 만드는 근거가 됩니다. 새 Transformer를 **증명한 것은
아닙니다.**

### 2. 학습/프롬프트 기반 extractor가 payoff를 유지했다

WP-10은 손으로 쓴 oracle pack을, 동일한 단일-binding 반사실 범위에서,
**학습/프롬프트 기반 extractor**로 대체했습니다. 추출된 pack은 payoff를
유지했습니다:

```text
C_PACK_LEARNED violation:  0.000
C_KNOW_ORACLE  violation:  0.000
```

WP-11은 같은 결과를 두 번째 subject 모델에서 재현했습니다:

```text
deepseek-chat
qwen2.5-32b-instruct-q8_0
```

범위가 한정된 해석:

```text
단일-모델 caveat은 두 개의 시험된 모델 family에 걸쳐 실질적으로 다뤄졌으며,
보편적으로 해소된 것은 아니다.
```

### 3. 단순 범위에서는 plain facts가 천장을 쳤다 (saturated)

같은 WP-10/WP-11 시험에서:

```text
PLAINFACTS      violation:  0.000
C_PACK_LEARNED  violation:  0.000
```

이것은 **saturated tie** (천장 동률)이며, 구조에 대한 보편적 실패가
아닙니다.

범위가 한정된 해석:

```text
단순 단일-binding, 짧은 컨텍스트 범위에서는 plain in-context facts만으로도 과제가 풀린다.
구조화된 contract pack은 그 자리에서 추가 가치를 입증하지 못한다.
```

이는 contract 구조가 일반적으로 무용함을 증명하는 것이 **아닙니다.** 그
단순 프롬프트 범위에 헤드룸이 없었음을 증명할 뿐입니다.

### 4. Complex HPIC는 router로 기각

complex phasor 형식을 분류기/router/gate 장치로 시험했고, 일반 routing
메커니즘으로는 실패했습니다.

핵심 결과:

```text
Z = Sigma rho * exp(i theta)
```

는, 시험된 결정들에 대해 두 개의 실수 feature를 **invertible (가역적)**
재매개화한 것이었습니다. 실제 자연어 routing에서는, 압축된
`(signed_strength, evidence_spread)` 표현이 지배적인 신호를 버리고 raw
feature 위 softmax에 크게 졌습니다.

범위가 한정된 결정:

```text
CER 베이스라인에는 plain softmax / raw-feature routing을 사용한다.
HPIC-complex를 routing 이점으로 주장하지 않는다.
```

### 5. CBT-v1 boundary-gated attention은 여전히 GATED

초기 Transformer 변형은 ecological 게이트를 통과하지 못했습니다. 채택되지
않았습니다.

---

## 잘못되었거나 철회된 것

중요하기 때문에 보존합니다.

- 초기 controls가 라벨 셔플링을 진짜 음성 control과 혼동했다.
- WSD는 concept-disambiguation probe로 유용했지만, 완전한 Concept Contract는
  아니었다.
- 몇몇 실험들은 태그나 사실을 contract처럼 다루었다. 충분하지 않았다.
- 첫 oracle-payoff 시험은 gagged baseline, 메모리에 박힌 도메인, 행동을
  주입하는 contract 때문에 confounded 되었다.
- HPIC-complex는 형식적으로 매력적이었지만 실제 routing 가치를 보태지 않았다.
- "CBT는 structured RAG"라는 일반 진술은 너무 강했다. 시험된 단순 범위에서는
  plain facts가 천장을 쳤다는 것이지, 구조가 어디서도 가치가 없다는
  증명은 아니었다.

---

## 현재 방향

프로젝트는 다시 첫 번째 아키텍처 게이트로 돌아왔습니다:

```text
Task-CER
```

당장의 대상은 완전한 CBT가 아닙니다. 실행 가능하고 시험 가능한 Task-CER
비계입니다:

```text
입력 코퍼스
  -> 필요 시 청크 분할
  -> TCLLM이 청크/구간별 TPS 추정
  -> 낮은 TPS: 일반 TextLLM으로 routing
  -> 높은 TPS: Task = (Actor, Input, Operation, Output, Constraint) 추출
  -> TTCLLM이 sub-task 분해 시도
  -> 더 작은 task가 없으면: 현재 task를 leaf로 반환
  -> 반복되는 leaf는 병합/중복 제거
  -> 하나 또는 여러 Task-CE 모듈 활성화
```

다음 공개-수준 시험이 평가해야 하는 것:

- TPS 보정;
- task / no-task false activation;
- task 경계 검출;
- 재귀적 분해;
- trivial-task 정지;
- 다중-task 추출;
- 과도-분절(over-fragmentation) 및 병합 거동;
- 추출된 Task 프레임이 인간 의도와 일치하는지.

Task / Context / Concept CE를 작동시킬 수 없다면, CBT 아키텍처는 폐기되거나
훨씬 더 작은 방법으로 축소되어야 합니다.

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
| 11 | 로컬 Qwen 재현 | 두 시험된 subject에 걸쳐 payoff 재현됨; saturation 재현됨 |
| Next | Task-CER 재구축 | 진행 중: 라벨-as-contracts가 아닌 진짜 Task contract 추출 |

---

## 재현 가능성

일부 실험은 CPU만 사용하며 API 키가 필요 없습니다. LLM 의존 실험은
로컬에서 설정한 OpenAI-호환 백엔드를 사용합니다. 비밀은 반드시 git 바깥에
있어야 합니다.

공개 릴리스는 curated surface를 사용해야 합니다:

```text
README.md           (한국어, 기본)
README.en.md        (English)
LICENSE
cbt/
scripts/
papers/
configs/
작게 감사된 데이터 아티팩트 및 해시
```

비공개 작업 브랜치를 그대로 공개하지 **마십시오.** 비공개 history에는 내부
계획 machinery가 들어 있으며 노출되어서는 안 됩니다. 신선한 public/orphan
branch나, 감사된 파일만 들어 있는 새 공개 저장소를 사용하십시오.

---

## 범위와 한계

이 저장소는 다음을 **주장하지 않습니다:**

- 완성되고 검증된 Transformer;
- 일반적인 hallucination 해법;
- 프롬프트 수준 contract가 아키텍처적으로 새롭다는 것;
- HPIC-complex가 router로 유용하다는 것;
- 현재의 CE 모듈이 실제 open-domain task 추출을 해결한다는 것.

이 저장소는 다음을 **주장합니다:**

- 문서화된 falsification-first 프로세스;
- 시험된 반사실 설정 안에서의 범위 한정된 contract-content payoff;
- 실패한 대리 지표들의 재현 가능한 기록;
- 진짜 Task / Context / Concept contract 추출을 향한 현재의 재구축.

---

## 인용

> David Seo / GEM².AI (2026). *gem2-CBT: Human-led falsification toward
> Contract-Bounded Transformer systems.* CC-BY-4.0.

---

*결론은 잠정적입니다. 이 저장소의 가치는 감사된 경로입니다: 무엇이
제안되었고, 무엇이 시험되었고, 무엇이 실패했고, 무엇이 살아남았고, 대상이
어떻게 교정되었는지.*
