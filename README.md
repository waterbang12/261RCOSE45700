# 무인 매장 AI 관리 시스템

LangGraph 기반 멀티에이전트 아키텍처로 구현된 무인 매장(무인 카페·편의점 등) 통합 관리 시스템.
영상 안전·위험행동 감지, 고객 응대, 일일 보고서를 역할별 전문 에이전트가 처리하며, 오케스트레이터가 에이전트 간 협력을 조율한다.
카메라단(MediaPipe + OpenCV)은 땀 닦기·몸 흔들림 등 이상 신호를 감지하고, 외부 분석에 보내기 전 **얼굴을 모자이크 처리**해 개인정보를 보호한다.
모든 에이전트는 **부수효과 없는 "결정 노드"**이며, 실제 실행(알림·환경 제어·TTS)은 그래프 말단의 `actuator` 노드가 전담한다.

> **방향 전환 (`docs/team_pivot.md`):** 같은 LangGraph 구조를 재사용해 **운영자용 데이터 인텔리전스 플랫폼**으로 확장 중이다. 새로 추가된 **상권 인사이트 엔진**(`trigger_type="insight"`)은 상권 인구통계 → 자연어 인사이트 + 상품 구성 추천을 제공한다. 자세한 내용은 [상권 인사이트 엔진](#상권-인사이트-엔진-trigger_typeinsight) 절 참고.

---

## 아키텍처

```
                    [관리자 (Human)]
                          ↑ SMS / 푸시 알림
                          │
              [오케스트레이터 — dispatch] ← 모든 진입은 여기서 분배
                          │
              ┌───────────┼───────────┐
    [안전 에이전트]  [고객봇 에이전트]  [보고서 에이전트]
    gpt-5-mini       gpt-5-mini       gpt-5-mini
                     (구역별 N개)
              └───────────┼───────────┘
                          │
             [오케스트레이터 — reconcile] ── gpt-5-mini
              (크로스존 LLM 판단 / 충돌 알림 / 이상 푸시)
                          │
               [실행(actuator) 노드]   ← 모든 부수효과는 여기서만 실행
                          ↓
              알림 발송 │ 환경 제어 │ TTS 음성 │ 관리자 푸시

설계 원칙: 결정(에이전트) ↔ 실행(actuator) 분리,
           진입(dispatch) ↔ 정책 조정(reconcile) 분리
```

### 에이전트별 역할

| 에이전트 | 파일 | 모델 | 역할 |
|---------|------|------|------|
| 오케스트레이터 — dispatch | `agents/orchestrator.py` (`orchestrator_dispatch_node`) | — (LLM 미사용) | 진입점. `trigger_type` 검증 후 전문 에이전트로 분배 |
| 오케스트레이터 — reconcile | `agents/orchestrator.py` (`orchestrator_reconcile_node`) | gpt-5-mini | 크로스존 요청 판단, 충돌 중재, 관리자 에스컬레이션 결정 |
| 안전 감지 | `agents/safety_agent.py` | gpt-5-mini → gpt-5 | 영상 분석, confidence 기반 모델 자율 에스컬레이션 |
| 고객봇 | `agents/customer_agent.py` | gpt-5-mini | 자연어 요청 처리, 환경 제어 판단, 크로스존 요청 감지 |
| 보고서 | `agents/report_agent.py` | gpt-5-mini | 일일 운영 데이터 요약, 이상 패턴 감지 |
| 인사이트 | `agents/insight_agent.py` | gpt-5-mini | 상권 인구통계 해석 → 피크 시간대·수요·적합 업종 인사이트 |
| 추천 | `agents/recommendation_agent.py` | gpt-5-mini | 프로필 + 인사이트 → 상품 구성/재고 조정 추천 |
| 실행(actuator) | `agents/actuator.py` | — (LLM 미사용) | 결정 노드가 발행한 `pending_actions` 일괄 실행 |

### LLM API 호출 전략

모든 LLM 호출이 OpenAI **Responses API** (`client.responses.create`)로 통일되어 있다. 응답은 모두 `raw.output_text`로 파싱하며, `service_tier`로 호출별 처리 우선순위를 지정한다.

| 호출 대상 | `service_tier` | 이유 |
|-----------|---------------|------|
| 절도 / 파손 / 낙상 감지 (VLM) | `"priority"` | 즉각 대응 필요한 안전 위협 |
| 고객 응대 | `"priority"` | 실시간 UX, 응답 지연 불가 |
| 땀 감지 (VLM) | `"default"` | 긴박하지 않은 환경 이벤트 |
| 크로스존 판단 (오케스트레이터) | `"default"` | 준실시간, 다른 요청보다 덜 긴박 |
| 일일 보고서 | `"flex"` | 배치성, 지연 허용 / 저비용 |

> `service_tier="priority"` 는 [OpenAI Priority Processing](https://openai.com/api-priority-processing/) 구독이 필요하다. 미구독 시 `"default"`로 변경해도 기능상 동일하게 동작한다.

> 결정 노드(안전/고객/보고서/오케스트레이터)는 알림·온도 제어·TTS 같은 부수효과를 직접 호출하지 않는다.
> 대신 실행 의도를 `state["pending_actions"]`에 적고, `actuator` 노드가 그래프 말단에서 이를 일괄 실행한다.

### 라우팅 규칙

```
START                  → [orchestrator_dispatch] : 항상
orchestrator_dispatch  → [safety|customer|report|insight] : state["trigger_type"]에 따라 분배
safety / customer / report → [orchestrator_reconcile] : 항상
insight                → [recommendation] → [orchestrator_reconcile] : 인사이트 경로 (부수효과 없이 통과)
orchestrator_reconcile → [actuator] : 항상
                                       ├─ cross_zone_request 있음   : LLM으로 크로스존 판단 후 의도 발행
                                       ├─ conflict_detected=True    : 관리자 알림 의도 발행
                                       ├─ anomaly_detected=True     : 관리자 푸시 의도 발행
                                       └─ 그 외                     : no-op (state 그대로 통과)
actuator               → [END] : 부수효과 실행 후 종료
```

---

## 프로젝트 구조

```
실전SW/
├── main.py                        # 진입점 — 구역 카메라 브리지 + 상태머신 루프 + 보고서 루프
├── test_modules.py                # 모듈별 수동 테스트 (메뉴 [1]~[6])
├── test_agent.py                  # 랜덤 시나리오 자동 검증 에이전트
├── test_insight.py                # 상권 인사이트 엔진 end-to-end 데모/검증
├── requirements.txt
├── .env.example                   # 환경변수 템플릿
│
├── llm_module/                    # LangGraph 멀티에이전트 코어
│   ├── graph.py                   # LangGraph StateGraph 정의 (싱글톤 facility_graph)
│   ├── state.py                   # FacilityState 스키마 + 팩토리 함수
│   ├── state_machine.py           # VLM 빈도 제어 / cooldown / TriggerSignals (구역별)
│   ├── insight_schema.py          # DistrictProfile 입력 계약 (상권 인구통계 스키마)
│   ├── customer_bot.py            # 고객봇 public API (graph 경유)
│   ├── vlm_analyzer.py            # OpenAI Vision API 래퍼 (DetectionType 4종)
│   ├── report_generator.py        # 일일 리포트 생성
│   ├── tts.py                     # 공용 TTS 유틸리티 (출력)
│   ├── stt.py                     # 마이크 녹음 + Whisper 전사 (입력)
│   ├── alert_manager.py           # SMS / 푸시 알림 Tool
│   ├── temperature_controller.py  # 환경 제어 API Tool
│   └── agents/
│       ├── orchestrator.py        # 오케스트레이터 (dispatch + reconcile 두 노드)
│       ├── safety_agent.py        # 안전 감지 에이전트 (결정)
│       ├── customer_agent.py      # 고객봇 에이전트 (결정)
│       ├── report_agent.py        # 보고서 에이전트 (결정)
│       ├── insight_agent.py       # 인사이트 에이전트 (결정) — 상권 분석
│       ├── recommendation_agent.py # 추천 에이전트 (결정) — 상품 구성
│       └── actuator.py            # 실행 노드 (부수효과 전담)
│
├── data_simulation/               # 인사이트 엔진 입력 — 상권 인구통계 프로필 생성
│   └── personas.py                # aggregate_personas / SAMPLE_PROFILES / load_nemotron_profiles
│
├── opencv/                        # 영상 감지 파이프라인 (MediaPipe + OpenCV)
│   ├── bridge.py                  # 카메라 → 위험감지 → TriggerSignals/프레임 큐 (asyncio 어댑터)
│   ├── face_detection.py          # 독립 실행형 얼굴/손/자세 감지 데모
│   ├── hand_landmarker.task       # MediaPipe 손 랜드마크 모델
│   ├── blaze_face_short_range.tflite  # MediaPipe 얼굴 감지 모델 (모자이크용)
│   └── pose_landmarker_lite.task  # MediaPipe 자세 모델 (몸 흔들림 감지, 없으면 비활성)
│
├── web/                           # 실시간 흐름 시각화 (FastAPI + SSE)
│   ├── app.py                     # broadcast 채널 / /trigger / /events / /stream
│   └── static/index.html          # 단일 페이지 (입력/그래프/응답 실시간 표시)
│
├── db/
│   └── models.py                  # SQLite 스키마 + 쿼리 (customers / visits / events / env_logs)
│
└── (Arduino 액추에이터 & 레이턴시 데모 — 본체와 독립)
    ├── arduino_actuator.py        # USB 시리얼 래퍼 (ALERT/NORMAL, 보드 없으면 mock)
    ├── camera_bridge_demo.py      # Tapo RTSP → OpenCV → mock 이벤트 → 상태머신
    ├── camera_arduino_demo.py     # 위 데모 + Arduino 액추에이터 연동
    ├── mock_latency_demo.py       # mock 감지 → OpenAI 판단 → Arduino 엔드투엔드 레이턴시 측정
    ├── latency_profiler.py        # 파이프라인 구간별 타이밍 수집 (CSV/JSON)
    └── test_arduino.py            # 최소 시리얼 연결 테스트
```

> `opencv/bridge.py`가 `main.py`에 연결되는 운영 경로이고, 루트의 `camera_*_demo.py` / `*_latency_*.py`는 카메라·Arduino·레이턴시를 독립적으로 검증하는 데모 스크립트다.

---

## 상권 인사이트 엔진 (`trigger_type="insight"`)

`docs/team_pivot.md`의 방향 전환에 따라, 기존 LangGraph 구조를 **운영자용 데이터 인텔리전스**로 확장한 경로다. 상권 인구통계 프로필을 입력받아 LLM이 무인 매장 입지·운영 인사이트와 상품 구성 추천을 생성한다. 기존 "결정 노드 + actuator" 원칙을 그대로 따르되, 인사이트 경로는 부수효과가 없어 `reconcile`/`actuator`를 통과만 한다.

```
make_insight_state(district_profile)
   → insight_node          : 인구통계 해석 → 요약·인사이트·피크시간·적합업종  (state["insight_result"])
   → recommendation_node    : 프로필 + 인사이트 → 상품 구성/재고 조정안          (state["recommendations"])
   → reconcile → actuator   : 발행 액션 0개, no-op 통과 → END
```

### 입력 계약 — `DistrictProfile`

`llm_module/insight_schema.py`에 정의. 개인 단위 페르소나를 `district`(예: `"서울-강남구"`) 기준으로 집계한 상권 프로필이다.

| 필드 | 내용 |
|------|------|
| `district` / `province` / `sample_size` | 식별 + 집계 표본 수 |
| `age` | `{mean, cohorts}` 연령 분포 |
| `sex_ratio` / `family_types` / `housing_types` / `education_levels` | 인구통계 분포 |
| `top_occupations` / `top_hobbies` | 직업군·취미 상위 항목 |
| `hourly_footfall` *(선택)* | 시간대별 유동인구. 없으면 LLM이 인구통계로 추론 |

### 데이터 공급 (`data_simulation/personas.py`)

| 함수/상수 | 역할 |
|----------|------|
| `aggregate_personas(rows)` | 개인 페르소나 리스트 → `DistrictProfile` 순수 집계 함수 |
| `SAMPLE_PROFILES` | 오프라인 데모/테스트용 사전 집계 프로필 (다운로드 불필요) |
| `load_nemotron_profiles()` | [`nvidia/Nemotron-Personas-Korea`](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) (개인 1M건) 스트리밍 집계 — `datasets` 패키지 필요 |

> **스키마 격리 설계:** 두 에이전트는 `district_profile` dict **전체를 JSON으로 LLM에 넘긴다**(코드가 직접 읽는 필드는 로깅용 `district`·`sample_size`뿐). 따라서 데이터 스키마가 확정되면 `aggregate_personas()`의 매핑(`_NEMOTRON_FIELDS`)만 수정하면 되고, 그래프·노드·API 코드는 불변이다. 새 신호(소비·이동 패턴)는 그대로 통과되며, 프롬프트에 필드 설명을 보강하면 추론 품질이 향상된다.

### 실행

```bash
# 오프라인 (내장 샘플 상권)
python test_insight.py                       # SAMPLE_PROFILES 전체
python test_insight.py 서울-강남구 --json       # 특정 상권, JSON 출력

# 실제 Nemotron 데이터셋 (pip install datasets 필요)
python test_insight.py --nemotron 서울-강남구 --max-rows 50000

# API (web/app.py)
#   POST /insight  body: {"district": "서울-강남구"}  또는  {"profile": {...DistrictProfile...}}
```

---

## 영상 위험 감지 파이프라인 (`opencv/bridge.py`)

카메라 한 대당 별도 스레드에서 동기 OpenCV 루프가 돌며, MediaPipe로 손/얼굴/자세를 추적해 두 종류의 신호를 만든다. 결과는 `TriggerSignals`로 `ZoneStateMachine`에 전달되고, 이상 시 얼굴을 모자이크한 4프레임 배치가 VLM 파이프라인으로 넘어간다.

| 감지 | 방식 | 출력 신호 |
|------|------|----------|
| 땀 닦기 / 가림 (occlusion) | 얼굴이 감지된 상태에서 손 랜드마크가 얼굴 근처에 `OCCLUSION_TRIGGER_FRAMES`회 이상 머무름 | `sweat_wiping=True` |
| 몸 흔들림 (위험 행동) | `BodySwayDetector`가 어깨 중심점 이동 범위를 추적, 임계 초과가 연속되면 트리거 | `body_sway=True` → 상태머신에서 `FALL_EMERGENCY`로 매핑 |

**몸 흔들림 → 위험 분석 경로 (개인정보 보호 내장):**

```
어깨 흔들림 연속 감지 → 10프레임 캡처 → 얼굴 모자이크 처리 → JPEG 임시 저장
   → 백그라운드 스레드에서 GPT-4o(Vision)로 "위험 상황 여부" 분석 → 콘솔 출력
   → 분석에 사용한 임시 파일은 호출 직후 즉시 삭제
```

- 캡처/저장 직전 `apply_face_mosaic()`로 모든 얼굴을 픽셀화 → 외부 API에는 얼굴 식별 불가 이미지만 전송
- `OPENAI_API_COOLDOWN_SEC`(기본 30초) cooldown으로 중복 분석 방지
- `pose_landmarker_lite.task`가 없으면 몸 흔들림 감지만 자동 비활성, 나머지 파이프라인은 그대로 동작
- `DEBUG_CAMERA=1`이면 감지 오버레이 창 표시, `TRACE_CAMERA_DATA=1`이면 `[DATA][BRIDGE->STATE]` 트레이스 로그 출력

---

## Arduino 액추에이터 & 레이턴시 측정 (데모)

물리 액추에이터(경광등/팬 등)를 Arduino USB 시리얼로 제어하고, "감지 → 판단 → 작동"까지의 엔드투엔드 지연을 측정하는 독립 데모 묶음. **본체(`main.py`)와 분리되어 있고, 보드가 없으면 자동으로 mock 모드로 동작**한다.

```bash
# Tapo RTSP 카메라만 띄우기 (e=mock 이벤트, s=스크린샷, q=종료)
python camera_bridge_demo.py

# 카메라 + Arduino 연동 (e=ALERT 트리거, r=NORMAL 복귀)
python camera_arduino_demo.py

# 감지→판단→액추에이터 레이턴시 벤치마크 → latency_results.csv / .json 저장
python mock_latency_demo.py --events 5            # OpenAI 판단 포함
python mock_latency_demo.py --events 5 --agent-mock   # 로컬 규칙 판단(오프라인)
python mock_latency_demo.py --arduino-mock        # 보드 없이 mock 액추에이터

# 보드 직결 최소 확인
python test_arduino.py
```

- `arduino_actuator.py` — `ArduinoActuator`는 `pyserial` 미설치/연결 실패 시 자동 mock 전환. `ALERT`/`NORMAL` 명령에 대한 왕복 시간(`round_trip_ms`)을 함께 반환
- `latency_profiler.py` — `detection_created → ... → actuator_response_received` 7개 체크포인트를 수집해 구간별 ms와 합계를 CSV/JSON으로 저장하고 요약 표를 출력
- Arduino는 `.env`의 `ARDUINO_ENABLED=1`, `ARDUINO_PORT`, `ARDUINO_BAUD`로 제어

> 데모의 OpenAI 판단(`mock_latency_demo.py`)과 위험 분석(`opencv/bridge.py`)은 멀티에이전트 본체와 달리 `chat.completions` + `gpt-4o`를 직접 사용한다. Responses API 통일 규칙은 `llm_module`의 그래프/에이전트에만 적용된다.

---

## 설치 및 실행

### 요구사항

- Python 3.11+
- OpenAI API 키
- (선택) USB 웹캠 또는 Tapo TC70 등 RTSP IP 카메라 + MediaPipe 모델 파일(`opencv/`)
- (선택) Arduino 보드 + `pyserial` (액추에이터 데모)
- (선택) `datasets` 패키지 (인사이트 엔진의 `--nemotron` 실데이터 집계 모드)
- (선택) Solapi SMS 키, 환경 제어 API

### 설치

```bash
pip install -r requirements.txt
```

### 환경변수 설정

프로젝트 루트에 `.env` 파일을 만들고 아래 키를 입력한다:

```bash
OPENAI_API_KEY=sk-...
```

주요 환경변수 (전체 목록과 설명은 [.env.example](.env.example) 참고):

| 변수 | 설명 | 기본값 |
|-----|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 | 필수 |
| `VISION_MODEL` | 안전 감지 1차 모델 | `gpt-5-mini` |
| `CUSTOMER_MODEL` | 고객봇 모델 | `gpt-5-mini` |
| `REPORT_MODEL` | 일일 보고서 모델 | `gpt-5-mini` |
| `ESCALATION_MODEL` | 안전 감지 에스컬레이션 모델 | `gpt-5` |
| `ORCHESTRATOR_MODEL` | 오케스트레이터 reconcile 모델 | `gpt-5-mini` |
| `INSIGHT_MODEL` | 상권 인사이트 모델 | `gpt-5-mini` |
| `RECOMMENDATION_MODEL` | 상품 추천 모델 | `gpt-5-mini` |
| `TTS_MODEL` / `TTS_VOICE` | TTS 모델 / 음성 | `tts-1` / `nova` |
| `STT_MODEL` / `STT_LANGUAGE` | STT 모델 / 언어 코드 | `whisper-1` / `ko` |
| `DB_PATH` | SQLite 파일 경로 | `store.db` |
| `OPEN_HOUR` / `CLOSE_HOUR` | 영업 시작 / 보고서 생성 시각 (24h) | `9` / `22` |
| **카메라** | | |
| `TAPO_RTSP_URL` | 설정 시 1번 구역에 RTSP IP 카메라 사용 | (없음) |
| `ZONE1_CAMERA_INDEX` | 1번 구역 USB 웹캠 인덱스 (RTSP 미설정 시) | `0` |
| `ZONE2_CAMERA_INDEX` / `ZONE3_CAMERA_INDEX` | 값이 있을 때만 해당 구역 카메라 활성화 | (없음) |
| `DEBUG_CAMERA` | 감지 오버레이 창 표시 (1/0) | `1` |
| `TRACE_CAMERA_DATA` | 데이터 흐름 트레이스 로그 (1/0) | `1` |
| **Arduino (데모)** | | |
| `ARDUINO_ENABLED` | Arduino 액추에이터 사용 (1/0, `pyserial` 필요) | `0` |
| `ARDUINO_PORT` / `ARDUINO_BAUD` | 시리얼 포트 / 보레이트 | `COM3` / `9600` |
| **외부 연동** | | |
| `ENV_CONTROL_API_URL` / `_KEY` | 환경 제어(온도/조명/팬) API | (없음) |
| `SMS_API_KEY` / `SENDER_PHONE` / `MANAGER_PHONE` | Solapi SMS 발송 | (없음) |
| `PUSH_WEBHOOK` | 앱 푸시 웹훅 URL | (없음) |

### 실행

```bash
python main.py
```

---

## 테스트

### 모듈별 수동 테스트

```bash
python test_modules.py
```

```
테스트할 모듈을 선택하세요:
  [1] 고객 응대 챗봇 (자동 텍스트)             ← 하드코딩 메시지 4개
  [2] 일일 리포트
  [3] 관리자 알림
  [4] 멀티에이전트 시나리오                    ← 오케스트레이터 / actuator 개입 여부 확인
  [5] 전체 실행 (1~4)
  [6] 고객 응대 챗봇 (마이크 입력, 인터랙티브)  ← Whisper STT + TTS 재생
```

[6]번 마이크 모드는 `sounddevice` + `soundfile`이 필요하다 (`pip install sounddevice soundfile`). Enter를 누르면 5초간 녹음 → Whisper로 한국어 전사 → customer_bot 응답 → TTS 자동 재생. Ctrl+C로 종료.

**[4] 멀티에이전트 시나리오** 테스트는 다음 3가지 경로를 검증한다:

| 시나리오 | 입력 | 기대 경로 |
|---------|------|---------|
| 단일 구역 요청 | "온도 낮춰줘" | dispatch → customer → reconcile(no-op) → actuator → END |
| 전체 구역 요청 | "전체 구역 온도 22도로" | dispatch → customer → reconcile(LLM 판단) → actuator → END |
| 고위험 불확실 감지 | confidence=0.72, severity=high | dispatch → safety → reconcile(알림 발송) → actuator → END |

### 웹 시각화 (테스트 진행 중 흐름 확인)

[web/app.py](web/app.py) — FastAPI 서버 + 단일 페이지(`web/static/index.html`)로 시나리오 버튼을 누르면 `facility_graph`를 실제 실행하고 노드 활성화·실시간 로그를 SSE로 stream해서 보여준다. actuator는 모킹되어 SMS/온도 API 미호출.

```bash
pip install -r requirements.txt          # fastapi, uvicorn 포함
py -m uvicorn web.app:app --reload --port 8000
# 브라우저: http://localhost:8000
```

지원 시나리오: 고객 단일/크로스존, 안전 명확/불확실, 일일 보고서, 상권 인사이트(서울-강남구). 활성 노드는 보라색 펄스, 통과한 노드는 초록 테두리. 우측 패널에 `[AGENT: ...]` 로그와 최종 state 요약이 표시된다.

#### test_agent → 웹 실시간 표시

`test_agent.py --web-url`로 돌리면 웹 서버에 broadcast되어 같은 페이지에 실시간으로 표시된다. 별도 브라우저 새로고침 불필요.

```bash
# 터미널 1
py -m uvicorn web.app:app --reload --port 8000

# 터미널 2 (브라우저는 http://localhost:8000 열어둔 채)
py test_agent.py --seed --web-url http://localhost:8000 --delay 1.5
py test_agent.py --runs 8 --web-url http://localhost:8000 --delay 1.0
```

`--delay`는 시나리오 간 대기 초 — 웹에서 단계별로 보기 편하게. 각 이벤트에 출처 배지(web / test_agent)가 표시되어 누가 트리거한 흐름인지 구분 가능.

### 랜덤 시나리오 테스트 에이전트

`test_agent.py`가 LLM(`gpt-5-mini`)으로 다양한 입력 시나리오를 생성하고 그래프 라우팅·상태 키를 자동 검증한다. actuator의 부수효과는 모킹되므로 SMS/온도 제어 API 미설정으로도 실행 가능하다.

```bash
python test_agent.py                  # LLM 생성 5개
python test_agent.py --runs 10        # LLM 생성 10개
python test_agent.py --seed           # 내장 시드 6개 (LLM 호출 없음)
python test_agent.py --seed --verbose # 실패 시 캡처 stdout 출력
```

검증 항목: `[AGENT: ...]` 로그 순서가 `orchestrator_dispatch → (safety|customer|report) → orchestrator_reconcile → actuator` 와 일치하는지, `cross_zone_request`/`conflict_detected`/`report_text` 등 트리거 의존 플래그가 기대대로 채워졌는지.

---

## 주요 설계 결정

- **결정/실행 분리** — 에이전트 노드는 판단만 하고 `pending_actions`로 의도를 발행, `actuator` 노드가 부수효과를 전담. 결정 로직을 외부 API 호출 없이 단위 테스트 가능
- **dispatch / reconcile 분리** — 모든 진입은 `orchestrator_dispatch`를 거쳐 전문 에이전트로 분배되고, 전문 에이전트 실행 후 `orchestrator_reconcile`이 크로스존·충돌·이상 정책을 일괄 판단. dispatch는 LLM 미사용, reconcile만 LLM 호출
- **모델 자율 에스컬레이션** — 안전 에이전트가 confidence 보고 스스로 gpt-5(flagship)로 재판단
- **Human-in-the-loop** — 에이전트 간 판단 충돌 시 관리자에게 알림, AI가 최종 결정하지 않음
- **누적 액션 리듀서** — `pending_actions`는 `operator.add` 리듀서로 여러 노드(customer→reconcile)의 발행이 합쳐짐
- **기존 async 구조 유지** — LangGraph `ainvoke()`가 asyncio 루프에 자연스럽게 통합
- **Responses API 통일** — 전체 LLM 호출을 `client.responses.create`로 통일. 응답 파싱이 모두 `raw.output_text`로 단순화되고, `service_tier`로 호출 우선순위를 명시적으로 제어
- **이벤트 브로커 패턴 (웹 시각화)** — `web/app.py`가 단일 broadcast 채널을 두고 페이지 버튼·외부 테스트 에이전트의 그래프 실행 이벤트를 모두 같은 SSE 채널로 fan-out
- **스키마 무관 인사이트 엔진** — 인사이트/추천 노드는 `district_profile` dict 전체를 JSON으로 LLM에 넘겨, 데이터 스키마 변경 시 집계 함수(`aggregate_personas`)만 수정하면 그래프·노드·API는 불변. 같은 그래프 구조를 제어 시스템에서 데이터 인텔리전스로 재활용
- **개인정보 보호 내장** — 카메라단이 위험 프레임을 외부 VLM/저장소로 보내기 전 `apply_face_mosaic()`로 얼굴을 픽셀화하고, 분석용 임시 파일은 호출 직후 즉시 삭제
- **점진적 성능 저하(graceful degradation)** — MediaPipe 모델·카메라·Arduino 보드가 없어도 폴백(Haar Cascade / 감지 비활성 / mock 모드)으로 나머지 파이프라인은 그대로 동작
