# 무인 매장 AI 관리 시스템

LangGraph 기반 멀티에이전트 아키텍처로 구현된 무인 매장(무인 카페·편의점 등) 통합 관리 시스템.
영상 안전 감지, 고객 응대, 일일 보고서를 역할별 전문 에이전트가 처리하며, 오케스트레이터가 에이전트 간 협력을 조율한다.
모든 에이전트는 **부수효과 없는 "결정 노드"**이며, 실제 실행(알림·환경 제어·TTS)은 그래프 말단의 `actuator` 노드가 전담한다.

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
| 실행(actuator) | `agents/actuator.py` | — (LLM 미사용) | 결정 노드가 발행한 `pending_actions` 일괄 실행 |

> 결정 노드(안전/고객/보고서/오케스트레이터)는 알림·온도 제어·TTS 같은 부수효과를 직접 호출하지 않는다.
> 대신 실행 의도를 `state["pending_actions"]`에 적고, `actuator` 노드가 그래프 말단에서 이를 일괄 실행한다.

### 라우팅 규칙

```
START                  → [orchestrator_dispatch] : 항상
orchestrator_dispatch  → [safety|customer|report] : state["trigger_type"]에 따라 분배
safety / customer / report → [orchestrator_reconcile] : 항상
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
무인골프/
├── main.py                        # 진입점 — 구역 루프 + 보고서 루프
├── test_modules.py                # 모듈별 수동 테스트 (메뉴 [1]~[6])
├── test_agent.py                  # 랜덤 시나리오 자동 검증 에이전트
├── requirements.txt
├── .env.example                   # 환경변수 템플릿
│
├── llm_module/
│   ├── graph.py                   # LangGraph StateGraph 정의
│   ├── state.py                   # FacilityState 스키마 + 팩토리 함수
│   ├── state_machine.py           # VLM 빈도 제어 / cooldown (구역별)
│   ├── customer_bot.py            # 고객봇 public API (graph 경유)
│   ├── vlm_analyzer.py            # OpenAI Vision API 래퍼
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
│       └── actuator.py            # 실행 노드 (부수효과 전담)
│
├── web/                           # 실시간 흐름 시각화 (FastAPI + SSE)
│   ├── app.py                     # broadcast 채널 / /trigger / /events / /stream
│   └── static/index.html          # 단일 페이지 (입력/그래프/응답 실시간 표시)
│
├── db/
│   └── models.py                  # SQLite 스키마 + 쿼리
└── opencv/
    ├── bridge.py                  # OpenCV ↔ asyncio 어댑터
    └── face_detection.py          # MediaPipe 감지
```

---

## 설치 및 실행

### 요구사항

- Python 3.11+
- OpenAI API 키
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

주요 환경변수:

| 변수 | 설명 | 기본값 |
|-----|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 | 필수 |
| `VISION_MODEL` | 안전 감지 1차 모델 | `gpt-5-mini` |
| `CUSTOMER_MODEL` | 고객봇 모델 | `gpt-5-mini` |
| `REPORT_MODEL` | 일일 보고서 모델 | `gpt-5-mini` |
| `ESCALATION_MODEL` | 안전 감지 에스컬레이션 모델 | `gpt-5` |
| `ORCHESTRATOR_MODEL` | 오케스트레이터 reconcile 모델 | `gpt-5-mini` |
| `TTS_MODEL` | TTS 모델 | `tts-1` |
| `TTS_VOICE` | TTS 음성 | `nova` |
| `STT_MODEL` | STT(마이크 입력) 모델 | `whisper-1` |
| `STT_LANGUAGE` | STT 언어 코드 | `ko` |
| `DB_PATH` | SQLite 파일 경로 | `golf.db` |
| `CLOSE_HOUR` | 보고서 생성 시각 (24h) | `22` |

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
  [7] STT 마이크 단독 테스트                   ← 마이크 → stt.py → 텍스트 확인
  [8] STT 영상/오디오 파일 테스트              ← 파일 음성 → stt.py → 텍스트 확인
```

[6]번 마이크 모드는 `sounddevice` + `soundfile`이 필요하다 (`pip install sounddevice soundfile`). Enter를 누르면 5초간 녹음 → Whisper로 한국어 전사 → customer_bot 응답 → TTS 자동 재생. Ctrl+C로 종료.

[7]번은 customer bot 없이 마이크 입력이 `llm_module/stt.py`를 통해 정상 전사되는지만 확인한다. 마이크 장치 목록을 출력하고 특정 장치 ID/이름을 선택할 수 있다.

[8]번은 저장된 영상/오디오 파일에서 음성을 전사한다. 영상 파일(`mp4`, `mov`, `avi` 등)은 로컬에 `ffmpeg`가 설치되어 있어야 오디오를 추출할 수 있다. 라이브 RTSP 영상의 실시간 오디오는 별도 캡처 단계가 필요하므로 현재는 저장된 파일 테스트용이다.

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

지원 시나리오: 고객 단일/크로스존, 안전 명확/불확실, 일일 보고서. 활성 노드는 보라색 펄스, 통과한 노드는 초록 테두리. 우측 패널에 `[AGENT: ...]` 로그와 최종 state 요약이 표시된다.

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

## Camera → AI Event → Arduino Actuator Demo

이 데모는 실제 물리 제어 경로를 확인하기 위한 최소 통합 테스트다.

```
Tapo TC70 local RTSP
  → Python/OpenCV
  → mock AI event
  → state machine
  → USB Serial
  → Arduino Uno
  → built-in LED ON/OFF
```

Tapo TC70은 로컬 RTSP 스트림을 제공하고, Python/OpenCV가 노트북에서 영상을 읽는다. 포트포워딩은 사용하지 않는다. 노트북이 카메라와 Arduino 사이의 브리지 역할을 한다.

현재는 `e` 키를 눌러 occlusion/sweat 감지를 시뮬레이션한다. state machine이 이벤트를 받으면 actuator가 USB Serial로 Arduino에 `ALERT`를 보내고, 현재 Arduino 펌웨어에서는 built-in LED가 켜진다. `r` 키를 누르면 `NORMAL`을 보내 LED를 끈다.

나중에는 같은 `ALERT` 명령이 built-in LED 대신 relay/fan 같은 실제 환경 제어 장치를 제어할 수 있다.

### Setup

`.env`에 카메라와 Arduino 설정을 넣는다:

```env
TAPO_RTSP_URL=rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream2
ARDUINO_ENABLED=1
ARDUINO_PORT=COM3
ARDUINO_BAUD=9600
```

Arduino가 다른 포트에 연결되어 있으면 `ARDUINO_PORT`만 변경한다. Arduino 연결에 실패해도 데모는 mock mode로 계속 실행된다.

### Run

```bash
pip install -r requirements.txt
python camera_arduino_demo.py
```

### Controls

| Key | Action |
|-----|--------|
| `e` | Simulate occlusion/sweat event and send `ALERT` |
| `r` | Reset Arduino to `NORMAL` |
| `s` | Save screenshot as `tapo_demo_frame.jpg` |
| `q` | Quit cleanly and send `NORMAL` before shutdown |

This proves the physical environment-control path:

```text
Camera → AI Event → State Machine → Actuator → Arduino
```

---

## Unmanned Store Environment-Control Latency Demo

This demo does not require the camera. It generates mock detection events that simulate unmanned-store environment situations such as customer discomfort, heat/sweat discomfort, occlusion, abnormal behavior, or an unsafe environment state.

The purpose is to measure latency across the AI-control pipeline:

```text
mock detection → signal queue → state machine → OpenAI AI agent decision → Arduino actuator
```

Arduino Uno currently represents the physical actuator. `ALERT` turns on the built-in LED, and `NORMAL` turns it off. Later, the same actuator command can control a relay, fan, motor PWM, IR AC control, lighting, ventilation, or other unmanned-store environment-control hardware.

Configuration is read from `.env`. If `.env` does not exist, the script prints a warning and uses safe defaults.
By default, the AI decision step calls OpenAI using `OPENAI_API_KEY` and `LATENCY_AGENT_MODEL` / `ORCHESTRATOR_MODEL` / `CUSTOMER_MODEL` from `.env`. Use `--agent-mock` only when you want an offline local-rule test.

### Setup

```env
ARDUINO_ENABLED=1
ARDUINO_PORT=COM3
ARDUINO_BAUD=9600
MOCK_CAMERA_ENABLED=1
MOCK_EVENT_INTERVAL_SEC=3
LATENCY_OUTPUT_CSV=latency_results.csv
LATENCY_OUTPUT_JSON=latency_results.json
OPENAI_AGENT_ENABLED=1
LATENCY_AGENT_MODEL=gpt-5-mini
```

### Run

```bash
pip install -r requirements.txt
# edit .env and set ARDUINO_PORT if needed
python mock_latency_demo.py --events 5 --interval 2
```

If Arduino is not connected:

```bash
python mock_latency_demo.py --events 5 --interval 2 --arduino-mock
```

If OpenAI should not be called:

```bash
python mock_latency_demo.py --events 5 --interval 2 --agent-mock
```

Expected output includes per-event latency:

```text
[MOCK DETECTION] Store event generated: event_0001
[STATE MACHINE] Received store event: event_0001
[AGENT] Decision: ALERT / customer discomfort or unsafe environment state detected (openai:gpt-5-mini)
[ARDUINO SEND] ALERT
[ARDUINO RESPONSE] OK ALERT LED ON
[LATENCY] event_0001 total=245.3ms actuator_round_trip=31.2ms
```

Output files:

- `latency_results.csv`
- `latency_results.json`

---

## Tapo TC70 Camera Integration

카메라 → 얼굴/손 감지 → 상태 머신 → LangGraph 안전 에이전트 파이프라인.

```
Tapo RTSP (stream2)
  → opencv/bridge.py (face + hand occlusion detection)
  → TriggerSignals (sweat_wiping, person_count)
  → llm_module/state_machine.py (VLM confirm)
  → LangGraph safety agent → actuator (alert, temperature)
```

고객 주문/응대는 별도 경로: `customer_bot.respond()` 또는 `test_modules.py [1]`.

### Setup

1. `.env`에 RTSP URL 추가 (포트포워딩 불필요, 같은 Wi-Fi):

   ```env
   TAPO_RTSP_URL=rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream2
   ```

   Tapo 앱: **Camera Settings → Advanced Settings → Camera Account**

2. MediaPipe hand model 다운로드 (프로젝트 루트에 저장):

   ```bash
   curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   ```

3. 의존성 설치 후 실행:

   ```bash
   pip install -r requirements.txt
   python main.py
   ```

   디버그 창에서 `q`로 bridge 창만 닫을 수 있다. `DEBUG_CAMERA=0`이면 창 없이 실행.

### Standalone camera test (no LangGraph)

RTSP 연결만 먼저 확인:

```bash
python camera_bridge_demo.py
```

| Key | Action |
|-----|--------|
| `e` | Mock detection event |
| `s` | Save screenshot |
| `q` | Quit |

---

## 주요 설계 결정

- **결정/실행 분리** — 에이전트 노드는 판단만 하고 `pending_actions`로 의도를 발행, `actuator` 노드가 부수효과를 전담. 결정 로직을 외부 API 호출 없이 단위 테스트 가능
- **dispatch / reconcile 분리** — 모든 진입은 `orchestrator_dispatch`를 거쳐 전문 에이전트로 분배되고, 전문 에이전트 실행 후 `orchestrator_reconcile`이 크로스존·충돌·이상 정책을 일괄 판단. dispatch는 LLM 미사용, reconcile만 LLM 호출
- **모델 자율 에스컬레이션** — 안전 에이전트가 confidence 보고 스스로 gpt-5(flagship)로 재판단
- **Human-in-the-loop** — 에이전트 간 판단 충돌 시 관리자에게 알림, AI가 최종 결정하지 않음
- **누적 액션 리듀서** — `pending_actions`는 `operator.add` 리듀서로 여러 노드(customer→reconcile)의 발행이 합쳐짐
- **기존 async 구조 유지** — LangGraph `ainvoke()`가 asyncio 루프에 자연스럽게 통합
- **이벤트 브로커 패턴 (웹 시각화)** — `web/app.py`가 단일 broadcast 채널을 두고 페이지 버튼·외부 테스트 에이전트의 그래프 실행 이벤트를 모두 같은 SSE 채널로 fan-out
