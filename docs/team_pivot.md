# 프로젝트 방향 전환 — 팀 역할 분담

> **작성일:** 2026-06-03  
> **작성자:** Lee (총괄 리더)

---

## 무엇이 바뀌는가

| 기존 시스템 | 새 방향 |
|---|---|
| 실시간 무인 매장 AI 제어 (안전 감지, 고객봇, Arduino 작동) | 매장 운영자를 위한 **데이터 인텔리전스 플랫폼** |
| 고객이 직접 사용하는 키오스크 | **B2B — 운영자/창업자가 사용하는 분석 대시보드** |
| 카메라 → 이벤트 → 액추에이터 파이프라인 | 인구통계 + 행동 데이터 → 입지·재고 의사결정 지원 |
| Arduino LED = 핵심 데모 | Arduino = 데모용 보조 요소 (유지하되 핵심 아님) |

**핵심:** 기존 LangGraph 멀티에이전트 구조는 그대로 살린다.  
에이전트가 판단하는 내용만 바뀐다.

- 기존: "지금 당장 팬을 켜야 하나?"
- 변경: "이 상권에 바나나를 얼마나 입고해야 하나?"

---

## 역할 분담

---

### 👤 Lee — 총괄 리더 + 데이터 아키텍처

**할 일**

- 한국 인구 기반 합성 페르소나 데이터셋 설계 및 생성
  - 52M명 수준, 연령대·직업·거주지·소비패턴·이동패턴 포함
  - 페르소나 → 시간대별 무인 매장 이용 이벤트 시뮬레이션 (트랜잭션 로그)
- 전체 데이터 스키마 정의 (다른 팀원 연결 기준점)
- 전체 아키텍처 방향 주도
- Arduino는 피치덱 데모용으로만 유지

**산출물**

- `data/personas.parquet` 또는 `data/personas.csv` — 합성 페르소나
- `data/events.csv` — 시뮬레이션된 매장 이용 트랜잭션
- `docs/schema.md` — 공통 데이터 스키마 문서

---

### 👤 에이전트 담당 — AI 인사이트 엔진

**할 일**

- 기존 LangGraph 에이전트 구조를 인사이트 추천 엔진으로 전환
- 입력: 상권 인구통계 데이터 (연령대, 직업군, 시간대별 이동패턴)
- 출력: 자연어 추천 인사이트
  - 예시: "이 위치는 야간(22–01시) 세탁 수요가 높습니다"
  - 예시: "30–40대 직장인 밀집 지역 — 고단백 스낵·커피 재고 +30% 권장"
- `report_agent.py` → `insight_agent.py` 로 전환
- LLM이 데이터를 해석해 운영자에게 자연어 추천 제공

**산출물**

- `llm_module/agents/insight_agent.py`
- `llm_module/agents/recommendation_agent.py`
- FastAPI 엔드포인트: `POST /insight` (상권 데이터 입력 → 추천 반환)

---

### 👤 카메라/OpenCV 담당 — 데이터 수집 레이어

**할 일**

- 기존 카메라 안전감지 파이프라인은 **데모용으로만 유지** (피치덱 임팩트용)
- 메인 작업 전환: 외부 공공 데이터 수집 모듈 구축
  - 서울 생활인구 공공 API (통계청 / 서울 열린데이터 광장)
  - 대중교통 승하차 데이터 (시간대별 유동인구 프록시)
  - 날씨 API 연동 (소비패턴 상관관계 분석용)
- 수집한 외부 데이터를 Lee의 합성 데이터와 결합하여 분석 파이프라인에 공급

**산출물**

- `data_collection/public_api.py` — 공공 API 수집 모듈
- `data_collection/weather.py` — 날씨 데이터 연동
- `data_collection/transit.py` — 교통 데이터 연동
- FastAPI 엔드포인트: `GET /data/district/{code}` (상권 원시 데이터 반환)

---

### 👤 Frontend Developer — Operator Analytics Dashboard

**What changed**

| Before | After |
|---|---|
| Customer kiosk (voice input, AI response) | Operator analytics dashboard |
| End-user facing | B2B operator facing |
| Real-time control UI | Data-driven decision UI |

> Your existing kiosk work is **not wasted** — it becomes the "Phase 2 in-store layer" in the pitch deck and shows the full product vision.

---

**What to build**

#### Screen 1 — Location Intelligence Map

- Korea map with heatmap overlay (demographic density by district)
- Operator selects a district → platform returns:
  - Dominant demographic profile (age, occupation)
  - Peak usage hours
  - Recommended store type (laundromat, snack vending, etc.)

#### Screen 2 — Product Mix Recommender

- Given a selected location profile, display recommended inventory
- Example output:
  ```
  Gangnam-gu office district
  → Cold brew coffee       ▲ High demand (07:00–09:00)
  → High-protein snacks    ▲ High demand (12:00–13:00)
  → Phone charger cables   ▲ Steady demand
  → Bananas                ▲ Moderate demand
  ```

#### Screen 3 — Demand Forecast

- Time-of-day demand curves per store category per demographic
- Simple line charts per product/category

---

**API endpoints to connect to (backend will provide)**

```
GET  /data/district/{code}     — raw district demographic data
POST /insight                  — AI-generated insight for a location
GET  /recommend/products        — product mix recommendation
GET  /forecast/demand           — hourly demand forecast
```

**Tech stack** — same as before: FastAPI backend + React or plain HTML dashboard

---

## 전체 요약

| 팀원 | 역할 | 핵심 산출물 |
|---|---|---|
| Lee | 총괄 + 데이터 생성 | 합성 페르소나 데이터셋, 스키마 |
| 에이전트 담당 | AI 인사이트 엔진 | insight_agent, recommendation_agent |
| 카메라 담당 | 공공 데이터 수집 | public_api, transit, weather 모듈 |
| 프론트엔드 담당 | 운영자 대시보드 | 지도, 추천, 수요예측 화면 |

---

## 기존 코드 재활용 여부

| 모듈 | 재활용 여부 | 비고 |
|---|---|---|
| LangGraph 에이전트 구조 | ✅ 유지 | 에이전트 역할만 변경 |
| FastAPI 백엔드 | ✅ 유지 | 엔드포인트 추가 |
| Arduino 제어 | ⚠️ 데모용 유지 | 핵심 기능 아님 |
| 카메라/OpenCV | ⚠️ 데모용 유지 | 메인 작업은 공공 데이터로 전환 |
| 고객 키오스크 프론트엔드 | ✅ 피치덱에서 Phase 2로 활용 | 버리지 말 것 |
| STT/TTS | ⚠️ 선택적 유지 | 운영자 음성 쿼리 기능으로 확장 가능 |
