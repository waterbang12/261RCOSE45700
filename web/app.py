"""
멀티에이전트 흐름 시각화 웹 서버.

브라우저는 /stream(SSE) 한 채널만 구독하고, 그래프 실행 이벤트는
어떤 경로로 들어오든 같은 broadcast 채널로 fan-out된다.

이벤트 소스 두 가지:
  1) 웹 페이지의 버튼 → POST /trigger/{key} → 서버가 그래프 직접 실행
  2) 외부 클라이언트(test_agent.py --web-url) → POST /events

브라우저:
  GET /stream (text/event-stream) - 모든 이벤트 fan-out 수신

actuator의 부수효과(SMS/온도 제어/TTS)는 데모 안전을 위해 모킹된다.

실행:
  pip install -r requirements.txt
  py -m uvicorn web.app:app --reload --port 8000
  → http://localhost:8000
"""
import asyncio
import contextlib
import io
import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from db.models import init_db
from llm_module.state import (
    make_customer_state, make_safety_state, make_report_state,
)
from test_agent import install_actuator_mocks

ZONE_IDS = ["1번 구역", "2번 구역", "3번 구역"]

app = FastAPI(title="무인 매장 멀티에이전트 흐름")

STATIC_DIR = Path(__file__).parent / "static"

_started = False


@app.on_event("startup")
async def _startup():
    global _started
    if _started:
        return
    await init_db()
    install_actuator_mocks()
    _started = True
    print("[WEB] 시작 - actuator 모킹 완료, DB 초기화 완료")


# ── Broadcast 채널 ────────────────────────────────────────────────────────────

_subscribers: list[asyncio.Queue] = []
_subs_lock = asyncio.Lock()


async def broadcast(event: str, data: dict) -> None:
    """모든 SSE subscriber에게 이벤트 fan-out."""
    payload = (event, data)
    async with _subs_lock:
        dead: list[asyncio.Queue] = []
        for q in _subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


@app.get("/stream")
async def stream():
    """브라우저용 SSE 채널. 페이지 로드 시 자동 연결되어 모든 이벤트 수신."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    async with _subs_lock:
        _subscribers.append(queue)

    async def gen():
        try:
            yield _sse("hello", {"subscribers": len(_subscribers)})
            while True:
                event, data = await queue.get()
                yield _sse(event, data)
        except asyncio.CancelledError:
            pass
        finally:
            async with _subs_lock:
                if queue in _subscribers:
                    _subscribers.remove(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 외부 이벤트 수신 (test_agent → web) ──────────────────────────────────────

class ExternalEvent(BaseModel):
    event: str
    data: dict


@app.post("/events")
async def post_event(evt: ExternalEvent):
    if evt.event not in ("start", "log", "done", "error"):
        raise HTTPException(400, f"unknown event: {evt.event}")
    await broadcast(evt.event, evt.data)
    return {"ok": True}


# ── 페이지에서 직접 트리거 ───────────────────────────────────────────────────

def _customer_single():
    return make_customer_state(
        zone_id="1번 구역", all_zone_ids=ZONE_IDS,
        user_message="온도 좀 낮춰줘",
        customer_context={"customer_name": "데모", "visit_count": 1,
                          "current_temp": 27.0, "remaining_min": 30, "reserved_min": 60},
        tts_enabled=False,
    )


def _customer_cross():
    return make_customer_state(
        zone_id="1번 구역", all_zone_ids=ZONE_IDS,
        user_message="전체 구역 온도 22도로 맞춰줘",
        customer_context={"customer_name": "데모", "visit_count": 1,
                          "current_temp": 26.0, "remaining_min": 40, "reserved_min": 60},
        tts_enabled=False,
    )


def _safety_conflict():
    return make_safety_state(
        zone_id="2번 구역", all_zone_ids=ZONE_IDS,
        analysis_result={
            "detection_type": "fall_emergency", "detected": True,
            "confidence": 0.72, "severity": "high",
            "evidence": "Person lying still, unclear if intentional",
            "action_required": "Verify",
        },
        signals={"temperature": 25.0, "humidity": 60.0},
    )


def _safety_normal():
    return make_safety_state(
        zone_id="2번 구역", all_zone_ids=ZONE_IDS,
        analysis_result={
            "detection_type": "theft", "detected": True,
            "confidence": 0.92, "severity": "high",
            "evidence": "Hand moving merchandise toward bag",
            "action_required": "Notify manager",
        },
        signals={"temperature": 24.0, "humidity": 55.0},
    )


def _report():
    return make_report_state(zone_id="1번 구역", all_zone_ids=ZONE_IDS)


SCENARIOS = {
    "customer_single":  ("고객 - 단일 구역 온도 요청",              _customer_single),
    "customer_cross":   ("고객 - 전체 구역 요청 (크로스존)",         _customer_cross),
    "safety_conflict":  ("안전 - 고위험 불확실 (충돌 라우팅)",       _safety_conflict),
    "safety_normal":    ("안전 - 명확한 도난 감지",                  _safety_normal),
    "report":           ("일일 보고서 생성",                          _report),
}

_NODE_LOG = re.compile(r"\[AGENT:\s*(\w+)\]")


@app.get("/scenarios")
async def list_scenarios():
    return [{"key": k, "name": name} for k, (name, _) in SCENARIOS.items()]


@app.post("/trigger/{key}")
async def trigger(key: str):
    """페이지의 시나리오 버튼이 호출. 백그라운드에서 그래프 실행하면서 이벤트 broadcast."""
    if key not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario: {key}")

    state = SCENARIOS[key][1]()
    name = SCENARIOS[key][0]

    asyncio.create_task(_run_and_broadcast(name, state, source="web"))
    return {"ok": True}


def _input_from_state(state: dict) -> dict:
    """초기 state에서 입력 요약 추출 (웹 페이지 표시용)."""
    trigger = state.get("trigger_type")
    if trigger == "customer":
        return {
            "trigger_type": "customer",
            "zone_id": state.get("zone_id"),
            "user_message": state.get("user_message"),
            "context": state.get("customer_context") or {},
        }
    if trigger == "safety":
        return {
            "trigger_type": "safety",
            "zone_id": state.get("zone_id"),
            "analysis_result": state.get("analysis_result") or {},
            "signals": state.get("signals") or {},
        }
    if trigger == "report":
        return {
            "trigger_type": "report",
            "zone_id": state.get("zone_id"),
            "all_zone_ids": state.get("all_zone_ids") or [],
        }
    return {"trigger_type": trigger}


async def _run_and_broadcast(name: str, state: dict, source: str) -> None:
    """그래프 실행 + stdout 캡처 → broadcast."""
    from llm_module.graph import facility_graph

    input_summary = _input_from_state(state)

    await broadcast("start", {
        "name": name,
        "source": source,
        "input": input_summary,
    })
    loop = asyncio.get_running_loop()
    line_q: asyncio.Queue = asyncio.Queue()

    class _Writer(io.TextIOBase):
        def __init__(self):
            self._buf = ""

        def write(self, s: str) -> int:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    loop.call_soon_threadsafe(line_q.put_nowait, line)
            return len(s)

    async def emit_lines():
        while True:
            line = await line_q.get()
            if line is None:
                return
            m = _NODE_LOG.search(line)
            await broadcast("log", {
                "node": m.group(1) if m else None,
                "line": line,
                "source": source,
                **input_summary,
            })

    emit_task = asyncio.create_task(emit_lines())
    try:
        with contextlib.redirect_stdout(_Writer()):
            result = await facility_graph.ainvoke(state)
        await asyncio.sleep(0.05)  # 마지막 라인이 큐로 flush 되도록
        await broadcast("done", {
            "name": name, "source": source,
            "input": input_summary,
            "state": _serialize_state(result),
        })
    except Exception as e:
        await broadcast("error", {
            "name": name, "source": source,
            "message": f"{type(e).__name__}: {e}",
        })
    finally:
        line_q.put_nowait(None)
        try:
            await asyncio.wait_for(emit_task, timeout=1.0)
        except asyncio.TimeoutError:
            emit_task.cancel()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _serialize_state(s: dict) -> dict:
    keep = (
        "trigger_type", "zone_id",
        "user_message", "analysis_result", "signals",
        "bot_response", "orchestrator_decision",
        "cross_zone_request", "conflict_detected", "anomaly_detected",
        "report_text", "pending_actions",
    )
    out: dict = {}
    for k in keep:
        if k in s and s[k] not in (None, [], {}, False):
            out[k] = s[k]
    return out


# ── 정적 페이지 ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
