"""
멀티에이전트 시스템 랜덤 시나리오 테스트 에이전트.

LLM(gpt-5-mini)이 다양한 입력 시나리오를 생성하고, 각 시나리오를
facility_graph로 실행해 라우팅 순서와 state 키를 결정론적으로 검증한다.

부수효과(SMS/온도/TTS)는 모킹되므로 OPENAI_API_KEY만 있으면 충분하다.

실행:
  py test_agent.py [--runs N] [--seed]

옵션:
  --runs N   생성·실행할 시나리오 수 (기본 5)
  --seed     LLM 호출 없이 내장 시드 시나리오 6개만 사용 (오프라인)

환경변수:
  OPENAI_API_KEY    필수 (--seed 모드는 그래프 실행에서만 필요)
  TEST_AGENT_MODEL  시나리오 생성 모델 (기본 gpt-5-mini)
"""
import argparse
import asyncio
import contextlib
import io
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()


ZONE_IDS = ["1번 구역", "2번 구역", "3번 구역"]
_CROSS_ZONE_KEYWORDS = ["전체", "모든 구역", "전관", "전부", "전 매장", "매장 전체"]


# ── 부수효과 모킹 (actuator가 import한 alias를 교체) ──────────────────────────

def install_actuator_mocks() -> None:
    """actuator가 호출하는 외부 함수들을 no-op으로 교체.

    actuator.py는 모듈 로드 시점에 `from ... import handle as alert_handle` 형태로
    함수를 alias로 가져오므로, 원본 모듈을 패치해도 효과가 없다.
    actuator 모듈 자체의 attribute를 교체해야 한다.
    """
    import llm_module.agents.actuator as act

    async def _noop(*_a, **_kw):
        return None

    async def _noop_tts(_text, filename):
        return pathlib.Path(f"audio_cache/{filename}")

    act.alert_handle = _noop
    act._send_push = _noop
    act.temp_handle = _noop
    act.apply_customer_pref = _noop
    act._tts = _noop_tts


# ── 시나리오 ──────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    trigger_type: str
    payload: dict
    expected_route: list[str]
    expected_flags: dict[str, Any] = field(default_factory=dict)


def _build_expected(scenario: dict) -> tuple[list[str], dict[str, Any]]:
    """trigger_type 기반으로 기대 라우팅과 기대 플래그를 결정론적으로 산출."""
    trigger = scenario["trigger_type"]
    payload = scenario.get("payload") or {}
    flags: dict[str, Any] = {}

    if trigger == "customer":
        route = ["orchestrator_dispatch", "customer", "orchestrator_reconcile", "actuator"]
        msg = payload.get("user_message", "") or ""
        if any(k in msg for k in _CROSS_ZONE_KEYWORDS):
            flags["cross_zone_request_not_none"] = True
    elif trigger == "safety":
        route = ["orchestrator_dispatch", "safety", "orchestrator_reconcile", "actuator"]
        ar = payload.get("analysis_result") or {}
        if ar.get("severity") == "high" and ar.get("confidence", 1.0) < 0.80:
            flags["conflict_detected"] = True
    elif trigger == "report":
        route = ["orchestrator_dispatch", "report", "orchestrator_reconcile", "actuator"]
        flags["report_text_truthy"] = True
    else:
        route = []
    return route, flags


_SCENARIO_SYSTEM = """\
당신은 무인 매장 멀티에이전트 시스템을 테스트하는 시나리오 생성기입니다.
다양성을 위해 trigger_type을 골고루 섞고, edge case도 일부 포함하세요.

각 시나리오 schema:
- trigger_type="customer"
    payload = {
      "zone_id": "1번 구역"|"2번 구역"|"3번 구역",
      "user_message": <한국어 자연어 요청 - 온도/조명/팬/연장/안내 등 다양>,
      "context": {"customer_name": str, "visit_count": int,
                  "current_temp": float, "remaining_min": int, "reserved_min": int}
    }
    - 약 30%는 "전체 구역"/"모든 구역"/"매장 전체" 같은 크로스존 키워드 포함

- trigger_type="safety"
    payload = {
      "zone_id": <구역>,
      "analysis_result": {
        "detection_type": "sweat_wiping"|"theft"|"property_damage"|"fall_emergency",
        "detected": true,
        "confidence": <0.50~0.99>,
        "severity": "low"|"medium"|"high",
        "evidence": <짧은 영어 설명>,
        "action_required": <짧은 영어 설명>
      },
      "signals": {"temperature": <20~32>, "humidity": <40~80>}
    }
    - 약 30%는 severity="high" AND confidence < 0.80 (충돌 라우팅 유도)

- trigger_type="report"
    payload = {"zone_id": <구역>}

반드시 JSON 객체 하나만 응답:
{"scenarios": [{"name": <한국어 라벨>, "trigger_type": ..., "payload": {...}}, ...]}
"""


SEED_SCENARIOS: list[dict] = [
    {
        "name": "단일 구역 온도 낮춤",
        "trigger_type": "customer",
        "payload": {
            "zone_id": "1번 구역",
            "user_message": "온도 좀 낮춰줘",
            "context": {"customer_name": "김철수", "visit_count": 2,
                        "current_temp": 27.0, "remaining_min": 20, "reserved_min": 60},
        },
    },
    {
        "name": "전체 구역 온도 설정 (크로스존)",
        "trigger_type": "customer",
        "payload": {
            "zone_id": "1번 구역",
            "user_message": "전체 구역 온도 22도로 맞춰줘",
            "context": {"customer_name": "이영희", "visit_count": 1,
                        "current_temp": 26.0, "remaining_min": 40, "reserved_min": 60},
        },
    },
    {
        "name": "이용 시간 연장 요청",
        "trigger_type": "customer",
        "payload": {
            "zone_id": "2번 구역",
            "user_message": "30분 연장하고 싶어",
            "context": {"customer_name": "박민수", "visit_count": 5,
                        "current_temp": 24.0, "remaining_min": 5, "reserved_min": 60},
        },
    },
    {
        "name": "안전 - 명확한 도난 감지",
        "trigger_type": "safety",
        "payload": {
            "zone_id": "2번 구역",
            "analysis_result": {
                "detection_type": "theft",
                "detected": True,
                "confidence": 0.92,
                "severity": "high",
                "evidence": "Hand moving merchandise toward bag",
                "action_required": "Notify manager",
            },
            "signals": {"temperature": 24.0, "humidity": 55.0},
        },
    },
    {
        "name": "안전 - 고위험 불확실 낙상",
        "trigger_type": "safety",
        "payload": {
            "zone_id": "3번 구역",
            "analysis_result": {
                "detection_type": "fall_emergency",
                "detected": True,
                "confidence": 0.72,
                "severity": "high",
                "evidence": "Person lying still, unclear if intentional",
                "action_required": "Verify",
            },
            "signals": {"temperature": 25.0, "humidity": 60.0},
        },
    },
    {
        "name": "일일 보고서",
        "trigger_type": "report",
        "payload": {"zone_id": "1번 구역"},
    },
]


async def generate_scenarios_llm(n: int) -> list[Scenario]:
    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("TEST_AGENT_MODEL", "gpt-5-mini")
    raw = await asyncio.to_thread(
        client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": _SCENARIO_SYSTEM},
            {"role": "user", "content": f"시나리오를 {n}개 생성해 주세요. trigger_type을 골고루 섞어주세요."},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(raw.choices[0].message.content)
    return _materialize(data.get("scenarios", [])[:n])


def _materialize(raw_list: list[dict]) -> list[Scenario]:
    out: list[Scenario] = []
    for s in raw_list:
        try:
            route, flags = _build_expected(s)
            if not route:
                print(f"[GEN] 알 수 없는 trigger 스킵: {s.get('trigger_type')!r}")
                continue
            out.append(Scenario(
                name=s.get("name", "(unnamed)"),
                trigger_type=s["trigger_type"],
                payload=s.get("payload") or {},
                expected_route=route,
                expected_flags=flags,
            ))
        except Exception as e:
            print(f"[GEN] 시나리오 스킵: {e!r}")
    return out


# ── 실행 / 검증 ───────────────────────────────────────────────────────────────

_NODE_LOG = re.compile(r"\[AGENT:\s*(\w+)\]")


def _extract_route(captured: str) -> list[str]:
    return _NODE_LOG.findall(captured)


async def _build_initial_state(s: Scenario) -> dict:
    from llm_module.state import (
        make_customer_state, make_safety_state, make_report_state,
    )
    payload = s.payload
    if s.trigger_type == "customer":
        return make_customer_state(
            zone_id=payload.get("zone_id", ZONE_IDS[0]),
            all_zone_ids=ZONE_IDS,
            user_message=payload.get("user_message", ""),
            customer_context=payload.get("context") or {},
            tts_enabled=False,
        )
    if s.trigger_type == "safety":
        return make_safety_state(
            zone_id=payload.get("zone_id", ZONE_IDS[0]),
            all_zone_ids=ZONE_IDS,
            analysis_result=payload.get("analysis_result") or {},
            signals=payload.get("signals") or {},
        )
    if s.trigger_type == "report":
        return make_report_state(
            zone_id=payload.get("zone_id", ZONE_IDS[0]),
            all_zone_ids=ZONE_IDS,
        )
    raise ValueError(f"unsupported trigger: {s.trigger_type}")


class _LineForwarder(io.TextIOBase):
    """stdout 라인을 누적 + 콜백으로 라인별 push."""

    def __init__(self, on_line):
        self._buf = ""
        self._captured: list[str] = []
        self._on_line = on_line

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._captured.append(line)
                try:
                    self._on_line(line)
                except Exception:
                    pass
        return len(s)

    @property
    def captured(self) -> str:
        return "\n".join(self._captured)


def _input_summary(s: Scenario) -> str:
    """콘솔용 한 줄 요약."""
    p = s.payload
    if s.trigger_type == "customer":
        return f"zone={p.get('zone_id','?')} msg=\"{p.get('user_message','')}\""
    if s.trigger_type == "safety":
        ar = p.get("analysis_result") or {}
        return (f"zone={p.get('zone_id','?')} det={ar.get('detection_type')} "
                f"conf={ar.get('confidence')} sev={ar.get('severity')}")
    if s.trigger_type == "report":
        return f"zone={p.get('zone_id','?')} (일일 리포트)"
    return str(p)


def _input_struct(s: Scenario) -> dict:
    """웹 broadcast용 구조화된 입력."""
    p = s.payload
    if s.trigger_type == "customer":
        return {
            "trigger_type": "customer",
            "zone_id": p.get("zone_id"),
            "user_message": p.get("user_message"),
            "context": p.get("context") or {},
        }
    if s.trigger_type == "safety":
        return {
            "trigger_type": "safety",
            "zone_id": p.get("zone_id"),
            "analysis_result": p.get("analysis_result") or {},
            "signals": p.get("signals") or {},
        }
    if s.trigger_type == "report":
        return {
            "trigger_type": "report",
            "zone_id": p.get("zone_id"),
            "all_zone_ids": ZONE_IDS,
        }
    return {"trigger_type": s.trigger_type, "payload": p}


async def run_scenario(s: Scenario, web_client=None) -> dict:
    """시나리오 하나 실행. web_client가 주어지면 매 라인을 broadcast.

    web_client: WebReporter 인스턴스 또는 None
    """
    from llm_module.graph import facility_graph

    state = await _build_initial_state(s)

    print(f"    입력: {_input_summary(s)}")

    if web_client:
        await web_client.send("start", {
            "name": s.name,
            "source": "test_agent",
            "trigger": s.trigger_type,
            "expected_route": s.expected_route,
            "input": _input_struct(s),
        })
    
    input_fields = _input_struct(s)

    def _on_line(line: str):
        if web_client is None:
            return
        m = _NODE_LOG.search(line)
        web_client.send_nowait("log", {
            "node": m.group(1) if m else None,
            "line": line,
            "source": "test_agent",
            **input_fields,
        })

    writer = _LineForwarder(_on_line)
    error: str | None = None
    final_state: dict | None = None
    with contextlib.redirect_stdout(writer):
        try:
            final_state = await facility_graph.ainvoke(state)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

    captured = writer.captured
    actual_route = _extract_route(captured)

    failures: list[str] = []
    if error:
        failures.append(f"실행 예외: {error}")
    if actual_route != s.expected_route:
        failures.append(f"라우팅 불일치")

    if final_state is not None:
        for flag in s.expected_flags:
            if flag == "cross_zone_request_not_none":
                if not final_state.get("cross_zone_request"):
                    failures.append("cross_zone_request 미설정")
            elif flag == "conflict_detected":
                if not final_state.get("conflict_detected"):
                    failures.append("conflict_detected=False (예상 True)")
            elif flag == "report_text_truthy":
                if not final_state.get("report_text"):
                    failures.append("report_text 비어있음")

    result_dict = {
        "name": s.name,
        "trigger": s.trigger_type,
        "input_summary": _input_summary(s),
        "passed": not failures,
        "failures": failures,
        "actual_route": actual_route,
        "expected_route": s.expected_route,
        "captured": captured,
    }

    if web_client:
        await web_client.send(
            "error" if error else "done",
            {
                "name": s.name,
                "source": "test_agent",
                "passed": not failures,
                "failures": failures,
                "actual_route": actual_route,
                "expected_route": s.expected_route,
                "input": _input_struct(s),
                "state": _slim_state(final_state) if final_state else {},
                "message": error,
            },
        )

    return result_dict


def _slim_state(s: dict | None) -> dict:
    if not s:
        return {}
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


# ── 웹 브로드캐스트 ────────────────────────────────────────────────────────────

class WebReporter:
    """test_agent → 웹 서버 /events 로 이벤트 fire-and-forget POST."""

    def __init__(self, base_url: str):
        import httpx
        self._url = base_url.rstrip("/") + "/events"
        self._client = httpx.AsyncClient(timeout=2.0)
        self._tasks: list[asyncio.Task] = []

    async def send(self, event: str, data: dict) -> None:
        try:
            await self._client.post(self._url, json={"event": event, "data": data})
        except Exception as e:
            print(f"[WEB] POST 실패(무시): {e}")

    def send_nowait(self, event: str, data: dict) -> None:
        """라인별 호출용 - 같은 루프에 task로 던지고 즉시 리턴."""
        task = asyncio.create_task(self.send(event, data))
        self._tasks.append(task)

    async def close(self):
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._client.aclose()


# ── 리포팅 ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], verbose: bool) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    by_trigger: dict[str, list[bool]] = {}
    for r in results:
        by_trigger.setdefault(r["trigger"], []).append(r["passed"])

    print("\n" + "=" * 70)
    print(f"테스트 결과: {passed}/{total} 통과")
    print("-" * 70)
    for trig, marks in by_trigger.items():
        print(f"  {trig:10} : {sum(marks)}/{len(marks)} 통과")
    print("=" * 70)

    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{mark}] {r['trigger']:8} | {r['name']}")
        print(f"     입력: {r.get('input_summary','')}")
        if not r["passed"]:
            print(f"     expected route: {' → '.join(r['expected_route'])}")
            print(f"     actual route:   {' → '.join(r['actual_route']) or '(empty)'}")
            for f in r["failures"]:
                print(f"     - {f}")
            if verbose:
                print("     --- captured stdout ---")
                for line in r["captured"].splitlines():
                    print(f"     | {line}")
    print()


# ── 진입점 ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="멀티에이전트 시스템 랜덤 시나리오 테스트")
    parser.add_argument("--runs", type=int, default=5, help="시나리오 수 (기본 5)")
    parser.add_argument("--seed", action="store_true", help="LLM 미사용 - 내장 시드 시나리오만")
    parser.add_argument("--verbose", action="store_true", help="실패 시 캡처된 stdout 출력")
    parser.add_argument("--web-url", default=None,
                        help="웹 서버 URL (예: http://localhost:8000). 지정 시 매 라인을 broadcast하여 웹 페이지에 실시간 표시.")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="시나리오 간 대기 초 (웹에서 보기 편하게)")
    args = parser.parse_args()

    from db.models import init_db
    await init_db()

    install_actuator_mocks()
    print("[TEST AGENT] actuator 부수효과 모킹 완료")

    if args.seed:
        scenarios = _materialize(SEED_SCENARIOS[: args.runs] if args.runs else SEED_SCENARIOS)
        print(f"[TEST AGENT] 시드 시나리오 {len(scenarios)}개 로드")
    else:
        model = os.getenv("TEST_AGENT_MODEL", "gpt-5-mini")
        print(f"[TEST AGENT] {args.runs}개 시나리오 생성 중 (model={model})...")
        scenarios = await generate_scenarios_llm(args.runs)
        print(f"[TEST AGENT] {len(scenarios)}개 시나리오 생성 완료")

    if not scenarios:
        print("시나리오가 비었습니다.")
        sys.exit(2)

    web_client: WebReporter | None = None
    if args.web_url:
        web_client = WebReporter(args.web_url)
        print(f"[TEST AGENT] 웹 broadcast 활성화 → {args.web_url}/events")

    results = []
    try:
        for i, s in enumerate(scenarios, 1):
            print(f"--- [{i}/{len(scenarios)}] {s.trigger_type}: {s.name}")
            result = await run_scenario(s, web_client=web_client)
            results.append(result)
            if args.delay > 0 and i < len(scenarios):
                await asyncio.sleep(args.delay)
    finally:
        if web_client:
            await web_client.close()

    print_report(results, verbose=args.verbose)
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
