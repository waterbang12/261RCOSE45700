"""
고객봇 에이전트 노드 (결정 전용).
- 안전 우선순위를 시스템 프롬프트에 내재화 (안전 > 고객 편의 > 환경 최적화).
- "전체/모든 구역" 키워드 감지 시 cross_zone_request 설정 → 오케스트레이터 위임.
- 환경 제어·TTS 생성은 직접 실행하지 않고 pending_actions로 발행한다.
- customer_persona 제공 시 Nemotron-Personas-Korea 페르소나 기반 맞춤 응대.
"""
import asyncio
import json
import os
from openai import OpenAI

from llm_module.state import FacilityState
from llm_module.persona_manager import format_persona_for_prompt

client = OpenAI()

_BASE_SYSTEM_PROMPT = """
당신은 무인 매장(무인 카페·편의점 등)의 AI 안내 도우미입니다.
우선순위 원칙: 안전 > 고객 편의 > 환경 최적화.
안전과 관련된 요청이 있으면 반드시 안전을 최우선으로 처리하세요.
고객의 요청에 친절하고 간결하게 한국어로 답변하세요.
환경 제어(온도, 조명, 팬)가 필요한 경우 반드시 JSON action을 포함하세요.
답변은 두 부분으로 구성하세요:
1. "message": 고객에게 보여줄 자연스러운 안내 문장
2. "action": 실행할 환경 제어 명령 (없으면 null)

action 형식:
{
  "type": "temperature" | "fan" | "light" | "extend_time" | "none",
  "value": <값>
}

반드시 JSON으로만 응답하세요.
"""

_PERSONA_SYSTEM_SUFFIX = """
[고객 페르소나 정보]
{persona_text}

위 페르소나를 참고해 고객의 연령대·직업·취향에 맞는 어조와 내용으로 응대하세요.
예: 취미/음식 취향과 관련된 추천, 연령대에 맞는 존댓말 수위 조절 등.
페르소나 정보를 직접 언급하거나 개인정보를 노출하지 마세요.
"""

_CROSS_ZONE_KEYWORDS = ["전체", "모든 구역", "전관", "전부", "전 매장", "매장 전체"]


def _build_system_prompt(persona: dict | None) -> str:
    if not persona:
        return _BASE_SYSTEM_PROMPT
    persona_text = format_persona_for_prompt(persona)
    return _BASE_SYSTEM_PROMPT + _PERSONA_SYSTEM_SUFFIX.format(persona_text=persona_text)


async def customer_node(state: FacilityState) -> dict:
    print(f"[AGENT: customer] zone={state['zone_id']}")
    zone_id = state["zone_id"]
    user_message = state.get("user_message") or ""
    context = state.get("customer_context") or {}
    persona = state.get("customer_persona")

    system_prompt = _build_system_prompt(persona)
    context_str = json.dumps(context, ensure_ascii=False)
    user_content = f"[현재 상황]\n{context_str}\n\n[고객 요청]\n{user_message}\n반드시 JSON으로만 응답하세요."

    raw = await asyncio.to_thread(
        client.responses.create,
        model=os.getenv("CUSTOMER_MODEL", "gpt-5-mini"),
        instructions=system_prompt,
        input=user_content,
        text={"format": {"type": "json_object"}},
        service_tier="priority",
        store=False,
    )

    data = json.loads(raw.output_text)
    message = data.get("message", "")
    action = data.get("action")

    # 매장 전체 영향 요청 감지 → 오케스트레이터 위임
    cross_zone = None
    if any(kw in user_message for kw in _CROSS_ZONE_KEYWORDS):
        cross_zone = {"action": action}

    pending: list[dict] = []

    # 단일 구역 환경 제어 요청 → 실행 의도 발행 (cross-zone은 오케스트레이터가 처리)
    if action and action.get("type") != "none" and not cross_zone:
        temp_action = _build_temperature_action(action, context)
        if temp_action:
            pending.append(temp_action)

    # 음성 안내 → TTS 생성 의도 발행
    if message and state.get("tts_enabled", True):
        filename = f"bot_{zone_id}_{abs(hash(message))}.mp3"
        pending.append({"kind": "tts", "text": message, "filename": filename})

    print(f"[{zone_id}][BOT] {message}")
    return {
        "bot_response": {"message": message, "audio_path": None, "action": action},
        "cross_zone_request": cross_zone,
        "pending_actions": pending,
    }


def _build_temperature_action(action: dict, context: dict) -> dict | None:
    """환경 제어 명령을 actuator용 temperature action으로 변환."""
    if action.get("type") != "temperature":
        # fan, light → 환경 API 확장 시 추가
        return None
    try:
        target = float(action.get("value"))
    except (TypeError, ValueError):
        # LLM이 숫자가 아닌 value를 반환한 경우 무시
        return None
    current = context.get("current_temp", 24.0)
    return {
        "kind": "temperature",
        "temperature": current,
        "humidity": 0,
        "reason": "고객 요청",
        "target_temp_delta": target - current,
    }
