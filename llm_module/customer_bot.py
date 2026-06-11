"""
고객 응대 챗봇 — LangGraph facility_graph 기반.
- 고객 요청을 customer 에이전트 노드로 라우팅
- 크로스존 요청은 오케스트레이터가 자동 처리
- closing_notice / extension_offer는 단순 TTS라 그래프 미사용
"""
import pathlib
from llm_module.tts import _tts

AUDIO_DIR = pathlib.Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)


async def respond(
    user_message: str,
    zone_id: str,
    context: dict,
    tts: bool = True,
    all_zone_ids: list[str] | None = None,
    persona: dict | None = None,
) -> dict:
    """
    user_message: 고객이 입력한 텍스트 or 음성 인식 결과
    context: {
        "customer_name": str,
        "visit_count": int,
        "current_temp": float,
        "remaining_min": int,
        "reserved_min": int,
    }
    persona: Nemotron-Personas-Korea 페르소나 dict (없으면 None).
             persona_manager.get_random_persona() 등으로 생성.
    반환: {"message": str, "audio_path": str | None, "action": dict | None}
    """
    from llm_module.graph import facility_graph
    from llm_module.state import make_customer_state

    state = make_customer_state(
        zone_id=zone_id,
        all_zone_ids=all_zone_ids or [zone_id],
        user_message=user_message,
        customer_context=context,
        tts_enabled=tts,
        customer_persona=persona,
    )

    result_state = await facility_graph.ainvoke(state)
    return result_state.get("bot_response") or {}


async def closing_notice(zone_id: str, remaining_min: int, context: dict) -> dict:
    """이용 종료 N분 전 자동 안내 (그래프 미사용 — 단순 TTS)"""
    msg = f"안내 말씀드립니다. {remaining_min}분 후 이용 시간이 종료됩니다. 연장을 원하시면 '연장'이라고 말씀해주세요."
    filename = f"closing_{zone_id}_{remaining_min}.mp3"
    audio_path = str(await _tts(msg, filename))
    return {"message": msg, "audio_path": audio_path, "action": None}


async def extension_offer(zone_id: str, remaining_min: int, context: dict) -> dict:
    """이용 시간 종료 임박 시 연장 제안"""
    return await respond(
        f"이용 시간이 {remaining_min}분 남았습니다. 연장을 안내해주세요.",
        zone_id,
        context,
    )
