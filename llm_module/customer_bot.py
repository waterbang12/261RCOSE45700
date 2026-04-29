"""
고객 응대 챗봇.
- 상황 인식 (예약정보 + 센서 + 게임상태) 기반 자연어 응답
- 자연어 명령 → 환경 제어 API 호출
- 웹/음성 동시 지원 (텍스트 반환 + TTS 파일 경로 반환)
"""
import os
import asyncio
import json
import pathlib
from openai import OpenAI
from llm_module.temperature_controller import apply_customer_pref, handle as temp_handle
from llm_module.coaching_engine import _tts

client = OpenAI()
AUDIO_DIR = pathlib.Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """
당신은 무인 스크린골프장의 AI 안내 도우미입니다.
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


async def respond(
    user_message: str,
    bay_id: str,
    context: dict,
    tts: bool = True,
) -> dict:
    """
    user_message: 고객이 입력한 텍스트 or 음성 인식 결과
    context: {
        "customer_name": str,
        "visit_count": int,
        "current_temp": float,
        "current_hole": int,
        "remaining_min": int,
        "reserved_min": int,
    }
    반환: {"message": str, "audio_path": str | None, "action": dict | None}
    """
    context_str = json.dumps(context, ensure_ascii=False)
    user_content = f"[현재 상황]\n{context_str}\n\n[고객 요청]\n{user_message}"

    raw = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=200,
    )

    data = json.loads(raw.choices[0].message.content)
    message = data.get("message", "")
    action  = data.get("action")

    # 환경 제어 실행
    if action and action.get("type") != "none":
        await _execute_action(action, bay_id, context)

    audio_path = None
    if tts and message:
        filename = f"bot_{bay_id}_{abs(hash(message))}.mp3"
        audio_path = str(await _tts(message, filename))

    print(f"[{bay_id}][BOT] {message}")
    return {"message": message, "audio_path": audio_path, "action": action}


async def _execute_action(action: dict, bay_id: str, context: dict) -> None:
    action_type = action.get("type")
    value       = action.get("value")

    if action_type == "temperature":
        current = context.get("current_temp", 24.0)
        delta   = float(value) - current
        await temp_handle(bay_id, current, 0, reason="고객 요청", target_temp_delta=delta)

    elif action_type == "fan":
        pass  # 환경 API 직접 호출 (temperature_controller 확장 시 추가)

    elif action_type == "light":
        pass


async def closing_notice(bay_id: str, remaining_min: int, context: dict) -> dict:
    """이용 종료 N분 전 자동 안내"""
    msg = f"안내 말씀드립니다. {remaining_min}분 후 이용 시간이 종료됩니다. 연장을 원하시면 '연장'이라고 말씀해주세요."
    filename = f"closing_{bay_id}_{remaining_min}.mp3"
    audio_path = str(await _tts(msg, filename))
    return {"message": msg, "audio_path": audio_path, "action": None}


async def extension_offer(bay_id: str, current_hole: int, context: dict) -> dict:
    """게임 진행 상태 기반 연장 제안"""
    return await respond(
        f"현재 {current_hole}홀 진행 중이고 시간이 거의 끝나갑니다. 연장을 안내해주세요.",
        bay_id,
        context,
    )
