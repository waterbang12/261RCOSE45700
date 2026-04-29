import os
import asyncio
import httpx
from db.models import log_env

ENV_API_URL = os.getenv("ENV_CONTROL_API_URL", "http://localhost:8000")
ENV_API_KEY = os.getenv("ENV_CONTROL_API_KEY", "")

# 온도 제어 한도
TEMP_MIN = 18.0
TEMP_MAX = 30.0


async def _call_env_api(bay_id: str, payload: dict) -> None:
    headers = {"Authorization": f"Bearer {ENV_API_KEY}"} if ENV_API_KEY else {}
    async with httpx.AsyncClient(timeout=3.0) as c:
        r = await c.post(f"{ENV_API_URL}/bay/{bay_id}/environment", json=payload, headers=headers)
        r.raise_for_status()
    print(f"[ENV API][{bay_id}] {payload}")


async def handle(
    bay_id: str,
    temperature: float,
    humidity: float,
    reason: str,
    target_temp_delta: float = -1.0,
) -> None:
    """
    땀 감지 or 센서 임계값 초과 시 호출.
    온도 + 선풍기 + 조명을 묶음으로 제어한다.
    """
    new_temp = max(TEMP_MIN, temperature + target_temp_delta)

    payload = {
        "temperature": new_temp,
        "fan": "on",
        "light_level": "dim",   # 조명 약간 낮춤으로 체감 온도 하락 효과
        "reason": reason,
    }

    print(f"[{bay_id}][온도제어] 요청 → 온도 {new_temp}°C / 팬 ON / 조명 dim / 사유: {reason[:40]}")
    try:
        await _call_env_api(bay_id, payload)
    except Exception as e:
        print(f"[{bay_id}][온도제어] 실패 → {e}")

    await log_env(bay_id, new_temp, humidity, f"auto_cool: {reason[:50]}")


async def restore_standby(bay_id: str) -> None:
    """이용자 없을 때 절전 상태로 복귀"""
    payload = {
        "temperature": 26.0,
        "fan": "off",
        "light_level": "off",
        "reason": "standby",
    }
    try:
        await _call_env_api(bay_id, payload)
    except Exception as e:
        print(f"[ENV API ERROR][{bay_id}] standby: {e}")


async def apply_customer_pref(bay_id: str, pref_temp: float) -> None:
    """DB에 저장된 고객 선호 온도 적용 (VIP 입장 시)"""
    payload = {
        "temperature": max(TEMP_MIN, min(TEMP_MAX, pref_temp)),
        "fan": "auto",
        "light_level": "normal",
        "reason": "customer_preference",
    }
    try:
        await _call_env_api(bay_id, payload)
    except Exception as e:
        print(f"[ENV API ERROR][{bay_id}] pref: {e}")
